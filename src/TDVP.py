import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    from optax.contrib import muon as optax_muon
except ImportError:  # pragma: no cover - depends on optax version
    optax_muon = None

from src.grad import (
    _ModelWavefunctionView,
    _tree_add,
    tdvp_vmc_gradient,
    tdvp_vmc_trajectory_gradient,
)
from src.hamiltonian import TransverseIsingHamiltonian
from src.loss import tdvp_residual_loss
from src.sampler import (
    metropolis_hastings_sample,
    metropolis_hastings_trajectory,
    autoregressive_trajectory_sample,
)
from src.wavefunction import Wavefunction, tSpinNQS


@dataclass
class TrainingConfig:
    """Configuration for TDVP training."""

    # System
    N: int  # number of sites
    J: float  # ZZ coupling
    h: float  # transverse field

    # Model architecture
    Num_boxes: int
    emb_dim: int
    num_heads: int
    head_dim: int

    # Training
    learning_rate: float = 0.001
    optimizer_name: str = "adamw"
    adamw_b1: float = 0.9
    adamw_b2: float = 0.999
    adamw_eps: float = 1e-8
    weight_decay: float = 1e-4
    n_steps: int = 100
    batch_size: int = 32

    # Sampling
    n_samples_per_chain: int = 32
    burn_in: int = 100
    thinning: int = 1
    n_chains: int = 1
    initial_chain_configurations: Optional[Any] = None

    # Time
    t_initial: float = 0.0
    t_final: float = 1.0
    time_steps: int = 10

    # Initial Condition Anchoring (t=0)
    pretrain_steps: int = 0
    pretrain_lr: float = 0.005
    lambda_ic: float = 0.0

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 10

    # Random seed
    seed: int = 0


def _validate_training_config(config: TrainingConfig) -> None:
    """Validate training hyperparameters before starting a run."""
    if config.N <= 0:
        raise ValueError(f"N must be > 0, got {config.N}")
    if config.n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {config.n_steps}")
    if config.time_steps <= 0:
        raise ValueError(f"time_steps must be > 0, got {config.time_steps}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.optimizer_name.lower() not in ("adamw", "muon"):
        raise ValueError(
            f"Unsupported optimizer_name {config.optimizer_name!r}; only 'adamw' and 'muon' are currently supported."
        )
    if not 0.0 <= config.adamw_b1 < 1.0:
        raise ValueError(f"adamw_b1 must be in [0, 1), got {config.adamw_b1}")
    if not 0.0 <= config.adamw_b2 < 1.0:
        raise ValueError(f"adamw_b2 must be in [0, 1), got {config.adamw_b2}")
    if config.adamw_eps <= 0:
        raise ValueError(f"adamw_eps must be > 0, got {config.adamw_eps}")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {config.weight_decay}")
    if config.n_chains <= 0:
        raise ValueError(f"n_chains must be > 0, got {config.n_chains}")
    if config.n_samples_per_chain <= 0:
        raise ValueError(
            f"n_samples_per_chain must be > 0, got {config.n_samples_per_chain}"
        )
    if config.burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {config.burn_in}")
    if config.thinning <= 0:
        raise ValueError(f"thinning must be > 0, got {config.thinning}")
    if config.pretrain_steps < 0:
        raise ValueError(f"pretrain_steps must be >= 0, got {config.pretrain_steps}")
    if config.pretrain_lr <= 0:
        raise ValueError(f"pretrain_lr must be > 0, got {config.pretrain_lr}")
    if config.lambda_ic < 0:
        raise ValueError(f"lambda_ic must be >= 0, got {config.lambda_ic}")
    if config.initial_chain_configurations is not None:
        _validate_chain_configurations(
            config.initial_chain_configurations,
            n_chains=config.n_chains,
            n_sites=config.N,
        )


def _initialize_wavefunction(config: TrainingConfig, rng: jax.Array) -> tSpinNQS:
    """Construct a wavefunction instance from architecture fields in config."""
    return tSpinNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(int(rng[0])),
    )


def _x_polarized_mse_loss(
    model: nnx.Module,
    configurations: jnp.ndarray,
    n_sites: int,
    target_logp: float,
) -> jnp.ndarray:
    """MSE loss against X-polarized state: logp = target, phase = 0."""
    wf_view = _ModelWavefunctionView(model)
    logp, phi = wf_view(configurations, t=0.0)

    target_phi = 0.0

    loss_logp = jnp.mean((logp - target_logp) ** 2)
    loss_phi = jnp.mean((phi - target_phi) ** 2)

    return loss_logp + loss_phi


def _validate_chain_configurations(
    configurations: jnp.ndarray,
    *,
    n_chains: int,
    n_sites: int,
) -> jnp.ndarray:
    """Validate persistent Markov-chain states."""
    configs = jnp.asarray(configurations).astype(jnp.int32)
    if configs.ndim != 2:
        raise ValueError(
            f"Expected chain configurations with shape ({n_chains}, {n_sites}), got {configs.shape}"
        )
    if configs.shape != (n_chains, n_sites):
        raise ValueError(
            f"Expected chain configurations shape ({n_chains}, {n_sites}), got {configs.shape}"
        )
    if not isinstance(configs, jax.core.Tracer):
        if not jnp.all((configs == 0) | (configs == 1)):
            raise ValueError("Chain configurations must be binary bits in {0,1}.")
    return configs


def _default_metrics_history() -> Dict[str, list]:
    """Create empty metric storage for TDVP training."""
    return {
        "loss": [],
        "acceptance_rate": [],
        "grad_norm_pathwise": [],
        "grad_norm_covariance": [],
        "grad_norm_total": [],
        "ell_mean": [],
        "ell_std": [],
        "finite_loss": [],
        "finite_grads": [],
        "e_real_mean": [],
        "e_imag_mean": [],
        "time": [],
        "step": [],
    }


def _create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Build the optimizer configured for TDVP training."""
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "adamw":
        return optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.adamw_b1,
            b2=config.adamw_b2,
            eps=config.adamw_eps,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "muon":
        if optax_muon is None:
            raise RuntimeError(
                "Muon optimizer is not available in the installed optax package."
            )
        return optax_muon(
            learning_rate=config.learning_rate,
            ns_coeffs="standard",
            ns_steps=5,
            beta=0.95,
            eps=config.adamw_eps,
            weight_decay=config.weight_decay,
            adam_b1=config.adamw_b1,
            adam_b2=config.adamw_b2,
            adam_eps_root=config.adamw_eps,
            adam_weight_decay=config.weight_decay,
            muon_weight_dimension_numbers=None,
            consistent_rms=0.2,
        )
    raise ValueError(
        f"Unsupported optimizer_name {config.optimizer_name!r}; only 'adamw' and 'muon' are currently supported."
    )


def save_training_checkpoint(
    filepath: str,
    *,
    wf: Wavefunction,
    ham: TransverseIsingHamiltonian,
    config: TrainingConfig,
    metrics_history: Dict[str, list],
    global_step: int,
    completed_time_steps: int,
    current_time: Optional[float],
    chain_configurations: jnp.ndarray,
    rng: jax.Array,
    opt_state: Optional[Any] = None,
) -> None:
    """Save wavefunction parameters and training state for reload/resume."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    model_state = jax.tree_util.tree_map(
        lambda x: np.asarray(x),
        nnx.state(wf.model, nnx.Param),
    )
    payload = {
        "model_state": model_state,
        "optimizer_state": None
        if opt_state is None
        else jax.tree_util.tree_map(lambda x: np.asarray(x), opt_state),
        "hamiltonian": {
            "J": float(ham.J),
            "h": float(ham.h),
            "N": int(ham.N),
        },
        "config": asdict(config),
        "metrics_history": metrics_history,
        "global_step": int(global_step),
        "completed_time_steps": int(completed_time_steps),
        "current_time": None if current_time is None else float(current_time),
        "chain_configurations": np.asarray(chain_configurations, dtype=np.int32),
        "rng": np.asarray(rng, dtype=np.uint32),
    }

    with open(filepath, "wb") as f:
        pickle.dump(payload, f)


def load_training_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load a saved wavefunction and associated TDVP training state."""
    with open(filepath, "rb") as f:
        payload = pickle.load(f)

    config = TrainingConfig(**payload["config"])
    wf = tSpinNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(config.seed),
    )
    nnx.update(wf.model, payload["model_state"])

    ham = TransverseIsingHamiltonian(**payload["hamiltonian"])

    return {
        "wavefunction": wf,
        "hamiltonian": ham,
        "optimizer_state": None
        if payload.get("optimizer_state") is None
        else jax.tree_util.tree_map(jnp.asarray, payload["optimizer_state"]),
        "config": config,
        "metrics_history": payload["metrics_history"],
        "global_step": int(payload["global_step"]),
        "completed_time_steps": int(payload["completed_time_steps"]),
        "current_time": payload["current_time"],
        "chain_configurations": jnp.asarray(
            payload["chain_configurations"], dtype=jnp.int32
        ),
        "rng": jnp.asarray(payload["rng"], dtype=jnp.uint32),
    }


def _save_checkpoint(
    config: TrainingConfig,
    wf: Wavefunction,
    ham: TransverseIsingHamiltonian,
    metrics_history: Dict[str, list],
    global_step: int,
    completed_time_steps: int,
    current_time: float,
    chain_configurations: jnp.ndarray,
    rng: jax.Array,
    opt_state: Optional[Any],
    filepath: str,
):
    """Backward-compatible wrapper around the real checkpoint serializer."""
    save_training_checkpoint(
        filepath,
        wf=wf,
        ham=ham,
        config=config,
        metrics_history=metrics_history,
        global_step=global_step,
        completed_time_steps=completed_time_steps,
        current_time=current_time,
        chain_configurations=chain_configurations,
        rng=rng,
        opt_state=opt_state,
    )


def _apply_optimizer_update(
    model: nnx.Module,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    grad_state: Any,
) -> Any:
    """Apply an in-place optimizer update through the model parameter state."""
    params = nnx.state(model, nnx.Param)
    updates, new_opt_state = optimizer.update(grad_state, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    return new_opt_state


def train_loop(
    config: TrainingConfig,
    verbose: bool = True,
    *,
    initial_wavefunction: Optional[Wavefunction] = None,
    initial_hamiltonian: Optional[TransverseIsingHamiltonian] = None,
    initial_metrics_history: Optional[Dict[str, list]] = None,
    initial_global_step: int = 0,
    initial_chain_configurations: Optional[jnp.ndarray] = None,
    initial_opt_state: Optional[Any] = None,
    initial_rng: Optional[jax.Array] = None,
    start_time_index: int = 0,
) -> Dict[str, Any]:
    """Full training loop over time points and optimization steps.

    Args:
        config: TrainingConfig with all hyperparameters.
        verbose: Whether to print progress.

    Returns:
        training_result dict with final model, metrics, and config.
    """
    _validate_training_config(config)
    if start_time_index < 0 or start_time_index > config.time_steps:
        raise ValueError(
            f"start_time_index must be in [0, {config.time_steps}], got {start_time_index}"
        )

    # 1) Initialize model / RNG / Hamiltonian.
    if initial_rng is None:
        rng = jax.random.PRNGKey(config.seed)
        rng, rng_model = jax.random.split(rng)
    else:
        rng = jnp.asarray(initial_rng, dtype=jnp.uint32)
        rng_model = rng

    if initial_wavefunction is None:
        wf = _initialize_wavefunction(config, rng_model)
    else:
        wf = initial_wavefunction

    if initial_hamiltonian is None:
        ham = TransverseIsingHamiltonian(J=config.J, h=config.h, N=config.N)
    else:
        ham = initial_hamiltonian

    optimizer = _create_optimizer(config)
    if initial_opt_state is None:
        opt_state = optimizer.init(nnx.state(wf.model, nnx.Param))
    else:
        opt_state = initial_opt_state

    # --- Target logp for IC anchoring (Stability improvement) ---
    ic_target_logp = 0.0
    if config.pretrain_steps > 0 or config.lambda_ic > 0.0:
        if hasattr(wf.model, 'amp_model'):
            # Autoregressive models output exact probabilities. Target must be exactly uniform.
            ic_target_logp = float(-config.N * jnp.log(2.0))
            if verbose:
                print(
                    f"Initial condition anchoring exact AR target logp set to: {ic_target_logp:.6f}"
                )
        else:
            # Forward once with random bits to get a baseline mean logp for the initial model.
            rng, ic_target_rng = jax.random.split(rng)
            baseline_configs = jax.random.randint(
                ic_target_rng, (config.batch_size, config.N), 0, 2
            )
            initial_logp, _ = wf(baseline_configs, 0.0)
            ic_target_logp = float(jnp.mean(initial_logp))
            if verbose:
                print(
                    f"Initial condition anchoring baseline logp set to: {ic_target_logp:.6f}"
                )

    # 2) Pretraining Phase (Optional)
    if config.pretrain_steps > 0:
        if verbose:
            print(
                f"\n=== Pretraining initial condition (X-polarized) for {config.pretrain_steps} steps ==="
            )

        pretrain_optimizer = optax.adam(config.pretrain_lr)
        pretrain_opt_state = pretrain_optimizer.init(nnx.state(wf.model, nnx.Param))

        @nnx.jit
        def pretrain_step(model, os, keys):
            # Generate random configurations for uniform target distribution
            configs = jax.random.randint(keys, (config.batch_size, config.N), 0, 2)

            def loss_fn(m):
                return _x_polarized_mse_loss(m, configs, config.N, ic_target_logp)

            loss, grad = nnx.value_and_grad(loss_fn)(model)

            # Standard optimizer update
            params = nnx.state(model, nnx.Param)
            updates, new_os = pretrain_optimizer.update(grad, os, params)
            new_params = optax.apply_updates(params, updates)
            nnx.update(model, new_params)
            return loss, new_os

        for i in range(config.pretrain_steps):
            rng, pretrain_rng = jax.random.split(rng)
            loss_val, pretrain_opt_state = pretrain_step(
                wf.model, pretrain_opt_state, pretrain_rng
            )
            if verbose and (i + 1) % max(1, config.pretrain_steps // 5) == 0:
                print(
                    f"  Pretrain step {i + 1}/{config.pretrain_steps}: MSE loss = {float(loss_val):.6f}"
                )

    # 3) Time schedule.
    times = jnp.linspace(config.t_initial, config.t_final, config.time_steps)

    # 4) Metrics tracking.
    metrics_history = (
        _default_metrics_history()
        if initial_metrics_history is None
        else initial_metrics_history
    )

    # 5) Training loop.
    global_step = int(initial_global_step)
    if initial_chain_configurations is not None:
        chain_configurations = _validate_chain_configurations(
            initial_chain_configurations,
            n_chains=config.n_chains,
            n_sites=ham.N,
        )
    elif config.initial_chain_configurations is None:
        chain_configurations = jnp.zeros((config.n_chains, ham.N), dtype=jnp.int32)
    else:
        chain_configurations = _validate_chain_configurations(
            config.initial_chain_configurations,
            n_chains=config.n_chains,
            n_sites=ham.N,
        )

    # Distribute chains across CPU/TPU cores using JAX SPMD.
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    n_devices = jax.device_count()
    if config.n_chains % n_devices != 0:
        # Fallback to single-device mesh if chains can't be evenly distributed
        mesh_devices = [jax.devices()[0]]
    else:
        mesh_devices = jax.devices()

    mesh = Mesh(mesh_devices, axis_names=("chains",))
    sharding = NamedSharding(mesh, PartitionSpec("chains"))
    chain_configurations = jax.device_put(chain_configurations, sharding)

    time_indices = list(range(start_time_index, config.time_steps))
    # Active time slices for this run
    active_times = times[jnp.array(time_indices)]

    @nnx.jit
    def jitted_trajectory_train_step(model, opt_st, configs, rng_val):
        wf_view = _ModelWavefunctionView(model)

        # 1) Sample spacetime trajectory (warm-started or autoregressive)
        # all_samples: (T_active, C, S, N)
        # next_configs: (C, N)
        rng_val, rng_sample = jax.random.split(rng_val)
        
        if hasattr(model, 'amp_model'):
            all_samples, all_stats, next_configs = autoregressive_trajectory_sample(
                wf=wf_view,
                times=active_times,
                n_sites=ham.N,
                batch_size=config.n_chains * config.n_samples_per_chain,
                key=rng_sample,
                n_chains=config.n_chains,
            )
        else:
            all_samples, all_stats, next_configs = metropolis_hastings_trajectory(
                wf=wf_view,
                initial_configurations=configs,
                times=active_times,
                n_sites=ham.N,
                n_samples=config.n_samples_per_chain,
                burn_in=config.burn_in,
                thinning=config.thinning,
                key=rng_sample,
            )

        # Reshape for unified gradient computation
        # T_active is the number of time slices being optimized jointly
        T_active = active_times.shape[0]
        batch_size_per_slice = config.n_chains * config.n_samples_per_chain
        all_configs_batch = all_samples.reshape((T_active, batch_size_per_slice, ham.N))

        # 2) Compute unified spacetime gradient
        grad_total, diag = tdvp_vmc_trajectory_gradient(
            ham=ham,
            wf=wf_view,
            all_configurations=all_configs_batch,
            times=active_times,
        )

        # Add sampling diagnostics (mean across active time slices)
        diag["acceptance_rate"] = jnp.mean(all_stats["acceptance_rate"])

        # 3) Initial Condition Anchoring (Lagrangian Penalty at t=0)
        if config.lambda_ic > 0.0:
            rng_val, ic_rng = jax.random.split(rng_val)
            ic_configs = jax.random.randint(ic_rng, (config.batch_size, config.N), 0, 2)

            def ic_loss_fn(m):
                return _x_polarized_mse_loss(m, ic_configs, config.N, ic_target_logp)

            grad_ic = nnx.grad(ic_loss_fn)(model)
            # Use tree_map for robust addition/scaling across dict/State types
            grad_total = jax.tree_util.tree_map(
                lambda gt, gi: gt + gi * config.lambda_ic, grad_total, grad_ic
            )

            # Update total norm diagnostic
            diag["grad_norm_total"] = jnp.asarray(
                jnp.linalg.norm(
                    jnp.concatenate(
                        [
                            jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32))
                            for leaf in jax.tree_util.tree_leaves(grad_total)
                        ]
                    )
                )
            )

        # 4) Apply Optimizer Update
        new_opt_state = _apply_optimizer_update(
            model=model,
            optimizer=optimizer,
            opt_state=opt_st,
            grad_state=grad_total,
        )

        return diag, new_opt_state, next_configs, rng_val

    for local_step in range(config.n_steps):
        if verbose:
            print(
                f"\n=== Spacetime Trajectory Step {local_step + 1}/{config.n_steps} "
                f"over {len(time_indices)} time slices ==="
            )

        diag, opt_state, chain_configurations, rng = jitted_trajectory_train_step(
            wf.model, opt_state, chain_configurations, rng
        )

        # Record metrics
        for key in metrics_history.keys():
            if key in ("time", "step"):
                continue
            if key in diag:
                metrics_history[key].append(float(diag[key]))

        metrics_history["time"].append(float(jnp.mean(active_times)))
        metrics_history["step"].append(global_step)

        if verbose:
            print(
                f"  Trajectory step {local_step + 1}/{config.n_steps}: "
                f"loss={float(diag['loss']):.6f}, "
                f"accept_rate={float(diag['acceptance_rate']):.3f}, "
                f"grad_norm={float(diag['grad_norm_total']):.6f}"
            )

        global_step += 1

        if (
            config.checkpoint_dir
            and (local_step + 1) % config.checkpoint_interval == 0
        ):
            ckpt_path = (
                f"{config.checkpoint_dir}/checkpoint_step{local_step + 1:04d}.pkl"
            )
            _save_checkpoint(
                config,
                wf,
                ham,
                metrics_history,
                global_step,
                config.time_steps,
                float(times[-1]),
                chain_configurations,
                rng,
                opt_state,
                ckpt_path,
            )
            if verbose:
                print(f"  Checkpoint saved to {ckpt_path}")

    if verbose:
        print("\n=== Training complete ===")
        print(f"Total steps: {global_step}")
        if metrics_history["loss"]:
            print(f"Final loss: {metrics_history['loss'][-1]:.6f}")
            print(f"Initial loss: {metrics_history['loss'][0]:.6f}")

    return {
        "wavefunction": wf,
        "hamiltonian": ham,
        "optimizer": optimizer,
        "opt_state": opt_state,
        "metrics_history": metrics_history,
        "final_configurations": chain_configurations,
        "global_step": global_step,
        "completed_time_steps": config.time_steps,
        "rng": rng,
        "config": config,
    }


if __name__ == "__main__":
    # Example minimal experiment on small transverse-field Ising system.
    config = TrainingConfig(
        N=5,
        J=1.0,
        h=0.5,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        learning_rate=0.001,
        n_steps=10,
        n_samples_per_chain=16,
        burn_in=50,
        thinning=1,
        n_chains=1,
        time_steps=3,
        checkpoint_dir=None,
        seed=42,
    )

    result = train_loop(config, verbose=True)

    print("\nTraining result keys:", list(result.keys()))
    print("Metrics tracked:", list(result["metrics_history"].keys()))
