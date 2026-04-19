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

from src.grad import _tree_add, tdvp_vmc_gradient
from src.hamiltonian import TransverseIsingHamiltonian
from src.loss import tdvp_residual_loss
from src.sampler import metropolis_hastings_sample
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
    time_loss_mode: str = "sum"

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
    if config.time_loss_mode not in ("sum", "serial"):
        raise ValueError(
            f"time_loss_mode must be 'sum' or 'serial', got {config.time_loss_mode!r}"
        )
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported optimizer_name {config.optimizer_name!r}; only 'adamw' is currently supported."
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
    if config.optimizer_name == "adamw":
        return optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.adamw_b1,
            b2=config.adamw_b2,
            eps=config.adamw_eps,
            weight_decay=config.weight_decay,
        )
    raise ValueError(
        f"Unsupported optimizer_name {config.optimizer_name!r}; only 'adamw' is currently supported."
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


def _evaluate_time_slice(
    wf: Wavefunction,
    ham: TransverseIsingHamiltonian,
    t: float,
    config: TrainingConfig,
    rng: jax.Array,
    *,
    initial_configurations: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Any, jax.Array, jnp.ndarray]:
    """Sample one time slice and compute its loss/gradient without updating parameters."""
    rng, rng_sample = jax.random.split(rng)

    if initial_configurations is None:
        initial_configurations = jnp.zeros((config.n_chains, ham.N), dtype=jnp.int32)
    init_config = _validate_chain_configurations(
        initial_configurations,
        n_chains=config.n_chains,
        n_sites=ham.N,
    )
    samples, sampler_stats = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=init_config,
        t=jnp.asarray(t, dtype=jnp.float32),
        n_sites=ham.N,
        n_samples=config.n_samples_per_chain,
        burn_in=config.burn_in,
        thinning=config.thinning,
        key=rng_sample,
        return_stats=True,
    )

    batch_size = config.n_chains * config.n_samples_per_chain
    configurations_batch = samples.reshape((batch_size, ham.N))

    loss_value, loss_diag = tdvp_residual_loss(
        ham=ham,
        wf=wf,
        configurations=configurations_batch,
        t=jnp.asarray(t, dtype=jnp.float32),
        return_diagnostics=True,
    )

    grad_total, grad_aux = tdvp_vmc_gradient(
        ham=ham,
        wf=wf,
        configurations=configurations_batch,
        t=jnp.asarray(t, dtype=jnp.float32),
        return_diagnostics=True,
    )

    loss = jnp.asarray(loss_value, dtype=jnp.float32)
    accept_rate_vals = jnp.asarray(sampler_stats["acceptance_rate"], dtype=jnp.float32)
    if accept_rate_vals.ndim > 0:
        accept_rate_mean = jnp.mean(accept_rate_vals)
    else:
        accept_rate_mean = accept_rate_vals

    e_real_mean = jnp.mean(jnp.asarray(loss_diag.e_real, dtype=jnp.float32))
    e_imag_mean = jnp.mean(jnp.asarray(loss_diag.e_imag, dtype=jnp.float32))

    diagnostics = {
        "loss": loss,
        "acceptance_rate": accept_rate_mean,
        "grad_norm_pathwise": jnp.asarray(
            grad_aux["grad_norm_pathwise"], dtype=jnp.float32
        ),
        "grad_norm_covariance": jnp.asarray(
            grad_aux["grad_norm_covariance"], dtype=jnp.float32
        ),
        "grad_norm_total": jnp.asarray(grad_aux["grad_norm_total"], dtype=jnp.float32),
        "ell_mean": jnp.asarray(grad_aux["ell_mean"], dtype=jnp.float32),
        "ell_std": jnp.asarray(grad_aux["ell_std"], dtype=jnp.float32),
        "finite_loss": jnp.asarray(grad_aux["finite_loss"]),
        "finite_grads": jnp.asarray(grad_aux["finite_grads"]),
        "e_real_mean": e_real_mean,
        "e_imag_mean": e_imag_mean,
    }

    next_configurations = samples[:, -1, :]
    return loss, diagnostics, grad_total, rng, next_configurations


def train_step(
    wf: Wavefunction,
    ham: TransverseIsingHamiltonian,
    t: float,
    config: TrainingConfig,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    rng: jax.Array,
    *,
    initial_configurations: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Any, jax.Array, jnp.ndarray]:
    """Single training step: sample → loss → grad → update.

    Returns:
        (loss, diagnostics_dict, next_opt_state, next_rng, next_chain_configurations)
    """
    loss, diagnostics, grad_total, rng, next_configurations = _evaluate_time_slice(
        wf=wf,
        ham=ham,
        t=t,
        config=config,
        rng=rng,
        initial_configurations=initial_configurations,
    )

    # AdamW update: update parameters in-place and carry optimizer state.
    opt_state = _apply_optimizer_update(
        wf.model,
        optimizer,
        opt_state,
        grad_total,
    )
    return loss, diagnostics, opt_state, rng, next_configurations


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

    time_indices = list(range(start_time_index, config.time_steps))
    if config.time_loss_mode == "serial":
        for t_idx in time_indices:
            t = times[t_idx]
            if verbose:
                print(
                    f"\n=== Time step {t_idx + 1}/{config.time_steps}, t={float(t):.4f} ==="
                )

            for local_step in range(config.n_steps):
                rng, step_rng = jax.random.split(rng)

                loss, diag, opt_state, rng, chain_configurations = train_step(
                    wf=wf,
                    ham=ham,
                    t=float(t),
                    config=config,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    rng=step_rng,
                    initial_configurations=chain_configurations,
                )

                for key in metrics_history.keys():
                    if key in ("time", "step"):
                        continue
                    if key in diag:
                        metrics_history[key].append(float(diag[key]))

                metrics_history["time"].append(float(t))
                metrics_history["step"].append(global_step)

                if verbose and (local_step + 1) % max(1, config.n_steps // 5) == 0:
                    print(
                        f"  Step {local_step + 1}/{config.n_steps}: "
                        f"loss={float(loss):.6f}, "
                        f"accept_rate={float(diag['acceptance_rate']):.3f}, "
                        f"grad_norm={float(diag['grad_norm_total']):.6f}"
                    )

                global_step += 1

            if config.checkpoint_dir and (t_idx + 1) % config.checkpoint_interval == 0:
                ckpt_path = f"{config.checkpoint_dir}/checkpoint_t{t_idx + 1:03d}.pkl"
                _save_checkpoint(
                    config,
                    wf,
                    ham,
                    metrics_history,
                    global_step,
                    t_idx + 1,
                    float(t),
                    chain_configurations,
                    rng,
                    opt_state,
                    ckpt_path,
                )
                if verbose:
                    print(f"  Checkpoint saved to {ckpt_path}")
    else:
        for local_step in range(config.n_steps):
            if verbose:
                print(
                    f"\n=== Joint time step {local_step + 1}/{config.n_steps} "
                    f"over {len(time_indices)} time slices ==="
                )

            step_chain_configurations = chain_configurations
            loss_total = jnp.asarray(0.0, dtype=jnp.float32)
            grad_total = None
            acceptance_rates = []
            grad_norm_pathwise_vals = []
            grad_norm_covariance_vals = []
            grad_norm_total_vals = []
            ell_mean_vals = []
            ell_std_vals = []
            finite_loss_flags = []
            finite_grads_flags = []
            e_real_means = []
            e_imag_means = []

            for t_idx in time_indices:
                t = times[t_idx]
                rng, step_rng = jax.random.split(rng)
                loss, diag, grad_slice, rng, step_chain_configurations = (
                    _evaluate_time_slice(
                        wf=wf,
                        ham=ham,
                        t=float(t),
                        config=config,
                        rng=step_rng,
                        initial_configurations=step_chain_configurations,
                    )
                )

                loss_total = loss_total + loss
                grad_total = grad_slice if grad_total is None else _tree_add(
                    grad_total, grad_slice
                )
                acceptance_rates.append(jnp.asarray(diag["acceptance_rate"]))
                grad_norm_pathwise_vals.append(jnp.asarray(diag["grad_norm_pathwise"]))
                grad_norm_covariance_vals.append(
                    jnp.asarray(diag["grad_norm_covariance"])
                )
                grad_norm_total_vals.append(jnp.asarray(diag["grad_norm_total"]))
                ell_mean_vals.append(jnp.asarray(diag["ell_mean"]))
                ell_std_vals.append(jnp.asarray(diag["ell_std"]))
                finite_loss_flags.append(jnp.asarray(diag["finite_loss"]))
                finite_grads_flags.append(jnp.asarray(diag["finite_grads"]))
                e_real_means.append(jnp.asarray(diag["e_real_mean"]))
                e_imag_means.append(jnp.asarray(diag["e_imag_mean"]))

            opt_state = _apply_optimizer_update(
                wf.model,
                optimizer,
                opt_state,
                grad_total,
            )
            chain_configurations = step_chain_configurations

            diag = {
                "loss": loss_total,
                "acceptance_rate": jnp.mean(jnp.stack(acceptance_rates)),
                "grad_norm_pathwise": jnp.mean(jnp.stack(grad_norm_pathwise_vals)),
                "grad_norm_covariance": jnp.mean(
                    jnp.stack(grad_norm_covariance_vals)
                ),
                "grad_norm_total": jnp.asarray(
                    jnp.linalg.norm(
                        jnp.concatenate(
                            [
                                jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32))
                                for leaf in jax.tree_util.tree_leaves(grad_total)
                            ]
                        )
                    )
                ),
                "ell_mean": jnp.mean(jnp.stack(ell_mean_vals)),
                "ell_std": jnp.mean(jnp.stack(ell_std_vals)),
                "finite_loss": jnp.asarray(jnp.all(jnp.stack(finite_loss_flags))),
                "finite_grads": jnp.asarray(jnp.all(jnp.stack(finite_grads_flags))),
                "e_real_mean": jnp.mean(jnp.stack(e_real_means)),
                "e_imag_mean": jnp.mean(jnp.stack(e_imag_means)),
            }

            for key in metrics_history.keys():
                if key in ("time", "step"):
                    continue
                if key in diag:
                    metrics_history[key].append(float(diag[key]))

            metrics_history["time"].append(float(jnp.mean(times[jnp.array(time_indices)])))
            metrics_history["step"].append(global_step)

            if verbose:
                print(
                    f"  Joint step {local_step + 1}/{config.n_steps}: "
                    f"loss={float(diag['loss']):.6f}, "
                    f"accept_rate={float(diag['acceptance_rate']):.3f}, "
                    f"grad_norm={float(diag['grad_norm_total']):.6f}"
                )

            global_step += 1

            if config.checkpoint_dir and (local_step + 1) % config.checkpoint_interval == 0:
                ckpt_path = f"{config.checkpoint_dir}/checkpoint_t{local_step + 1:03d}.pkl"
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
        print(f"\n=== Training complete ===")
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
