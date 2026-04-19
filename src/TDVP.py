import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from src.grad import tdvp_vmc_gradient
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
    n_steps: int = 100
    batch_size: int = 32

    # Sampling
    n_samples_per_chain: int = 32
    burn_in: int = 100
    thinning: int = 1
    n_chains: int = 1

    # Time
    t_initial: float = 0.0
    t_final: float = 1.0
    time_steps: int = 10

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


def _save_checkpoint(
    config: TrainingConfig,
    step: int,
    time_val: float,
    metrics_history: Dict[str, list],
    filepath: str,
):
    """Save checkpoint metadata to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "step": int(step),
        "time": float(time_val),
        "config": asdict(config),
        "metrics_history": metrics_history,
    }

    with open(filepath, "w") as f:
        json.dump(checkpoint_data, f, indent=2, default=str)


def _apply_gradient_descent_update(
    model: nnx.Module,
    grad_state,
    learning_rate: float,
) -> None:
    """Apply an in-place SGD update through the model parameter state."""
    params = nnx.state(model, nnx.Param)
    new_params = jax.tree_util.tree_map(
        lambda param, grad: param - learning_rate * grad,
        params,
        grad_state,
    )
    nnx.update(model, new_params)


def train_step(
    wf: Wavefunction,
    ham: TransverseIsingHamiltonian,
    t: float,
    config: TrainingConfig,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jax.Array]:
    """Single training step: sample → loss → grad → update.

    Returns:
        (loss, diagnostics_dict, next_rng)
    """
    rng, rng_sample = jax.random.split(rng)

    # 1) Sample configurations from the current wavefunction.
    init_config = jnp.zeros((config.n_chains, ham.N), dtype=jnp.int32)
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

    # Reshape samples to batch: (n_chains, n_samples, N) -> (batch, N)
    batch_size = config.n_chains * config.n_samples_per_chain
    configurations_batch = samples.reshape((batch_size, ham.N))

    # 2) Compute loss diagnostics and gradient on the sampled batch.
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

    # 3) Manual gradient descent: update parameters in-place.
    _apply_gradient_descent_update(
        wf.model,
        grad_total,
        config.learning_rate,
    )

    # 4) Collect diagnostics.
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

    return loss, diagnostics, rng


def train_loop(
    config: TrainingConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full training loop over time points and optimization steps.

    Args:
        config: TrainingConfig with all hyperparameters.
        verbose: Whether to print progress.

    Returns:
        training_result dict with final model, metrics, and config.
    """
    _validate_training_config(config)

    # 1) Initialize model and wavefunction.
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_model = jax.random.split(rng)

    wf = tSpinNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(int(rng_model[0])),
    )

    # 2) Initialize Hamiltonian.
    ham = TransverseIsingHamiltonian(J=config.J, h=config.h, N=config.N)

    # 3) Time schedule.
    times = jnp.linspace(config.t_initial, config.t_final, config.time_steps)

    # 4) Metrics tracking.
    metrics_history = {
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

    # 5) Training loop.
    global_step = 0

    for t_idx, t in enumerate(times):
        if verbose:
            print(
                f"\n=== Time step {t_idx + 1}/{config.time_steps}, t={float(t):.4f} ==="
            )

        for local_step in range(config.n_steps):
            rng, step_rng = jax.random.split(rng)

            loss, diag, rng = train_step(
                wf=wf,
                ham=ham,
                t=float(t),
                config=config,
                rng=step_rng,
            )

            # Record metrics.
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

        # Checkpointing.
        if config.checkpoint_dir and (t_idx + 1) % config.checkpoint_interval == 0:
            ckpt_path = f"{config.checkpoint_dir}/checkpoint_t{t_idx + 1:03d}.json"
            _save_checkpoint(config, global_step, float(t), metrics_history, ckpt_path)
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
        "optimizer": None,
        "opt_state": None,
        "metrics_history": metrics_history,
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
