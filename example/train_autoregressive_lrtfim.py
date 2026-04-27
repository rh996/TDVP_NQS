import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep Matplotlib cache in a writable project-local temp directory.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.grad import _ModelWavefunctionView
from src.hamiltonian import LongRangeTransverseIsingHamiltonian
from src.observables import (
    sample_and_measure_energy_curve,
    sample_and_measure_observables,
)
from src.TDVP import TrainingConfig, save_training_checkpoint, train_loop
from src.wavefunction import AutoregressiveNQS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train autoregressive TDVP on the long-range transverse-field Ising model."
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-samples-per-chain", type=int, default=2500)
    parser.add_argument("--optimizer-name", type=str, default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adamw-b1", type=float, default=0.9)
    parser.add_argument("--adamw-b2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--use-unique-ar-samples", action="store_true")
    parser.add_argument("--t-initial", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--pretrain-lr", type=float, default=0.005)
    parser.add_argument("--lambda-ic", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target-site", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "example" / "outputs" / "autoregressive_lrtfim",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        N=args.n_sites,
        J=-1.0,
        h=0.5,
        Num_boxes=3,
        emb_dim=16,
        num_heads=4,
        head_dim=8,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        adamw_b1=args.adamw_b1,
        adamw_b2=args.adamw_b2,
        adamw_eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        n_steps=args.n_steps,
        n_samples_per_chain=args.n_samples_per_chain,
        burn_in=0,
        thinning=1,
        n_chains=args.n_chains,
        use_unique_ar_samples=args.use_unique_ar_samples,
        initial_chain_configurations=jnp.ones(
            (args.n_chains, args.n_sites), dtype=jnp.int32
        ),
        t_initial=args.t_initial,
        t_final=args.t_final,
        time_steps=args.time_steps,
        pretrain_steps=args.pretrain_steps,
        pretrain_lr=args.pretrain_lr,
        lambda_ic=args.lambda_ic,
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_interval=200,
        seed=args.seed,
    )
    ham = LongRangeTransverseIsingHamiltonian(
        J=config.J,
        h=config.h,
        N=config.N,
        alpha=args.alpha,
    )

    rng = jax.random.PRNGKey(config.seed)
    wf = AutoregressiveNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(int(rng[0])),
    )

    print(
        f"Running autoregressive TDVP (LRTFIM): "
        f"alpha={args.alpha}, "
        f"optimizer={config.optimizer_name.upper()}, "
        f"lr={config.learning_rate}, "
        f"weight_decay={config.weight_decay}, "
        f"gradient_clip_norm={config.gradient_clip_norm}, "
        f"n_chains={config.n_chains}, "
        f"n_samples_per_chain={config.n_samples_per_chain}, "
        f"use_unique_ar_samples={config.use_unique_ar_samples}, "
        f"pretrain_steps={config.pretrain_steps}, "
        f"lambda_ic={config.lambda_ic}"
    )

    result = train_loop(
        config,
        verbose=True,
        initial_wavefunction=wf,
        initial_hamiltonian=ham,
    )
    metrics = result["metrics_history"]

    figure_path = output_dir / "autoregressive_lrtfim_loss.png"
    plt.figure(figsize=(9, 5))
    plt.plot(metrics["step"], metrics["loss"], linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("TDVP Loss")
    plt.title(f"Autoregressive TDVP on LRTFIM (alpha={args.alpha:g})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {figure_path}")

    wf = result["wavefunction"]
    times = jnp.linspace(config.t_initial, config.t_final, config.time_steps)

    print("\nMeasuring energy curve...")
    energy_curve, _, _ = sample_and_measure_energy_curve(
        result["hamiltonian"],
        wf,
        times,
        n_sites=config.N,
        initial_configurations=jnp.zeros((config.n_chains, config.N), dtype=jnp.int32),
        n_samples=config.n_samples_per_chain,
        burn_in=config.burn_in,
        thinning=config.thinning,
        key=jax.random.PRNGKey(args.seed + 2),
    )

    energy_figure_path = output_dir / "energy_curve.png"
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        times,
        energy_curve.energy_real,
        yerr=energy_curve.standard_error,
        fmt="o-",
        linewidth=1.5,
        markersize=4,
        color="tab:green",
        capsize=3,
    )
    plt.xlabel("Time t")
    plt.ylabel("<H(t)>")
    plt.title(f"LRTFIM Autoregressive Energy, N={config.N}, alpha={args.alpha:g}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(energy_figure_path, dpi=150)
    plt.close()
    print(f"Saved energy curve to {energy_figure_path}")

    print(f"\nMeasuring observables for site {args.target_site}...")
    z_values = []
    x_values = []
    obs_configs = jnp.zeros((config.n_chains, config.N), dtype=jnp.int32)
    measure_rng = jax.random.PRNGKey(args.seed + 1)

    @nnx.jit
    def jitted_measure(model, t_val, configs, key):
        wf_view = _ModelWavefunctionView(model)
        return sample_and_measure_observables(
            wf=wf_view,
            t=t_val,
            n_sites=config.N,
            initial_configurations=configs,
            n_samples=config.n_samples_per_chain,
            burn_in=config.burn_in,
            thinning=config.thinning,
            key=key,
        )

    for t_val in times:
        measure_rng, subkey = jax.random.split(measure_rng)
        obs_est, _, obs_configs = jitted_measure(
            wf.model,
            float(t_val),
            obs_configs,
            subkey,
        )
        z_values.append(float(obs_est.z_sites[args.target_site]))
        x_values.append(float(obs_est.x_sites_real[args.target_site]))

    z_figure_path = output_dir / "z_trajectory.png"
    plt.figure(figsize=(9, 5))
    plt.plot(times, z_values, "o-", linewidth=1.5, markersize=4, color="tab:blue")
    plt.xlabel("Time t")
    plt.ylabel(f"<Z_{args.target_site + 1}(t)>")
    plt.title(f"LRTFIM Autoregressive Z_{args.target_site + 1}(t), N={config.N}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(z_figure_path, dpi=150)
    plt.close()
    print(f"Saved Z(t) plot to {z_figure_path}")

    x_figure_path = output_dir / "x_trajectory.png"
    plt.figure(figsize=(9, 5))
    plt.plot(times, x_values, "o-", linewidth=1.5, markersize=4, color="tab:orange")
    plt.xlabel("Time t")
    plt.ylabel(f"<X_{args.target_site + 1}(t)>")
    plt.title(f"LRTFIM Autoregressive X_{args.target_site + 1}(t), N={config.N}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(x_figure_path, dpi=150)
    plt.close()
    print(f"Saved X(t) plot to {x_figure_path}")

    checkpoint_path = output_dir / "final_wavefunction.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=result["wavefunction"],
        ham=result["hamiltonian"],
        config=result["config"],
        metrics_history=result["metrics_history"],
        global_step=result["global_step"],
        completed_time_steps=result["completed_time_steps"],
        current_time=float(config.t_final),
        chain_configurations=result["final_configurations"],
        rng=result["rng"],
        opt_state=result["opt_state"],
    )
    print(f"Saved final wavefunction and state to {checkpoint_path}")


if __name__ == "__main__":
    main()
