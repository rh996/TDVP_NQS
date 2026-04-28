import argparse
import os
import sys
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.grad import _ModelWavefunctionView
from src.hamiltonian import LongRangeTransverseIsingHamiltonian
from src.observables import sample_and_measure_observables
from src.TDVP import TrainingConfig, train_loop, save_training_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep Matplotlib cache in a writable project-local temp directory.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TDVP from a fully polarized spin chain with AdamW and plot the loss."
        )
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--optimizer-name", type=str, default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adamw-b1", type=float, default=0.9)
    parser.add_argument("--adamw-b2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--n-samples-per-chain", type=int, default=1280)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--t-initial", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=200,
        help=(
            "Number of initial-condition pretraining steps applied across all "
            "configured time slices before TDVP evolution."
        ),
    )
    parser.add_argument("--pretrain-lr", type=float, default=0.005)
    parser.add_argument("--lambda-ic", type=float, default=10.0)
    parser.add_argument(
        "--target-site",
        type=int,
        default=0,
        help="Site index for per-site observables.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Decay exponent for long-range Ising interaction.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "example" / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sites = args.n_sites
    n_chains = args.n_chains
    fully_polarized = jnp.ones((n_chains, n_sites), dtype=jnp.int32)

    config = TrainingConfig(
        N=n_sites,
        J=-1.0,
        h=0.5,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        adamw_b1=args.adamw_b1,
        adamw_b2=args.adamw_b2,
        adamw_eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        n_steps=args.n_steps,
        n_samples_per_chain=args.n_samples_per_chain,
        burn_in=50,
        thinning=args.thinning,
        n_chains=n_chains,
        initial_chain_configurations=fully_polarized,
        t_initial=args.t_initial,
        t_final=args.t_final,
        time_steps=args.time_steps,
        pretrain_steps=args.pretrain_steps,
        pretrain_lr=args.pretrain_lr,
        lambda_ic=args.lambda_ic,
        checkpoint_dir=None,
        seed=args.seed,
    )

    print(
        f"Running TDVP (LRTFIM) with {config.optimizer_name.upper()}: "
        f"alpha={args.alpha}, "
        f"lr={config.learning_rate}, "
        f"weight_decay={config.weight_decay}, "
        f"b1={config.adamw_b1}, "
        f"b2={config.adamw_b2}, "
        f"eps={config.adamw_eps}, "
        f"pretrain_steps={config.pretrain_steps}, "
        f"pretrain_time_slices={config.time_steps}, "
        f"lambda_ic={config.lambda_ic}"
    )

    ham = LongRangeTransverseIsingHamiltonian(
        J=config.J, h=config.h, N=config.N, alpha=args.alpha
    )

    result = train_loop(config, verbose=True, initial_hamiltonian=ham)
    metrics = result["metrics_history"]

    steps = metrics["step"]
    loss_values = metrics["loss"]
    figure_path = output_dir / "fully_polarized_loss.png"

    plt.figure(figsize=(9, 5))
    plt.plot(steps, loss_values, linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("TDVP Loss")
    plt.title(
        f"TDVP Training From a Fully Polarized Spin Chain ({config.optimizer_name.upper()})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Saved loss curve to {figure_path}")

    # --- Measure and Plot Observables ---
    print(f"\nMeasuring observables for site {args.target_site}...")
    wf = result["wavefunction"]
    import jax

    times = jnp.linspace(config.t_initial, config.t_final, config.time_steps)
    z_values = []
    x_values = []

    # Use a dummy initial configuration for the first measurement
    obs_configs = jnp.zeros((config.n_chains, config.N), dtype=jnp.int32)

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
        rng, measure_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        obs_est, _, obs_configs = jitted_measure(
            wf.model, float(t_val), obs_configs, measure_rng
        )
        z_values.append(float(obs_est.z_sites[args.target_site]))
        x_values.append(float(obs_est.x_sites_real[args.target_site]))

    # Plot Z(t)
    z_figure_path = output_dir / "z_trajectory.png"
    plt.figure(figsize=(9, 5))
    plt.plot(times, z_values, "o-", linewidth=1.5, markersize=4, color="tab:blue")
    plt.xlabel("Time t")
    plt.ylabel(f"<Z_{args.target_site + 1}(t)>")
    plt.title(f"Magnetization Trajectory Z_{args.target_site + 1}(t) (N={config.N})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(z_figure_path, dpi=150)
    plt.close()
    print(f"Saved Z(t) plot to {z_figure_path}")

    # Plot X(t)
    x_figure_path = output_dir / "x_trajectory.png"
    plt.figure(figsize=(9, 5))
    plt.plot(times, x_values, "o-", linewidth=1.5, markersize=4, color="tab:orange")
    plt.xlabel("Time t")
    plt.ylabel(f"<X_{args.target_site + 1}(t)>")
    plt.title(f"Magnetization Trajectory X_{args.target_site + 1}(t) (N={config.N})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(x_figure_path, dpi=150)
    plt.close()
    print(f"Saved X(t) plot to {x_figure_path}")

    # --- Save Final Wavefunction ---
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
