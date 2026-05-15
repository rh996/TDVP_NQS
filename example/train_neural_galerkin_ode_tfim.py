import argparse
import os
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib.pyplot as plt

from src.galerkin import estimate_galerkin_matrices, sample_galerkin_basis_mixture
from src.observables import sample_and_measure_observables
from src.TDVP import TrainingConfig, save_training_checkpoint, train_loop
from src.wavefunction import FixedCoefficientNeuralGalerkinNQS, NeuralGalerkinNQS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Neural Galerkin NQS on TFIM, then solve projected ODE "
            "c(t)=exp(-it S^{-1}H)c(0) and measure site-0 observables."
        )
    )
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-samples-per-chain", type=int, default=512)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--t-initial", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--measure-time-steps", type=int, default=None)
    parser.add_argument("--measure-t-initial", type=float, default=None)
    parser.add_argument("--measure-t-final", type=float, default=None)
    parser.add_argument("--num-basis", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=16)
    parser.add_argument("--optimizer-name", type=str, default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--target-site", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-samples-per-chain", type=int, default=None)
    parser.add_argument("--projection-burn-in", type=int, default=None)
    parser.add_argument("--projection-thinning", type=int, default=None)
    parser.add_argument("--projection-regularization", type=float, default=1e-6)
    parser.add_argument(
        "--fixed-time-grid",
        action="store_true",
        help="Disable random continuous-time collocation and train only on the fixed time grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "example" / "outputs" / "neural_galerkin_ode_tfim",
    )
    return parser.parse_args()


def _plot_site_observable(
    times,
    values,
    *,
    train_t_final: float,
    measure_t_final: float,
    ylabel: str,
    title: str,
    color: str,
    path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(times, values, "o-", linewidth=1.5, markersize=4, color=color)
    if measure_t_final > train_t_final:
        plt.axvline(train_t_final, linestyle="--", linewidth=1.0, color="tab:gray")
        plt.axvspan(train_t_final, measure_t_final, color="tab:gray", alpha=0.12)
    plt.xlabel("Time t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.target_site < args.n_sites:
        raise ValueError(
            f"target_site must be in [0, {args.n_sites}), got {args.target_site}"
        )

    projection_samples_per_chain = (
        args.n_samples_per_chain
        if args.projection_samples_per_chain is None
        else args.projection_samples_per_chain
    )
    projection_burn_in = (
        args.burn_in if args.projection_burn_in is None else args.projection_burn_in
    )
    projection_thinning = (
        args.thinning
        if args.projection_thinning is None
        else args.projection_thinning
    )

    initial_configs = jnp.ones((args.n_chains, args.n_sites), dtype=jnp.int32)
    config = TrainingConfig(
        N=args.n_sites,
        J=-1.0,
        h=0.5,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        num_galerkin_basis=args.num_basis,
        num_galerkin_modes=args.num_modes,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        residual_loss_mode="variance",
        n_steps=args.n_steps,
        n_samples_per_chain=args.n_samples_per_chain,
        burn_in=args.burn_in,
        thinning=args.thinning,
        n_chains=args.n_chains,
        initial_chain_configurations=initial_configs,
        t_initial=args.t_initial,
        t_final=args.t_final,
        time_steps=args.time_steps,
        random_time_collocation=not args.fixed_time_grid,
        pretrain_steps=0,
        lambda_ic=0.0,
        seed=args.seed,
    )

    measure_t_initial = (
        config.t_initial if args.measure_t_initial is None else args.measure_t_initial
    )
    train_duration = config.t_final - config.t_initial
    measure_t_final = (
        config.t_final + train_duration
        if args.measure_t_final is None
        else args.measure_t_final
    )
    measure_time_steps = (
        max(4 * config.time_steps, 50)
        if args.measure_time_steps is None
        else args.measure_time_steps
    )

    wf = NeuralGalerkinNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        num_basis=config.num_galerkin_basis,
        num_modes=config.num_galerkin_modes,
        rngs=nnx.Rngs(config.seed),
    )

    print(
        "Training NeuralGalerkinNQS basis with MCMC: "
        f"optimizer={config.optimizer_name.upper()}, "
        f"num_basis={config.num_galerkin_basis}, "
        f"num_modes={config.num_galerkin_modes}, "
        f"loss={config.residual_loss_mode}, "
        f"n_chains={config.n_chains}, "
        f"n_samples_per_chain={config.n_samples_per_chain}, "
        f"time_steps={config.time_steps}, "
        f"random_time_collocation={config.random_time_collocation}"
    )
    result = train_loop(config, verbose=True, initial_wavefunction=wf)
    metrics = result["metrics_history"]
    print(f"Final training loss: {metrics['loss'][-1]:.6f}")

    loss_path = output_dir / "loss.png"
    plt.figure(figsize=(9, 5))
    plt.plot(metrics["step"], metrics["loss"], linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("TDVP Loss")
    plt.title("Neural Galerkin Basis Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {loss_path}")

    rng = jax.random.PRNGKey(args.seed)
    rng, projection_rng = jax.random.split(rng)
    projection_samples, projection_stats, projection_configs = (
        sample_galerkin_basis_mixture(
            result["wavefunction"],
            result["final_configurations"],
            n_samples=projection_samples_per_chain,
            burn_in=projection_burn_in,
            thinning=projection_thinning,
            key=projection_rng,
        )
    )
    projection = estimate_galerkin_matrices(
        result["hamiltonian"],
        result["wavefunction"],
        projection_samples,
        regularization=args.projection_regularization,
    )
    print(
        "Projected Galerkin matrices: "
        f"S.shape={projection.S.shape}, "
        f"H.shape={projection.H.shape}, "
        f"projection_acceptance={float(jnp.mean(projection_stats['acceptance_rate'])):.4f}, "
        f"regularization={projection.regularization:g}"
    )

    np.savez(
        output_dir / "galerkin_projection.npz",
        S=np.asarray(projection.S),
        H=np.asarray(projection.H),
        generator=np.asarray(projection.generator),
        projection_acceptance_rate=np.asarray(projection_stats["acceptance_rate"]),
        regularization=np.asarray(projection.regularization),
    )
    print(f"Saved projected matrices to {output_dir / 'galerkin_projection.npz'}")

    ode_wf = FixedCoefficientNeuralGalerkinNQS(
        result["wavefunction"],
        projection.generator,
        t_initial=config.t_initial,
    )

    print(f"\nMeasuring ODE-propagated observables for site {args.target_site}...")
    times = jnp.linspace(measure_t_initial, measure_t_final, measure_time_steps)
    obs_configs = projection_configs
    z_values = []
    x_values = []
    acceptance_values = []
    for t_val in times:
        rng, measure_rng = jax.random.split(rng)
        obs_est, obs_stats, obs_configs = sample_and_measure_observables(
            wf=ode_wf,
            t=t_val,
            n_sites=config.N,
            initial_configurations=obs_configs,
            n_samples=config.n_samples_per_chain,
            burn_in=config.burn_in,
            thinning=config.thinning,
            key=measure_rng,
        )
        z_values.append(float(obs_est.z_sites[args.target_site]))
        x_values.append(float(obs_est.x_sites_real[args.target_site]))
        acceptance_values.append(float(jnp.mean(obs_stats["acceptance_rate"])))

    z_path = output_dir / "ode_z_trajectory.png"
    _plot_site_observable(
        np.asarray(times),
        z_values,
        train_t_final=config.t_final,
        measure_t_final=measure_t_final,
        ylabel=f"<Z_{args.target_site + 1}(t)>",
        title=f"Projected Neural Galerkin TFIM Z_{args.target_site + 1}(t), N={config.N}",
        color="tab:blue",
        path=z_path,
    )
    print(f"Saved ODE Z(t) plot to {z_path}")

    x_path = output_dir / "ode_x_trajectory.png"
    _plot_site_observable(
        np.asarray(times),
        x_values,
        train_t_final=config.t_final,
        measure_t_final=measure_t_final,
        ylabel=f"<X_{args.target_site + 1}(t)>",
        title=f"Projected Neural Galerkin TFIM X_{args.target_site + 1}(t), N={config.N}",
        color="tab:orange",
        path=x_path,
    )
    print(f"Saved ODE X(t) plot to {x_path}")

    np.savez(
        output_dir / "ode_site0_observables.npz",
        times=np.asarray(times),
        z_site=np.asarray(z_values),
        x_site_real=np.asarray(x_values),
        acceptance_rate=np.asarray(acceptance_values),
    )
    print(f"Saved ODE observable data to {output_dir / 'ode_site0_observables.npz'}")

    checkpoint_path = output_dir / "trained_basis_wavefunction.pkl"
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
    print(f"Saved trained basis checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
