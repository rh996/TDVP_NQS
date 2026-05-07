import argparse
import os
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

from src.grad import _ModelWavefunctionView
from src.observables import (
    enumerate_binary_configurations,
    normalized_statevector,
    sample_and_measure_observables,
)
from src.TDVP import TrainingConfig, save_training_checkpoint, train_loop
from src.wavefunction import AutoregressiveNQS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original non-Z2 AutoregressiveNQS with direct AR sampling."
    )
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-samples-per-chain", type=int, default=1280)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--t-initial", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--optimizer-name", type=str, default="muon")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument(
        "--fixed-time-grid",
        action="store_true",
        help="Disable random continuous-time collocation and train only on the fixed time grid.",
    )
    parser.add_argument(
        "--residual-loss-mode",
        choices=("variance", "schrodinger_l2", "phase_speed"),
        default="phase_speed",
        help="Residual loss mode. Defaults to phase_speed for autoregressive training.",
    )
    parser.add_argument("--use-unique-ar-samples", action="store_true")
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=10,
        help="Optional all-time-slice initial-condition pretraining steps.",
    )
    parser.add_argument("--pretrain-lr", type=float, default=0.005)
    parser.add_argument("--lambda-ic", type=float, default=10.0)
    parser.add_argument("--target-site", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-statevector-max-sites",
        type=int,
        default=16,
        help=(
            "Maximum N for exhaustive psi(x,t) export. "
            "Set lower to avoid large 2^N statevector files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "example" / "outputs" / "autoregressive_no_z2",
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
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        residual_loss_mode=args.residual_loss_mode,
        n_steps=args.n_steps,
        n_samples_per_chain=args.n_samples_per_chain,
        burn_in=0,
        thinning=1,
        n_chains=args.n_chains,
        use_unique_ar_samples=args.use_unique_ar_samples,
        initial_chain_configurations=jnp.ones((args.n_chains, args.n_sites), dtype=jnp.int32),
        t_initial=args.t_initial,
        t_final=args.t_final,
        time_steps=args.time_steps,
        random_time_collocation=not args.fixed_time_grid,
        pretrain_steps=args.pretrain_steps,
        pretrain_lr=args.pretrain_lr,
        lambda_ic=args.lambda_ic,
        seed=args.seed,
    )
    wf = AutoregressiveNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(config.seed),
    )

    print(
        "Running original non-Z2 AutoregressiveNQS: "
        f"optimizer={config.optimizer_name.upper()}, "
        f"n_chains={config.n_chains}, "
        f"n_samples_per_chain={config.n_samples_per_chain}, "
        f"use_unique_ar_samples={config.use_unique_ar_samples}, "
        f"residual_loss_mode={config.residual_loss_mode}, "
        f"pretrain_steps={config.pretrain_steps}, "
        f"time_steps={config.time_steps}, "
        f"random_time_collocation={config.random_time_collocation}"
    )
    result = train_loop(config, verbose=True, initial_wavefunction=wf)
    metrics = result["metrics_history"]
    print(f"Final loss: {metrics['loss'][-1]:.6f}")

    loss_path = output_dir / "loss.png"
    plt.figure(figsize=(9, 5))
    plt.plot(metrics["step"], metrics["loss"], linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("TDVP Loss")
    plt.title("Original non-Z2 AutoregressiveNQS Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {loss_path}")

    print(f"\nMeasuring observables for site {args.target_site}...")
    times = jnp.linspace(config.t_initial, config.t_final, config.time_steps)
    obs_configs = jnp.zeros((config.n_chains, config.N), dtype=jnp.int32)
    z_values = []
    x_values = []
    rng = jax.random.PRNGKey(args.seed)

    @nnx.jit
    def jitted_measure(model, t_val, configs, key):
        wf_view = _ModelWavefunctionView(model)
        return sample_and_measure_observables(
            wf=wf_view,
            t=t_val,
            n_sites=config.N,
            initial_configurations=configs,
            n_samples=config.n_samples_per_chain,
            burn_in=0,
            thinning=1,
            key=key,
        )

    for t_val in times:
        rng, measure_rng = jax.random.split(rng)
        obs_est, _, obs_configs = jitted_measure(
            result["wavefunction"].model,
            float(t_val),
            obs_configs,
            measure_rng,
        )
        z_values.append(float(obs_est.z_sites[args.target_site]))
        x_values.append(float(obs_est.x_sites_real[args.target_site]))

    z_path = output_dir / "z_trajectory.png"
    plt.figure(figsize=(9, 5))
    plt.plot(times, z_values, "o-", linewidth=1.5, markersize=4, color="tab:blue")
    plt.xlabel("Time t")
    plt.ylabel(f"<Z_{args.target_site + 1}(t)>")
    plt.title(f"Original non-Z2 AR Z_{args.target_site + 1}(t), N={config.N}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(z_path, dpi=150)
    plt.close()
    print(f"Saved Z(t) plot to {z_path}")

    x_path = output_dir / "x_trajectory.png"
    plt.figure(figsize=(9, 5))
    plt.plot(times, x_values, "o-", linewidth=1.5, markersize=4, color="tab:orange")
    plt.xlabel("Time t")
    plt.ylabel(f"<X_{args.target_site + 1}(t)>")
    plt.title(f"Original non-Z2 AR X_{args.target_site + 1}(t), N={config.N}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(x_path, dpi=150)
    plt.close()
    print(f"Saved X(t) plot to {x_path}")

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
    print(f"Saved final wavefunction checkpoint to {checkpoint_path}")

    if config.N <= args.save_statevector_max_sites:
        basis_configs = enumerate_binary_configurations(config.N)
        psi_by_time = []
        for t_val in times:
            psi_by_time.append(normalized_statevector(result["wavefunction"], config.N, t_val))
        psi_xt = jnp.stack(psi_by_time, axis=0)

        psi_path = output_dir / "final_psi_xt.npz"
        np.savez(
            psi_path,
            times=np.asarray(jax.device_get(times)),
            configurations=np.asarray(jax.device_get(basis_configs), dtype=np.int32),
            psi=np.asarray(jax.device_get(psi_xt)),
            psi_real=np.asarray(jax.device_get(jnp.real(psi_xt))),
            psi_imag=np.asarray(jax.device_get(jnp.imag(psi_xt))),
            probability=np.asarray(jax.device_get(jnp.abs(psi_xt) ** 2)),
        )
        print(f"Saved normalized psi(x,t) table to {psi_path}")
    else:
        print(
            "Skipped exhaustive psi(x,t) export because "
            f"N={config.N} exceeds --save-statevector-max-sites="
            f"{args.save_statevector_max_sites}."
        )


if __name__ == "__main__":
    main()
