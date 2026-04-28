import argparse
import os
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

from src.grad import _ModelWavefunctionView
from src.observables import sample_and_measure_observables
from src.TDVP import TrainingConfig, train_loop
from src.wavefunction import tSpinNQS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original tSpinNQS with Metropolis-Hastings MCMC sampling."
    )
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-samples-per-chain", type=int, default=1280)
    parser.add_argument("--burn-in", type=int, default=50)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--t-initial", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
        "--output-dir",
        type=Path,
        default=ROOT / "example" / "outputs" / "mcmc_tspinnqs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_configs = jnp.ones((args.n_chains, args.n_sites), dtype=jnp.int32)
    config = TrainingConfig(
        N=args.n_sites,
        J=-1.0,
        h=0.5,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_steps=args.n_steps,
        n_samples_per_chain=args.n_samples_per_chain,
        burn_in=args.burn_in,
        thinning=args.thinning,
        n_chains=args.n_chains,
        initial_chain_configurations=initial_configs,
        t_initial=args.t_initial,
        t_final=args.t_final,
        time_steps=args.time_steps,
        pretrain_steps=args.pretrain_steps,
        pretrain_lr=args.pretrain_lr,
        lambda_ic=args.lambda_ic,
        seed=args.seed,
    )
    wf = tSpinNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(config.seed),
    )

    print(
        "Running original tSpinNQS with MCMC: "
        f"n_chains={config.n_chains}, "
        f"n_samples_per_chain={config.n_samples_per_chain}, "
        f"burn_in={config.burn_in}, "
        f"thinning={config.thinning}, "
        f"pretrain_steps={config.pretrain_steps}, "
        f"pretrain_time_slices={config.time_steps}"
    )
    result = train_loop(config, verbose=True, initial_wavefunction=wf)
    metrics = result["metrics_history"]
    print(f"Final loss: {metrics['loss'][-1]:.6f}")

    steps = metrics["step"]
    loss_values = metrics["loss"]
    loss_path = output_dir / "loss.png"
    plt.figure(figsize=(9, 5))
    plt.plot(steps, loss_values, linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("TDVP Loss")
    plt.title("Original tSpinNQS MCMC Training Loss")
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
            burn_in=config.burn_in,
            thinning=config.thinning,
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
    plt.title(f"Original tSpinNQS MCMC Z_{args.target_site + 1}(t), N={config.N}")
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
    plt.title(f"Original tSpinNQS MCMC X_{args.target_site + 1}(t), N={config.N}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(x_path, dpi=150)
    plt.close()
    print(f"Saved X(t) plot to {x_path}")


if __name__ == "__main__":
    main()
