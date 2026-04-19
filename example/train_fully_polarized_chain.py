import argparse
from pathlib import Path
import os
import sys

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep Matplotlib cache in a writable project-local temp directory.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib.pyplot as plt

from src.TDVP import TrainingConfig, train_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TDVP from a fully polarized spin chain with AdamW and plot the loss."
        )
    )
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-sites", type=int, default=10)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--optimizer-name", type=str, default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adamw-b1", type=float, default=0.9)
    parser.add_argument("--adamw-b2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--n-samples-per-chain", type=int, default=16)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--t-initial", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=0.0)
    parser.add_argument("--time-steps", type=int, default=1)
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
        J=1.0,
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
        time_loss_mode="sum",
        checkpoint_dir=None,
        seed=args.seed,
    )

    print(
        f"Running TDVP with {config.optimizer_name.upper()}: "
        f"lr={config.learning_rate}, "
        f"weight_decay={config.weight_decay}, "
        f"b1={config.adamw_b1}, "
        f"b2={config.adamw_b2}, "
        f"eps={config.adamw_eps}, "
        f"time_loss_mode={config.time_loss_mode}"
    )

    result = train_loop(config, verbose=True)
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


if __name__ == "__main__":
    main()
