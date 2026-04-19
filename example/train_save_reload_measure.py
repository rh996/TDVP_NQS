import argparse
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.TDVP import (
    TrainingConfig,
    load_training_checkpoint,
    save_training_checkpoint,
    train_loop,
)
from src.observables import exact_observables_from_wf, sample_and_measure_observables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a TDVP wavefunction, save a checkpoint, reload it, optionally "
            "resume optimization with AdamW, and measure <Z> / <X>."
        )
    )
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--resume-steps", type=int, default=25)
    parser.add_argument("--n-sites", type=int, default=4)
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--measure-samples", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adamw-b1", type=float, default=0.9)
    parser.add_argument("--adamw-b2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "example" / "outputs-save-reload-measure",
    )
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _observables_to_dict(obs) -> dict:
    return {
        "z_total": float(obs.z_total),
        "z_mean": float(obs.z_mean),
        "x_total_real": float(obs.x_total_real),
        "x_total_imag": float(obs.x_total_imag),
        "x_mean_real": float(obs.x_mean_real),
        "x_mean_imag": float(obs.x_mean_imag),
        "n_samples": int(obs.n_samples),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sites = args.n_sites
    n_chains = args.n_chains
    fully_polarized = jnp.ones((n_chains, n_sites), dtype=jnp.int32)
    t_measure = jnp.float32(0.0)

    config = TrainingConfig(
        N=n_sites,
        J=1.0,
        h=0.5,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        learning_rate=args.learning_rate,
        optimizer_name="adamw",
        adamw_b1=args.adamw_b1,
        adamw_b2=args.adamw_b2,
        adamw_eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        n_steps=args.train_steps,
        n_samples_per_chain=16,
        burn_in=50,
        thinning=1,
        n_chains=n_chains,
        initial_chain_configurations=fully_polarized,
        t_initial=0.0,
        t_final=0.0,
        time_steps=1,
        time_loss_mode="sum",
        checkpoint_dir=None,
        seed=args.seed,
    )

    print(
        "Running TDVP with AdamW: "
        f"lr={config.learning_rate}, "
        f"weight_decay={config.weight_decay}, "
        f"b1={config.adamw_b1}, "
        f"b2={config.adamw_b2}, "
        f"eps={config.adamw_eps}, "
        f"time_loss_mode={config.time_loss_mode}"
    )

    print("=== Phase 1: initial training ===")
    trained = train_loop(config, verbose=True)

    checkpoint_path = output_dir / "trained_wavefunction.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=trained["wavefunction"],
        ham=trained["hamiltonian"],
        config=trained["config"],
        metrics_history=trained["metrics_history"],
        global_step=trained["global_step"],
        completed_time_steps=trained["completed_time_steps"],
        current_time=float(t_measure),
        chain_configurations=trained["final_configurations"],
        rng=trained["rng"],
        opt_state=trained["opt_state"],
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    print("\n=== Phase 2: reload checkpoint ===")
    loaded = load_training_checkpoint(str(checkpoint_path))

    resumed = loaded
    if args.resume_steps > 0:
        print(f"\n=== Phase 3: resume for {args.resume_steps} more steps ===")
        resume_config = loaded["config"]
        resume_config.n_steps = args.resume_steps
        resume_config.time_steps = 1
        resume_config.t_initial = 0.0
        resume_config.t_final = 0.0

        resumed_result = train_loop(
            resume_config,
            verbose=True,
            initial_wavefunction=loaded["wavefunction"],
            initial_hamiltonian=loaded["hamiltonian"],
            initial_metrics_history=loaded["metrics_history"],
            initial_global_step=loaded["global_step"],
            initial_chain_configurations=loaded["chain_configurations"],
            initial_opt_state=loaded["optimizer_state"],
            initial_rng=loaded["rng"],
            start_time_index=0,
        )
        resumed = {
            "wavefunction": resumed_result["wavefunction"],
            "hamiltonian": resumed_result["hamiltonian"],
            "config": resumed_result["config"],
            "metrics_history": resumed_result["metrics_history"],
            "global_step": resumed_result["global_step"],
            "completed_time_steps": resumed_result["completed_time_steps"],
            "chain_configurations": resumed_result["final_configurations"],
            "rng": resumed_result["rng"],
        }

        resumed_checkpoint_path = output_dir / "resumed_wavefunction.pkl"
        save_training_checkpoint(
            str(resumed_checkpoint_path),
            wf=resumed["wavefunction"],
            ham=resumed["hamiltonian"],
            config=resumed["config"],
            metrics_history=resumed["metrics_history"],
            global_step=resumed["global_step"],
            completed_time_steps=resumed["completed_time_steps"],
            current_time=float(t_measure),
            chain_configurations=resumed["chain_configurations"],
            rng=resumed["rng"],
            opt_state=resumed_result["opt_state"],
        )
        print(f"Saved resumed checkpoint to {resumed_checkpoint_path}")

    print("\n=== Phase 4: measure observables ===")
    measure_key = jnp.array([0, args.seed + 123], dtype=jnp.uint32)
    observables_mc, sampler_stats, final_configs = sample_and_measure_observables(
        resumed["wavefunction"],
        t_measure,
        n_sites=n_sites,
        initial_configurations=resumed["chain_configurations"],
        n_samples=args.measure_samples,
        burn_in=50,
        thinning=1,
        key=measure_key,
    )

    observables_payload = {
        "optimizer": {
            "name": resumed["config"].optimizer_name,
            "learning_rate": float(resumed["config"].learning_rate),
            "weight_decay": float(resumed["config"].weight_decay),
            "adamw_b1": float(resumed["config"].adamw_b1),
            "adamw_b2": float(resumed["config"].adamw_b2),
            "adamw_eps": float(resumed["config"].adamw_eps),
            "time_loss_mode": resumed["config"].time_loss_mode,
        },
        "training_steps_initial": int(args.train_steps),
        "training_steps_resumed": int(max(args.resume_steps, 0)),
        "global_step_final": int(resumed["global_step"]),
        "monte_carlo": _observables_to_dict(observables_mc),
        "sampler": {
            "acceptance_rate_mean": float(jnp.mean(sampler_stats["acceptance_rate"])),
            "n_chains": int(sampler_stats["n_chains"]),
            "n_samples_per_chain": int(sampler_stats["n_samples_per_chain"]),
            "burn_in": int(sampler_stats["burn_in"]),
            "thinning": int(sampler_stats["thinning"]),
        },
    }

    if n_sites <= 10:
        observables_exact = exact_observables_from_wf(
            resumed["wavefunction"], n_sites, t_measure
        )
        observables_payload["exact"] = _observables_to_dict(observables_exact)

    observables_path = output_dir / "observables.json"
    _write_json(observables_path, observables_payload)

    metrics = resumed["metrics_history"]
    loss_plot_path = output_dir / "loss_curve.png"
    plt.figure(figsize=(9, 5))
    plt.plot(metrics["step"], metrics["loss"], linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("TDVP Loss")
    plt.title("Train → Save → Reload → Measure Workflow (AdamW)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "observables_path": str(observables_path),
        "loss_plot_path": str(loss_plot_path),
        "optimizer_name": resumed["config"].optimizer_name,
        "time_loss_mode": resumed["config"].time_loss_mode,
        "final_configurations_shape": list(final_configs.shape),
        "initial_loss": float(metrics["loss"][0]),
        "final_loss": float(metrics["loss"][-1]),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    print(f"Saved observables to {observables_path}")
    print(f"Saved loss curve to {loss_plot_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
