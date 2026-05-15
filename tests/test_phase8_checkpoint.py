from pathlib import Path

import jax.numpy as jnp

from src.hamiltonian import LongRangeTransverseIsingHamiltonian
from src.TDVP import (
    TrainingConfig,
    load_training_checkpoint,
    save_training_checkpoint,
    train_loop,
)
from src.wavefunction import AutoregressiveNQS, NeuralGalerkinNQS


def _make_config():
    return TrainingConfig(
        N=4,
        J=1.0,
        h=0.5,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        learning_rate=1e-3,
        n_steps=2,
        n_samples_per_chain=3,
        burn_in=2,
        thinning=1,
        n_chains=2,
        time_steps=1,
        seed=3,
    )


def test_save_and_load_checkpoint_preserves_wavefunction_outputs(tmp_path: Path):
    config = _make_config()
    result = train_loop(config, verbose=False)

    checkpoint_path = tmp_path / "tdvp_checkpoint.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=result["wavefunction"],
        ham=result["hamiltonian"],
        config=result["config"],
        metrics_history=result["metrics_history"],
        global_step=result["global_step"],
        completed_time_steps=result["completed_time_steps"],
        current_time=0.0,
        chain_configurations=result["final_configurations"],
        rng=result["rng"],
        opt_state=result["opt_state"],
    )

    loaded = load_training_checkpoint(str(checkpoint_path))

    configs = jnp.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=jnp.int32,
    )
    t = jnp.float32(0.25)

    logp_before, phi_before = result["wavefunction"](configs, t)
    logp_after, phi_after = loaded["wavefunction"](configs, t)

    assert jnp.allclose(logp_before, logp_after, atol=1e-6)
    assert jnp.allclose(phi_before, phi_after, atol=1e-6)
    assert loaded["global_step"] == result["global_step"]
    assert loaded["completed_time_steps"] == result["completed_time_steps"]
    assert loaded["optimizer_state"] is not None
    assert jnp.array_equal(
        loaded["chain_configurations"], result["final_configurations"]
    )


def test_loaded_checkpoint_can_resume_training(tmp_path: Path):
    config = _make_config()
    first = train_loop(config, verbose=False)

    checkpoint_path = tmp_path / "resume_checkpoint.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=first["wavefunction"],
        ham=first["hamiltonian"],
        config=first["config"],
        metrics_history=first["metrics_history"],
        global_step=first["global_step"],
        completed_time_steps=0,
        current_time=0.0,
        chain_configurations=first["final_configurations"],
        rng=first["rng"],
        opt_state=first["opt_state"],
    )

    loaded = load_training_checkpoint(str(checkpoint_path))
    resume_config = loaded["config"]
    resume_config.n_steps = 1
    resume_config.time_steps = 1

    resumed = train_loop(
        resume_config,
        verbose=False,
        initial_wavefunction=loaded["wavefunction"],
        initial_hamiltonian=loaded["hamiltonian"],
        initial_metrics_history=loaded["metrics_history"],
        initial_global_step=loaded["global_step"],
        initial_chain_configurations=loaded["chain_configurations"],
        initial_opt_state=loaded["optimizer_state"],
        initial_rng=loaded["rng"],
        start_time_index=loaded["completed_time_steps"],
    )

    assert resumed["global_step"] == first["global_step"] + resume_config.n_steps
    assert len(resumed["metrics_history"]["loss"]) == len(
        first["metrics_history"]["loss"]
    ) + resume_config.n_steps


def test_train_loop_writes_real_checkpoint_files(tmp_path: Path):
    config = _make_config()
    config.checkpoint_dir = str(tmp_path)
    config.checkpoint_interval = 1

    _ = train_loop(config, verbose=False)

    ckpt1 = tmp_path / "checkpoint_step0001.pkl"
    ckpt2 = tmp_path / "checkpoint_step0002.pkl"
    assert ckpt1.exists()
    assert ckpt2.exists()


def test_muon_optimizer_state_roundtrip(tmp_path: Path):
    config = _make_config()
    config.optimizer_name = "muon"
    config.n_steps = 1
    config.time_steps = 1

    result = train_loop(config, verbose=False)

    checkpoint_path = tmp_path / "muon_checkpoint.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=result["wavefunction"],
        ham=result["hamiltonian"],
        config=result["config"],
        metrics_history=result["metrics_history"],
        global_step=result["global_step"],
        completed_time_steps=result["completed_time_steps"],
        current_time=0.0,
        chain_configurations=result["final_configurations"],
        rng=result["rng"],
        opt_state=result["opt_state"],
    )

    loaded = load_training_checkpoint(str(checkpoint_path))
    assert loaded["optimizer_state"] is not None
    assert loaded["config"].optimizer_name == "muon"


def test_long_range_hamiltonian_checkpoint_roundtrip(tmp_path: Path):
    config = _make_config()
    config.n_steps = 1
    ham = LongRangeTransverseIsingHamiltonian(
        J=config.J,
        h=config.h,
        N=config.N,
        alpha=1.4,
    )
    result = train_loop(config, verbose=False, initial_hamiltonian=ham)

    checkpoint_path = tmp_path / "lrtfim_checkpoint.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=result["wavefunction"],
        ham=result["hamiltonian"],
        config=result["config"],
        metrics_history=result["metrics_history"],
        global_step=result["global_step"],
        completed_time_steps=result["completed_time_steps"],
        current_time=0.0,
        chain_configurations=result["final_configurations"],
        rng=result["rng"],
        opt_state=result["opt_state"],
    )

    loaded = load_training_checkpoint(str(checkpoint_path))

    assert isinstance(loaded["hamiltonian"], LongRangeTransverseIsingHamiltonian)
    assert loaded["hamiltonian"].alpha == 1.4


def test_autoregressive_wavefunction_checkpoint_roundtrip(tmp_path: Path):
    import flax.nnx as nnx

    config = _make_config()
    config.n_steps = 1
    config.burn_in = 0
    config.thinning = 1
    wf = AutoregressiveNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(config.seed),
    )
    result = train_loop(config, verbose=False, initial_wavefunction=wf)

    checkpoint_path = tmp_path / "ar_checkpoint.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=result["wavefunction"],
        ham=result["hamiltonian"],
        config=result["config"],
        metrics_history=result["metrics_history"],
        global_step=result["global_step"],
        completed_time_steps=result["completed_time_steps"],
        current_time=0.0,
        chain_configurations=result["final_configurations"],
        rng=result["rng"],
        opt_state=result["opt_state"],
    )

    loaded = load_training_checkpoint(str(checkpoint_path))

    assert isinstance(loaded["wavefunction"], AutoregressiveNQS)


def test_neural_galerkin_checkpoint_roundtrip(tmp_path: Path):
    import flax.nnx as nnx

    config = _make_config()
    config.n_steps = 1
    config.num_galerkin_basis = 2
    config.num_galerkin_modes = 2
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
    result = train_loop(config, verbose=False, initial_wavefunction=wf)

    checkpoint_path = tmp_path / "neural_galerkin_checkpoint.pkl"
    save_training_checkpoint(
        str(checkpoint_path),
        wf=result["wavefunction"],
        ham=result["hamiltonian"],
        config=result["config"],
        metrics_history=result["metrics_history"],
        global_step=result["global_step"],
        completed_time_steps=result["completed_time_steps"],
        current_time=0.0,
        chain_configurations=result["final_configurations"],
        rng=result["rng"],
        opt_state=result["opt_state"],
    )

    loaded = load_training_checkpoint(str(checkpoint_path))
    configs = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)

    logp_before, phi_before = result["wavefunction"](configs, jnp.float32(0.3))
    logp_after, phi_after = loaded["wavefunction"](configs, jnp.float32(0.3))

    assert isinstance(loaded["wavefunction"], NeuralGalerkinNQS)
    assert loaded["config"].num_galerkin_basis == 2
    assert loaded["config"].num_galerkin_modes == 2
    assert jnp.allclose(logp_before, logp_after, atol=1e-6)
    assert jnp.allclose(phi_before, phi_after, atol=1e-6)
