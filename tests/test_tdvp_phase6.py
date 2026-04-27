import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from src.TDVP import TrainingConfig, _create_optimizer, train_loop
from src.hamiltonian import TransverseIsingHamiltonian
from src.sampler import metropolis_hastings_trajectory
from src.wavefunction import AutoregressiveNQS, tSpinNQS


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
        time_steps=2,
        seed=0,
    )


def _make_wf_and_ham(config: TrainingConfig):
    wf = tSpinNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(config.seed),
    )
    ham = TransverseIsingHamiltonian(J=config.J, h=config.h, N=config.N)
    return wf, ham


def test_train_loop_runs_and_records_metrics_history():
    config = _make_config()
    expected_len = config.n_steps

    result = train_loop(config, verbose=False)

    assert set(result.keys()) == {
        "wavefunction",
        "hamiltonian",
        "optimizer",
        "opt_state",
        "metrics_history",
        "final_configurations",
        "global_step",
        "completed_time_steps",
        "rng",
        "config",
    }
    assert result["optimizer"] is not None
    assert result["opt_state"] is not None
    assert result["final_configurations"].shape == (config.n_chains, config.N)
    assert result["global_step"] == expected_len
    assert result["completed_time_steps"] == config.time_steps

    metrics = result["metrics_history"]

    for key in (
        "loss",
        "acceptance_rate",
        "grad_norm_total",
        "ell_mean",
        "e_real_mean",
        "e_imag_mean",
        "time",
        "step",
    ):
        assert len(metrics[key]) == expected_len

    assert all(value >= 0.0 for value in metrics["grad_norm_total"])


def test_train_loop_accepts_explicit_initial_chain_configurations():
    config = _make_config()
    config.initial_chain_configurations = jnp.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=jnp.int32,
    )

    result = train_loop(config, verbose=False)

    assert result["final_configurations"].shape == (config.n_chains, config.N)


def test_train_loop_validates_configuration():
    bad = _make_config()
    bad.learning_rate = 0.0

    try:
        train_loop(bad, verbose=False)
        assert False, "Expected ValueError for invalid learning rate."
    except ValueError as e:
        assert "learning_rate must be > 0" in str(e)

    bad = _make_config()
    bad.gradient_clip_norm = 0.0

    try:
        train_loop(bad, verbose=False)
        assert False, "Expected ValueError for invalid gradient clip norm."
    except ValueError as e:
        assert "gradient_clip_norm must be > 0" in str(e)


def test_train_loop_supports_muon_optimizer():
    config = _make_config()
    config.optimizer_name = "muon"
    config.n_steps = 1
    config.time_steps = 1
    result = train_loop(config, verbose=False)

    assert result["optimizer"] is not None
    assert result["opt_state"] is not None
    assert len(result["metrics_history"]["loss"]) == 1


def test_train_loop_with_anchoring():
    config = _make_config()
    config.pretrain_steps = 2
    config.lambda_ic = 1.0
    config.n_steps = 1
    
    result = train_loop(config, verbose=False)
    assert len(result["metrics_history"]["loss"]) == 1


def test_train_loop_supports_gradient_clipping():
    config = _make_config()
    config.gradient_clip_norm = 0.1
    config.n_steps = 1

    result = train_loop(config, verbose=False)

    assert result["optimizer"] is not None
    assert result["opt_state"] is not None
    assert len(result["metrics_history"]["loss"]) == 1
    assert result["config"].gradient_clip_norm == 0.1


def test_train_loop_supports_unique_autoregressive_samples():
    config = TrainingConfig(
        N=3,
        J=-1.0,
        h=0.5,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        learning_rate=1e-3,
        n_steps=1,
        n_samples_per_chain=2,
        burn_in=0,
        thinning=1,
        n_chains=2,
        time_steps=2,
        use_unique_ar_samples=True,
        seed=0,
    )
    wf = AutoregressiveNQS(
        N=config.N,
        Num_boxes=config.Num_boxes,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        rngs=nnx.Rngs(config.seed),
    )

    result = train_loop(config, verbose=False, initial_wavefunction=wf)

    assert result["final_configurations"].shape == (config.n_chains, config.N)
    assert len(result["metrics_history"]["loss"]) == 1
    assert len(result["metrics_history"]["ar_unique_count"]) == 1
    assert result["metrics_history"]["ar_unique_fraction"][0] <= 1.0
