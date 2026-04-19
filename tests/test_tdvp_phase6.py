import flax.nnx as nnx
import jax
import jax.numpy as jnp

from src.TDVP import TrainingConfig, train_loop, train_step
from src.hamiltonian import TransverseIsingHamiltonian
from src.wavefunction import tSpinNQS


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


def test_train_step_returns_finite_metrics_and_next_rng():
    config = _make_config()
    wf, ham = _make_wf_and_ham(config)
    rng = jax.random.PRNGKey(123)

    loss, diag, next_rng = train_step(
        wf=wf,
        ham=ham,
        t=0.25,
        config=config,
        rng=rng,
    )

    assert jnp.asarray(loss).ndim == 0
    assert jnp.isfinite(loss)
    assert next_rng.shape == rng.shape

    required = {
        "loss",
        "acceptance_rate",
        "grad_norm_pathwise",
        "grad_norm_covariance",
        "grad_norm_total",
        "ell_mean",
        "ell_std",
        "finite_loss",
        "finite_grads",
        "e_real_mean",
        "e_imag_mean",
    }
    assert required.issubset(set(diag.keys()))
    assert bool(diag["finite_loss"])
    assert bool(diag["finite_grads"])
    assert 0.0 <= float(diag["acceptance_rate"]) <= 1.0
    assert jnp.isfinite(diag["e_real_mean"])
    assert jnp.isfinite(diag["e_imag_mean"])


def test_train_loop_runs_and_records_metrics_history():
    config = _make_config()

    result = train_loop(config, verbose=False)

    assert set(result.keys()) == {
        "wavefunction",
        "hamiltonian",
        "optimizer",
        "opt_state",
        "metrics_history",
        "config",
    }
    assert result["optimizer"] is None
    assert result["opt_state"] is None

    metrics = result["metrics_history"]
    expected_len = config.n_steps * config.time_steps

    for key in (
        "loss",
        "acceptance_rate",
        "grad_norm_pathwise",
        "grad_norm_covariance",
        "grad_norm_total",
        "ell_mean",
        "ell_std",
        "finite_loss",
        "finite_grads",
        "e_real_mean",
        "e_imag_mean",
        "time",
        "step",
    ):
        assert len(metrics[key]) == expected_len

    assert all(value >= 0.0 for value in metrics["grad_norm_total"])
    assert all(value in (0.0, 1.0) for value in metrics["finite_loss"])
    assert all(value in (0.0, 1.0) for value in metrics["finite_grads"])


def test_train_loop_validates_configuration():
    bad = _make_config()
    bad.learning_rate = 0.0

    try:
        train_loop(bad, verbose=False)
        assert False, "Expected ValueError for invalid learning rate."
    except ValueError as e:
        assert "learning_rate must be > 0" in str(e)
