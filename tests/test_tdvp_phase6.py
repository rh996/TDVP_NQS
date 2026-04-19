import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from src.TDVP import TrainingConfig, _create_optimizer, train_loop, train_step
from src.hamiltonian import TransverseIsingHamiltonian
from src.sampler import metropolis_hastings_sample
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
    optimizer = _create_optimizer(config)
    opt_state = optimizer.init(nnx.state(wf.model, nnx.Param))
    rng = jax.random.PRNGKey(123)

    loss, diag, next_opt_state, next_rng, next_configs = train_step(
        wf=wf,
        ham=ham,
        t=0.25,
        config=config,
        optimizer=optimizer,
        opt_state=opt_state,
        rng=rng,
    )

    assert jnp.asarray(loss).ndim == 0
    assert jnp.isfinite(loss)
    assert next_rng.shape == rng.shape
    assert next_configs.shape == (config.n_chains, config.N)
    assert jnp.all((next_configs == 0) | (next_configs == 1))
    assert next_opt_state is not None

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


def test_train_loop_preserves_chain_state_between_steps():
    config = _make_config()
    config.n_steps = 2
    config.time_steps = 1
    config.n_samples_per_chain = 2
    config.burn_in = 1
    config.thinning = 1
    config.seed = 7

    wf, ham = _make_wf_and_ham(config)
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        b1=config.adamw_b1,
        b2=config.adamw_b2,
        eps=config.adamw_eps,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(nnx.state(wf.model, nnx.Param))
    t = jnp.float32(config.t_initial)
    rng = jax.random.PRNGKey(config.seed)
    rng, _ = jax.random.split(rng)  # matches train_loop model-init split

    initial_configs = jnp.zeros((config.n_chains, config.N), dtype=jnp.int32)

    rng, step_rng1 = jax.random.split(rng)
    _, _, opt_state_after_1, rng_after_1, next_configs_1 = train_step(
        wf=wf,
        ham=ham,
        t=float(t),
        config=config,
        optimizer=optimizer,
        opt_state=opt_state,
        rng=step_rng1,
        initial_configurations=initial_configs,
    )

    rng_after_1, step_rng2 = jax.random.split(rng_after_1)
    _, _, _, _, next_configs_2 = train_step(
        wf=wf,
        ham=ham,
        t=float(t),
        config=config,
        optimizer=optimizer,
        opt_state=opt_state_after_1,
        rng=step_rng2,
        initial_configurations=next_configs_1,
    )

    loop_result = train_loop(config, verbose=False)
    assert jnp.array_equal(loop_result["final_configurations"], next_configs_2)


def test_train_step_uses_provided_chain_state():
    config = _make_config()
    config.learning_rate = 0.0
    config.n_samples_per_chain = 1
    config.burn_in = 0
    config.thinning = 1

    wf, ham = _make_wf_and_ham(config)
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        b1=config.adamw_b1,
        b2=config.adamw_b2,
        eps=config.adamw_eps,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(nnx.state(wf.model, nnx.Param))
    t = 0.25
    rng = jax.random.PRNGKey(123)
    initial_configs = jnp.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )

    expected_rng, rng_sample = jax.random.split(rng)
    expected_samples, _ = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=initial_configs,
        t=jnp.asarray(t, dtype=jnp.float32),
        n_sites=ham.N,
        n_samples=config.n_samples_per_chain,
        burn_in=config.burn_in,
        thinning=config.thinning,
        key=rng_sample,
        return_stats=True,
    )

    _, _, _, next_rng, next_configs = train_step(
        wf=wf,
        ham=ham,
        t=t,
        config=config,
        optimizer=optimizer,
        opt_state=opt_state,
        rng=rng,
        initial_configurations=initial_configs,
    )

    assert jnp.array_equal(next_rng, expected_rng)
    assert jnp.array_equal(next_configs, expected_samples[:, -1, :])


def test_train_loop_supports_muon_optimizer():
    config = _make_config()
    config.optimizer_name = "muon"
    config.n_steps = 1
    config.time_steps = 1
    result = train_loop(config, verbose=False)

    assert result["optimizer"] is not None
    assert result["opt_state"] is not None
    assert len(result["metrics_history"]["loss"]) == 1
