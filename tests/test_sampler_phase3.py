import jax
import jax.numpy as jnp

from src.sampler import metropolis_hastings_sample


class DummyWF:
    """Deterministic toy wavefunction exposing log_prob API."""

    def __init__(self, alpha=0.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, configuration, t):
        cfg = jnp.asarray(configuration)
        t = jnp.asarray(t, dtype=jnp.float32)
        if cfg.ndim == 1:
            return self.alpha * jnp.sum(cfg, dtype=jnp.float32) + self.beta * t
        if cfg.ndim == 2:
            return self.alpha * jnp.sum(cfg, axis=1, dtype=jnp.float32) + self.beta * t
        raise ValueError(f"Unexpected configuration shape: {cfg.shape}")

    def __call__(self, configuration, t):
        # Included for compatibility with broader codebase expectations.
        return self.log_prob(configuration, t), jnp.asarray(0.0, dtype=jnp.float32)

    def phase(self, configuration, t):
        cfg = jnp.asarray(configuration)
        if cfg.ndim == 1:
            return jnp.asarray(0.0, dtype=jnp.float32)
        if cfg.ndim == 2:
            return jnp.zeros((cfg.shape[0],), dtype=jnp.float32)
        raise ValueError(f"Unexpected configuration shape: {cfg.shape}")


def test_sampler_shapes_and_bit_domain_single_chain():
    key = jax.random.PRNGKey(0)
    wf = DummyWF(alpha=0.2, beta=0.1)
    n_sites = 6
    init = jnp.array([0, 1, 0, 1, 1, 0], dtype=jnp.int32)

    samples, stats = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=init,
        t=jnp.float32(0.3),
        n_sites=n_sites,
        n_samples=20,
        burn_in=10,
        thinning=2,
        key=key,
        return_stats=True,
    )

    assert samples.shape == (1, 20, n_sites)
    assert samples.dtype == jnp.int32
    assert jnp.all((samples == 0) | (samples == 1))

    assert "acceptance_rate" in stats
    assert "accepted_moves" in stats
    assert "proposed_moves" in stats

    assert stats["acceptance_rate"].shape == (1,)
    assert float(stats["acceptance_rate"][0]) >= 0.0
    assert float(stats["acceptance_rate"][0]) <= 1.0


def test_sampler_shapes_and_stats_multi_chain():
    key = jax.random.PRNGKey(1)
    wf = DummyWF(alpha=-0.1)
    n_sites = 5
    n_chains = 4

    init = jnp.array(
        [
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )

    n_samples = 12
    burn_in = 8
    thinning = 3
    total_steps = burn_in + n_samples * thinning

    samples, stats = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=init,
        t=jnp.float32(0.0),
        n_sites=n_sites,
        n_samples=n_samples,
        burn_in=burn_in,
        thinning=thinning,
        key=key,
        return_stats=True,
    )

    assert samples.shape == (n_chains, n_samples, n_sites)
    assert jnp.all((samples == 0) | (samples == 1))

    assert stats["acceptance_rate"].shape == (n_chains,)
    assert stats["accepted_moves"].shape == (n_chains,)
    assert int(stats["proposed_moves"]) == total_steps
    assert int(stats["n_chains"]) == n_chains
    assert int(stats["n_samples_per_chain"]) == n_samples
    assert int(stats["burn_in"]) == burn_in
    assert int(stats["thinning"]) == thinning

    assert jnp.all(stats["acceptance_rate"] >= 0.0)
    assert jnp.all(stats["acceptance_rate"] <= 1.0)
    assert jnp.all(stats["accepted_moves"] >= 0)
    assert jnp.all(stats["accepted_moves"] <= total_steps)


def test_sampler_deterministic_given_same_seed():
    wf = DummyWF(alpha=0.15, beta=0.02)
    n_sites = 7
    init = jnp.array(
        [
            [0, 0, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, 1],
        ],
        dtype=jnp.int32,
    )
    key = jax.random.PRNGKey(42)

    kwargs = dict(
        wf=wf,
        initial_configurations=init,
        t=jnp.float32(0.9),
        n_sites=n_sites,
        n_samples=16,
        burn_in=5,
        thinning=2,
        return_stats=True,
    )

    samples1, stats1 = metropolis_hastings_sample(key=key, **kwargs)
    samples2, stats2 = metropolis_hastings_sample(key=key, **kwargs)

    assert jnp.array_equal(samples1, samples2)
    assert jnp.array_equal(stats1["accepted_moves"], stats2["accepted_moves"])
    assert jnp.allclose(stats1["acceptance_rate"], stats2["acceptance_rate"], atol=1e-8)


def test_sampler_burn_in_changes_stream_for_same_seed():
    wf = DummyWF(alpha=0.25)
    n_sites = 8
    init = jnp.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    key = jax.random.PRNGKey(7)

    samples_no_burn, _ = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=init,
        t=jnp.float32(0.0),
        n_sites=n_sites,
        n_samples=25,
        burn_in=0,
        thinning=1,
        key=key,
        return_stats=True,
    )
    samples_with_burn, _ = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=init,
        t=jnp.float32(0.0),
        n_sites=n_sites,
        n_samples=25,
        burn_in=30,
        thinning=1,
        key=key,
        return_stats=True,
    )

    assert not jnp.array_equal(samples_no_burn, samples_with_burn)


def test_sampler_input_validation():
    wf = DummyWF()
    key = jax.random.PRNGKey(0)

    # Bad config rank
    bad_rank = jnp.zeros((2, 3, 4), dtype=jnp.int32)
    try:
        _ = metropolis_hastings_sample(
            wf=wf,
            initial_configurations=bad_rank,
            t=0.0,
            n_sites=4,
            n_samples=10,
            burn_in=0,
            thinning=1,
            key=key,
        )
        assert False, "Expected ValueError for rank-3 configurations."
    except ValueError as e:
        assert "rank 1 or 2" in str(e)

    # Bad n_samples
    try:
        _ = metropolis_hastings_sample(
            wf=wf,
            initial_configurations=jnp.zeros((4,), dtype=jnp.int32),
            t=0.0,
            n_sites=4,
            n_samples=0,
            burn_in=0,
            thinning=1,
            key=key,
        )
        assert False, "Expected ValueError for n_samples <= 0."
    except ValueError as e:
        assert "n_samples" in str(e)

    # Bad thinning
    try:
        _ = metropolis_hastings_sample(
            wf=wf,
            initial_configurations=jnp.zeros((4,), dtype=jnp.int32),
            t=0.0,
            n_sites=4,
            n_samples=5,
            burn_in=0,
            thinning=0,
            key=key,
        )
        assert False, "Expected ValueError for thinning <= 0."
    except ValueError as e:
        assert "thinning" in str(e)

    # Non-binary values
    try:
        _ = metropolis_hastings_sample(
            wf=wf,
            initial_configurations=jnp.array([0, 1, 2, 0], dtype=jnp.int32),
            t=0.0,
            n_sites=4,
            n_samples=5,
            burn_in=0,
            thinning=1,
            key=key,
        )
        assert False, "Expected ValueError for non-binary configurations."
    except ValueError as e:
        assert "binary bits in {0,1}" in str(e)
