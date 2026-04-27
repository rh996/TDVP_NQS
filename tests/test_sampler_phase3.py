import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.sampler import (
    autoregressive_trajectory_sample,
    metropolis_hastings_sample,
    metropolis_hastings_trajectory,
)
from src.wavefunction import AutoregressiveNQS


class DummyWF:
    """Deterministic toy wavefunction exposing main API."""

    def __init__(self, alpha=0.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, configuration, t):
        cfg = jnp.asarray(configuration)
        t = jnp.asarray(t, dtype=jnp.float32)
        if cfg.ndim == 1:
            logp = self.alpha * jnp.sum(cfg, dtype=jnp.float32) + self.beta * t
        elif cfg.ndim == 2:
            logp = self.alpha * jnp.sum(cfg, axis=1, dtype=jnp.float32) + self.beta * t
        else:
            raise ValueError(f"Unexpected configuration shape: {cfg.shape}")
        
        return logp, jnp.zeros_like(logp)


def test_sampler_trajectory_warm_starting():
    key = jax.random.PRNGKey(42)
    wf = DummyWF(alpha=0.1, beta=-0.05)
    n_sites = 4
    n_chains = 2
    times = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    init = jnp.zeros((n_chains, n_sites), dtype=jnp.int32)

    n_samples = 5
    burn_in = 10
    thinning = 2

    all_samples, all_stats, final_configs = metropolis_hastings_trajectory(
        wf=wf,
        initial_configurations=init,
        times=times,
        n_sites=n_sites,
        n_samples=n_samples,
        burn_in=burn_in,
        thinning=thinning,
        key=key,
    )

    # Output shapes
    assert all_samples.shape == (len(times), n_chains, n_samples, n_sites)
    assert final_configs.shape == (n_chains, n_sites)
    
    # Warm start check: last sample of T=0 should be starting point for T=1 internals, 
    # but since we return samples after thinning/burn_in, we check continuity of final state.
    # The final config of slice 0 is the starting point of slice 1.
    assert jnp.array_equal(final_configs, all_samples[-1, :, -1, :])


def test_autoregressive_trajectory_preserves_chain_shape():
    key = jax.random.PRNGKey(0)
    n_sites = 4
    n_chains = 3
    n_samples = 5
    wf = AutoregressiveNQS(
        N=n_sites,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        rngs=nnx.Rngs(0),
    )

    all_samples, all_stats, final_configs = autoregressive_trajectory_sample(
        wf=wf,
        times=jnp.array([0.0, 0.25], dtype=jnp.float32),
        n_sites=n_sites,
        batch_size=n_chains * n_samples,
        key=key,
        n_chains=n_chains,
    )

    assert all_samples.shape == (2, n_chains, n_samples, n_sites)
    assert final_configs.shape == (n_chains, n_sites)
    assert jnp.all((all_samples == 0) | (all_samples == 1))
    assert all_stats["acceptance_rate"].shape == (2, n_chains)
    assert int(all_stats["n_chains"][0]) == n_chains
    assert int(all_stats["n_samples_per_chain"][0]) == n_samples


def test_autoregressive_trajectory_unique_counts_are_weighted_batch():
    key = jax.random.PRNGKey(1)
    n_sites = 3
    n_chains = 3
    n_samples = 4
    wf = AutoregressiveNQS(
        N=n_sites,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        rngs=nnx.Rngs(1),
    )

    all_samples, all_stats, final_configs = autoregressive_trajectory_sample(
        wf=wf,
        times=jnp.array([0.0, 0.25], dtype=jnp.float32),
        n_sites=n_sites,
        batch_size=n_chains * n_samples,
        key=key,
        n_chains=n_chains,
        use_unique=True,
    )

    assert all_samples.shape == (2, n_chains, n_samples, n_sites)
    assert final_configs.shape == (n_chains, n_sites)
    assert all_stats["sample_counts"].shape == (2, n_chains, n_samples)
    assert jnp.all(jnp.sum(all_stats["sample_counts"], axis=(1, 2)) == n_chains * n_samples)
    assert jnp.all(all_stats["unique_count"] <= 2**n_sites)
    assert jnp.all(all_stats["unique_fraction"] <= 1.0)


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
