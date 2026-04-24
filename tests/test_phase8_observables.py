import jax
import jax.numpy as jnp

from src.observables import (
    exact_observables_from_wf,
    measure_observables,
    sample_and_measure_observables,
)


class DummyWavefunction:
    """Deterministic wavefunction with closed-form logp / phase."""

    def __init__(self, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

    def _sum_bits(self, configuration):
        cfg = jnp.asarray(configuration)
        if cfg.ndim == 1:
            return jnp.sum(cfg, dtype=jnp.float32)
        if cfg.ndim == 2:
            return jnp.sum(cfg, axis=1, dtype=jnp.float32)
        raise ValueError(f"Unexpected configuration shape: {cfg.shape}")

    def __call__(self, configuration, t):
        logp = self.alpha * self._sum_bits(configuration) + self.beta * jnp.asarray(
            t, dtype=jnp.float32
        )
        phi = self.gamma * self._sum_bits(configuration) + self.delta * jnp.asarray(
            t, dtype=jnp.float32
        )
        return logp, phi


def test_measure_observables_handles_per_site_values():
    wf = DummyWavefunction(alpha=0.0, gamma=0.0)
    configs = jnp.array([[0, 1, 0, 1]], dtype=jnp.int32)
    
    obs = measure_observables(wf, configs, t=0.0, n_sites=4)
    
    # Z per site: 1 - 2*bits -> [1, -1, 1, -1]
    assert jnp.allclose(obs.z_sites, jnp.array([1.0, -1.0, 1.0, -1.0]))
    assert obs.x_sites_real.shape == (4,)
    assert obs.x_sites_imag.shape == (4,)


def test_measure_observables_matches_manual_formula_on_samples():
    wf = DummyWavefunction(alpha=0.4, gamma=0.3)
    configs = jnp.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=jnp.int32,
    )

    obs = measure_observables(wf, configs, t=jnp.float32(0.0), n_sites=3)

    spins = 1.0 - 2.0 * configs.astype(jnp.float32)
    expected_z_total = jnp.mean(jnp.sum(spins, axis=1))

    delta_sum = 1.0 - 2.0 * configs.astype(jnp.float32)
    delta_logp = wf.alpha * delta_sum
    delta_phi = wf.gamma * delta_sum
    x_real_per_sample = jnp.sum(
        jnp.exp(0.5 * delta_logp) * jnp.cos(delta_phi), axis=1
    )
    x_imag_per_sample = jnp.sum(
        jnp.exp(0.5 * delta_logp) * jnp.sin(delta_phi), axis=1
    )

    assert jnp.allclose(obs.z_total, expected_z_total, atol=1e-6)
    assert jnp.allclose(obs.z_mean, expected_z_total / 3.0, atol=1e-6)
    assert jnp.allclose(obs.x_total_real, jnp.mean(x_real_per_sample), atol=1e-6)
    assert jnp.allclose(obs.x_total_imag, jnp.mean(x_imag_per_sample), atol=1e-6)
    assert obs.n_samples == configs.shape[0]


def test_exact_observables_match_manual_uniform_state():
    wf = DummyWavefunction(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    obs = exact_observables_from_wf(wf, n_sites=3, t=jnp.float32(0.0))

    assert jnp.allclose(obs.z_total, 0.0, atol=1e-6)
    assert jnp.allclose(obs.z_mean, 0.0, atol=1e-6)
    assert jnp.allclose(obs.x_total_real, 3.0, atol=1e-6)
    assert jnp.allclose(obs.x_total_imag, 0.0, atol=1e-6)
    assert jnp.allclose(obs.x_mean_real, 1.0, atol=1e-6)


def test_sample_and_measure_observables_returns_shapes_and_stats():
    wf = DummyWavefunction(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
    initial = jnp.zeros((2, 4), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)

    obs, stats, final_configs = sample_and_measure_observables(
        wf,
        t=jnp.float32(0.1),
        n_sites=4,
        initial_configurations=initial,
        n_samples=3,
        burn_in=2,
        thinning=1,
        key=key,
    )

    assert final_configs.shape == (2, 4)
    assert obs.n_samples == 6
    assert 0.0 <= float(jnp.mean(stats["acceptance_rate"])) <= 1.0
