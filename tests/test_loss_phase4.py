import jax.numpy as jnp

from src.hamiltonian import TransverseIsingHamiltonian
from src.loss import (
    tdvp_residual_components,
    tdvp_residual_loss,
    time_derivatives_autodiff,
)


class DummyWavefunction:
    """Deterministic wavefunction for phase-4 loss tests.

    logp(sigma, t) = alpha * sum(sigma) + beta * t
    phi(sigma, t)  = gamma * sum(sigma) + delta * t
    """

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

    def log_prob(self, configuration, t):
        t = jnp.asarray(t, dtype=jnp.float32)
        return self.alpha * self._sum_bits(configuration) + self.beta * t

    def phase(self, configuration, t):
        t = jnp.asarray(t, dtype=jnp.float32)
        return self.gamma * self._sum_bits(configuration) + self.delta * t

    def __call__(self, configuration, t):
        return self.log_prob(configuration, t), self.phase(configuration, t)


def _expected_from_manual_formula(ham, wf, configs):
    """Manual A/B construction using known closed form for DummyWavefunction."""
    configs = jnp.asarray(configs, dtype=jnp.int32)
    B, N = configs.shape

    # d/dt terms from dummy model
    dlogp_dt = jnp.full((B,), wf.beta, dtype=jnp.float32)
    dphi_dt = jnp.full((B,), wf.delta, dtype=jnp.float32)

    # diagonal ZZ with OBC and bit->sz map s = 1 - 2*bit
    s = 1.0 - 2.0 * configs.astype(jnp.float32)
    zz = ham.J * jnp.sum(s[:, :-1] * s[:, 1:], axis=1)

    # transverse contributions
    # flipping site i changes sum(bits) by +1 if bit=0 else -1
    delta_sum = 1.0 - 2.0 * configs.astype(jnp.float32)  # shape (B, N): +1/-1
    delta_logp = wf.alpha * delta_sum
    delta_phi = wf.gamma * delta_sum

    mag = jnp.exp(0.5 * delta_logp)
    x_real = ham.h * jnp.sum(mag * jnp.cos(delta_phi), axis=1)
    x_imag = ham.h * jnp.sum(mag * jnp.sin(delta_phi), axis=1)

    e_real = zz + x_real
    e_imag = x_imag

    A = 0.5 * dlogp_dt - e_imag
    Bv = dphi_dt + e_real

    A_mean = jnp.mean(A)
    B_mean = jnp.mean(Bv)
    ell = (A - A_mean) ** 2 + (Bv - B_mean) ** 2
    loss = jnp.mean(ell)

    return {
        "A": A,
        "B": Bv,
        "A_mean": A_mean,
        "B_mean": B_mean,
        "ell": ell,
        "loss": loss,
        "dlogp_dt": dlogp_dt,
        "dphi_dt": dphi_dt,
        "e_real": e_real,
        "e_imag": e_imag,
    }


def test_time_derivatives_autodiff_match_dummy_closed_form():
    wf = DummyWavefunction(alpha=0.2, beta=1.7, gamma=-0.4, delta=0.9)
    configs = jnp.array(
        [
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=jnp.int32,
    )

    dlogp_dt, dphi_dt = time_derivatives_autodiff(wf, configs, t=jnp.float32(0.35))

    assert dlogp_dt.shape == (configs.shape[0],)
    assert dphi_dt.shape == (configs.shape[0],)
    assert jnp.allclose(dlogp_dt, wf.beta, atol=1e-6)
    assert jnp.allclose(dphi_dt, wf.delta, atol=1e-6)


def test_tdvp_residual_components_match_manual_formula():
    ham = TransverseIsingHamiltonian(J=1.3, h=0.8, N=4)
    wf = DummyWavefunction(alpha=0.4, beta=0.6, gamma=-0.25, delta=1.1)
    t = jnp.float32(0.2)

    configs = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=jnp.int32,
    )

    A, Bv, extras = tdvp_residual_components(ham, wf, configs, t)
    expected = _expected_from_manual_formula(ham, wf, configs)

    assert A.shape == (configs.shape[0],)
    assert Bv.shape == (configs.shape[0],)
    assert jnp.allclose(A, expected["A"], atol=1e-6)
    assert jnp.allclose(Bv, expected["B"], atol=1e-6)

    assert jnp.allclose(extras["dlogp_dt"], expected["dlogp_dt"], atol=1e-6)
    assert jnp.allclose(extras["dphi_dt"], expected["dphi_dt"], atol=1e-6)
    assert jnp.allclose(extras["e_real"], expected["e_real"], atol=1e-6)
    assert jnp.allclose(extras["e_imag"], expected["e_imag"], atol=1e-6)


def test_tdvp_residual_loss_matches_manual_centered_variance():
    ham = TransverseIsingHamiltonian(J=0.9, h=1.1, N=3)
    wf = DummyWavefunction(alpha=-0.3, beta=0.2, gamma=0.45, delta=-0.7)
    t = jnp.float32(0.8)

    configs = jnp.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=jnp.int32,
    )

    loss, diag = tdvp_residual_loss(ham, wf, configs, t, return_diagnostics=True)
    expected = _expected_from_manual_formula(ham, wf, configs)

    assert jnp.allclose(loss, expected["loss"], atol=1e-6)
    assert jnp.allclose(diag.A, expected["A"], atol=1e-6)
    assert jnp.allclose(diag.B, expected["B"], atol=1e-6)
    assert jnp.allclose(diag.A_mean, expected["A_mean"], atol=1e-6)
    assert jnp.allclose(diag.B_mean, expected["B_mean"], atol=1e-6)
    assert jnp.allclose(diag.ell, expected["ell"], atol=1e-6)
    assert jnp.allclose(diag.dlogp_dt, expected["dlogp_dt"], atol=1e-6)
    assert jnp.allclose(diag.dphi_dt, expected["dphi_dt"], atol=1e-6)
    assert jnp.allclose(diag.e_real, expected["e_real"], atol=1e-6)
    assert jnp.allclose(diag.e_imag, expected["e_imag"], atol=1e-6)


def test_weighted_tdvp_residual_loss_matches_expanded_duplicates():
    ham = TransverseIsingHamiltonian(J=0.9, h=1.1, N=3)
    wf = DummyWavefunction(alpha=-0.3, beta=0.2, gamma=0.45, delta=-0.7)
    t = jnp.float32(0.8)

    a = jnp.array([0, 1, 0], dtype=jnp.int32)
    b = jnp.array([1, 1, 0], dtype=jnp.int32)
    c = jnp.array([0, 0, 1], dtype=jnp.int32)
    dense_configs = jnp.stack([a, b, a, c, b, a], axis=0)
    unique_configs = jnp.stack(
        [a, b, c, jnp.array([1, 1, 1], dtype=jnp.int32), a, b],
        axis=0,
    )
    counts = jnp.array([3, 2, 1, 0, 0, 0], dtype=jnp.float32)

    dense_loss, dense_diag = tdvp_residual_loss(
        ham, wf, dense_configs, t, return_diagnostics=True
    )
    weighted_loss, weighted_diag = tdvp_residual_loss(
        ham,
        wf,
        unique_configs,
        t,
        return_diagnostics=True,
        sample_weights=counts,
    )

    assert jnp.allclose(weighted_loss, dense_loss, atol=1e-6)
    assert jnp.allclose(weighted_diag.A_mean, dense_diag.A_mean, atol=1e-6)
    assert jnp.allclose(weighted_diag.B_mean, dense_diag.B_mean, atol=1e-6)


def test_tdvp_residual_loss_without_diagnostics_returns_scalar():
    ham = TransverseIsingHamiltonian(J=0.0, h=0.5, N=4)
    wf = DummyWavefunction(alpha=0.1, beta=0.2, gamma=0.3, delta=0.4)
    configs = jnp.array(
        [
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )

    loss = tdvp_residual_loss(
        ham, wf, configs, t=jnp.float32(0.0), return_diagnostics=False
    )
    assert jnp.asarray(loss).ndim == 0
    assert jnp.isfinite(loss)


def test_tdvp_residual_loss_zero_when_batch_has_identical_samples():
    # With identical samples, A and B are constant across batch, so centered variance is zero.
    ham = TransverseIsingHamiltonian(J=1.2, h=0.7, N=5)
    wf = DummyWavefunction(alpha=0.25, beta=-0.5, gamma=-0.35, delta=0.9)

    single = jnp.array([0, 1, 1, 0, 1], dtype=jnp.int32)
    configs = jnp.stack([single, single, single, single], axis=0)

    loss, diag = tdvp_residual_loss(ham, wf, configs, t=jnp.float32(0.3))

    assert jnp.allclose(diag.A, diag.A[0], atol=1e-7)
    assert jnp.allclose(diag.B, diag.B[0], atol=1e-7)
    assert jnp.allclose(diag.ell, 0.0, atol=1e-7)
    assert jnp.allclose(loss, 0.0, atol=1e-7)


def test_tdvp_residual_components_validates_batch_shape_and_length():
    ham = TransverseIsingHamiltonian(J=1.0, h=1.0, N=4)
    wf = DummyWavefunction()

    # wrong rank
    bad_rank = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    try:
        _ = tdvp_residual_components(ham, wf, bad_rank, t=jnp.float32(0.0))
        assert False, "Expected ValueError for non-2D configurations."
    except ValueError as e:
        assert "Expected 2D configurations" in str(e)

    # wrong N dimension
    bad_len = jnp.array([[0, 1, 0], [1, 0, 1]], dtype=jnp.int32)
    try:
        _ = tdvp_residual_components(ham, wf, bad_len, t=jnp.float32(0.0))
        assert False, "Expected ValueError for wrong configuration length."
    except ValueError as e:
        assert "Expected configuration length" in str(e)

    # non-binary
    bad_binary = jnp.array([[0, 1, 2, 0]], dtype=jnp.int32)
    try:
        _ = tdvp_residual_components(ham, wf, bad_binary, t=jnp.float32(0.0))
        assert False, "Expected ValueError for non-binary configurations."
    except ValueError as e:
        assert "binary bits in {0,1}" in str(e)
