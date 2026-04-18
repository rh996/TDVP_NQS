import jax.numpy as jnp

from src.hamiltonian import (
    TransverseIsingHamiltonian,
    local_energy,
    local_energy_batch,
    zz_energy_open,
)


class DummyWavefunction:
    """Simple deterministic wavefunction for testing.

    logp(sigma, t) = alpha * sum(sigma) + beta * t
    phi(sigma, t)  = gamma * sum(sigma) + delta * t
    """

    def __init__(self, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def __call__(self, configuration, t):
        configuration = jnp.asarray(configuration)
        ssum = jnp.sum(configuration).astype(jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        logp = self.alpha * ssum + self.beta * t
        phi = self.gamma * ssum + self.delta * t
        return logp, phi


def test_zz_energy_open_obc_and_bit_to_sz_mapping():
    # Convention check:
    # bit -> s^z via s = 1 - 2*bit
    # bits [0, 1, 0] -> s [ +1, -1, +1 ]
    # OBC bonds: (0,1), (1,2): products = -1, -1 => sum = -2
    ham = TransverseIsingHamiltonian(J=1.5, h=0.0, N=3)
    config = jnp.array([0, 1, 0], dtype=jnp.int32)

    ezz = zz_energy_open(ham, config)
    expected = -2.0 * ham.J

    assert jnp.allclose(ezz, expected, atol=1e-6), f"{ezz=} {expected=}"


def test_local_energy_real_imag_constant_wavefunction():
    # If logp and phi are constant over configurations:
    # Psi(sigma^i)/Psi(sigma) = 1 for all i
    # => transverse real = h * N, transverse imag = 0
    # total real = JZZ + hN, imag = 0
    ham = TransverseIsingHamiltonian(J=2.0, h=0.7, N=4)
    wf = DummyWavefunction(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)

    config = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    # s = [ +1, +1, -1, -1 ]
    # OBC products: (+1)(+1)=+1, (+1)(-1)=-1, (-1)(-1)=+1 => sum=+1
    # JZZ = 2.0 * 1 = 2.0
    expected_real = 2.0 + ham.h * ham.N  # 2.0 + 2.8 = 4.8
    expected_imag = 0.0

    e_real, e_imag = local_energy(ham, wf, config, t=0.3)

    assert jnp.allclose(e_real, expected_real, atol=1e-6), f"{e_real=} {expected_real=}"
    assert jnp.allclose(e_imag, expected_imag, atol=1e-6), f"{e_imag=} {expected_imag=}"


def test_local_energy_transverse_parts_from_delta_logp_and_delta_phi():
    # Choose logp = alpha * sum(bits), phi = gamma * sum(bits).
    # Flipping one bit changes sum(bits) by:
    #   +1 if bit=0, -1 if bit=1
    # Thus per-site ratio magnitude:
    #   exp(0.5 * alpha * delta_sum)
    # phase shift:
    #   gamma * delta_sum
    #
    # For config [0,1,0], delta_sum list = [+1, -1, +1]
    ham = TransverseIsingHamiltonian(J=0.0, h=1.2, N=3)
    alpha = 0.4
    gamma = 0.3
    wf = DummyWavefunction(alpha=alpha, gamma=gamma)

    config = jnp.array([0, 1, 0], dtype=jnp.int32)
    deltas = jnp.array([+1.0, -1.0, +1.0], dtype=jnp.float32)

    mags = jnp.exp(0.5 * alpha * deltas)
    expected_real = ham.h * jnp.sum(mags * jnp.cos(gamma * deltas))
    expected_imag = ham.h * jnp.sum(mags * jnp.sin(gamma * deltas))

    e_real, e_imag = local_energy(ham, wf, config, t=0.0)

    assert jnp.allclose(e_real, expected_real, atol=1e-6), f"{e_real=} {expected_real=}"
    assert jnp.allclose(e_imag, expected_imag, atol=1e-6), f"{e_imag=} {expected_imag=}"


def test_local_energy_uses_open_boundary_not_periodic():
    # N=2 already trivial; use N=3 to distinguish OBC from PBC clearly.
    # config [0,0,1] -> s [ +1, +1, -1 ]
    # OBC bonds: (0,1)=+1, (1,2)=-1 => sum=0
    # PBC would include (2,0)=-1 => sum=-1 (different)
    ham = TransverseIsingHamiltonian(J=3.0, h=0.0, N=3)
    wf = DummyWavefunction()

    config = jnp.array([0, 0, 1], dtype=jnp.int32)
    e_real, e_imag = local_energy(ham, wf, config, t=0.0)

    assert jnp.allclose(e_real, 0.0, atol=1e-6), f"{e_real=}"
    assert jnp.allclose(e_imag, 0.0, atol=1e-6), f"{e_imag=}"


def test_local_energy_input_validation_shape_and_length():
    ham = TransverseIsingHamiltonian(J=1.0, h=1.0, N=4)
    wf = DummyWavefunction()

    # wrong length
    bad_len = jnp.array([0, 1, 0], dtype=jnp.int32)
    try:
        _ = local_energy(ham, wf, bad_len, t=0.0)
        assert False, "Expected ValueError for wrong configuration length."
    except ValueError as e:
        assert "Expected length" in str(e)

    # wrong rank
    bad_rank = jnp.array([[0, 1, 0, 1]], dtype=jnp.int32)
    try:
        _ = local_energy(ham, wf, bad_rank, t=0.0)
        assert False, "Expected ValueError for non-1D configuration."
    except ValueError as e:
        assert "Expected 1D configuration" in str(e)


def test_local_energy_batch_matches_per_sample_results():
    ham = TransverseIsingHamiltonian(J=1.1, h=0.4, N=4)
    wf = DummyWavefunction(alpha=0.2, beta=0.1, gamma=-0.3, delta=0.05)
    t = 0.7

    batch = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=jnp.int32,
    )

    batch_real, batch_imag = local_energy_batch(ham, wf, batch, t=t)

    per_sample = [local_energy(ham, wf, cfg, t=t) for cfg in batch]
    expected_real = jnp.array([x[0] for x in per_sample])
    expected_imag = jnp.array([x[1] for x in per_sample])

    assert batch_real.shape == (batch.shape[0],)
    assert batch_imag.shape == (batch.shape[0],)
    assert jnp.allclose(batch_real, expected_real, atol=1e-6), (
        f"{batch_real=} {expected_real=}"
    )
    assert jnp.allclose(batch_imag, expected_imag, atol=1e-6), (
        f"{batch_imag=} {expected_imag=}"
    )


def test_local_energy_batch_singleton_batch_consistency():
    ham = TransverseIsingHamiltonian(J=0.8, h=1.3, N=3)
    wf = DummyWavefunction(alpha=-0.1, gamma=0.25)
    t = 0.2

    config = jnp.array([0, 1, 1], dtype=jnp.int32)
    batch = config[jnp.newaxis, :]

    e_real_single, e_imag_single = local_energy(ham, wf, config, t=t)
    e_real_batch, e_imag_batch = local_energy_batch(ham, wf, batch, t=t)

    assert e_real_batch.shape == (1,)
    assert e_imag_batch.shape == (1,)
    assert jnp.allclose(e_real_batch[0], e_real_single, atol=1e-6)
    assert jnp.allclose(e_imag_batch[0], e_imag_single, atol=1e-6)


def test_local_energy_batch_input_validation_shape_and_length():
    ham = TransverseIsingHamiltonian(J=1.0, h=0.5, N=4)
    wf = DummyWavefunction()

    # wrong rank: expecting (B, N)
    bad_rank = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    try:
        _ = local_energy_batch(ham, wf, bad_rank, t=0.0)
        assert False, "Expected ValueError for non-2D batched configurations."
    except ValueError as e:
        assert "Expected 2D batched configurations" in str(e)

    # wrong N dimension
    bad_length = jnp.array([[0, 1, 0], [1, 0, 1]], dtype=jnp.int32)
    try:
        _ = local_energy_batch(ham, wf, bad_length, t=0.0)
        assert False, "Expected ValueError for wrong configuration length in batch."
    except ValueError as e:
        assert "Expected second dimension" in str(e)
