import flax.nnx as nnx
import jax
import jax.numpy as jnp

from src.galerkin import (
    GalerkinBasisMixtureWavefunction,
    estimate_galerkin_matrices,
    sample_galerkin_basis_mixture,
)
from src.hamiltonian import TransverseIsingHamiltonian
from src.observables import enumerate_binary_configurations
from src.wavefunction import NeuralGalerkinNQS


def _make_galerkin_wf():
    return NeuralGalerkinNQS(
        N=3,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        num_basis=2,
        num_modes=2,
        rngs=nnx.Rngs(0),
    )


def test_galerkin_basis_mixture_wavefunction_is_finite():
    wf = _make_galerkin_wf()
    sampler_wf = GalerkinBasisMixtureWavefunction(wf)
    configs = enumerate_binary_configurations(3)

    logp, phi = sampler_wf(configs, jnp.float32(0.0))

    assert logp.shape == (8,)
    assert phi.shape == (8,)
    assert jnp.all(jnp.isfinite(logp))
    assert jnp.allclose(phi, 0.0)


def test_estimate_galerkin_matrices_returns_hermitian_shapes():
    wf = _make_galerkin_wf()
    ham = TransverseIsingHamiltonian(J=-1.0, h=0.5, N=3)
    samples = enumerate_binary_configurations(3)

    projection = estimate_galerkin_matrices(ham, wf, samples, regularization=1e-5)

    assert projection.S.shape == (3, 3)
    assert projection.H.shape == (3, 3)
    assert projection.generator.shape == (3, 3)
    assert jnp.all(jnp.isfinite(jnp.real(projection.S)))
    assert jnp.all(jnp.isfinite(jnp.imag(projection.S)))
    assert jnp.all(jnp.isfinite(jnp.real(projection.H)))
    assert jnp.all(jnp.isfinite(jnp.imag(projection.H)))
    assert jnp.allclose(projection.S, jnp.conj(projection.S.T), atol=1e-5)
    assert jnp.allclose(projection.H, jnp.conj(projection.H.T), atol=1e-5)


def test_sample_galerkin_basis_mixture_smoke():
    wf = _make_galerkin_wf()
    initial = jnp.zeros((2, 3), dtype=jnp.int32)

    samples, stats, final = sample_galerkin_basis_mixture(
        wf,
        initial,
        n_samples=2,
        burn_in=1,
        thinning=1,
        key=jax.random.PRNGKey(0),
    )

    assert samples.shape == (2, 2, 3)
    assert final.shape == (2, 3)
    assert stats["acceptance_rate"].shape == (2,)
