from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from src.hamiltonian import (
    LongRangeTransverseIsingHamiltonian,
    TransverseIsingHamiltonian,
    zz_energy_lrtfim,
    zz_energy_open,
)
from src.sampler import metropolis_hastings_sample
from src.wavefunction import NeuralGalerkinNQS, Wavefunction


@dataclass
class GalerkinProjectionResult:
    """Estimated Neural Galerkin projected operators."""

    S: jnp.ndarray
    H: jnp.ndarray
    generator: jnp.ndarray
    regularization: float


class GalerkinBasisMixtureWavefunction(Wavefunction):
    """Sampler target p(sigma) proportional to sum_i |psi_i(sigma)|^2."""

    def __init__(self, galerkin_wf: NeuralGalerkinNQS, eps: float = 1e-12):
        self.galerkin_wf = galerkin_wf
        self.model = galerkin_wf.model
        self.N = galerkin_wf.N
        self.eps = eps

    def __call__(self, configuration, t):
        basis_values = self.galerkin_wf.model.basis_values(configuration)
        mixture_density = jnp.sum(jnp.abs(basis_values) ** 2, axis=0)
        logp = jnp.log(jnp.maximum(mixture_density, self.eps))
        phi = jnp.zeros_like(logp)
        return logp, phi


def _as_flat_samples(samples: jnp.ndarray, n_sites: int) -> jnp.ndarray:
    samples = jnp.asarray(samples).astype(jnp.int32)
    if samples.ndim == 2:
        flat_samples = samples
    elif samples.ndim == 3:
        flat_samples = samples.reshape((-1, n_sites))
    else:
        raise ValueError(
            f"Expected samples with shape (B,N) or (C,S,N), got {samples.shape}"
        )

    if flat_samples.shape[1] != n_sites:
        raise ValueError(
            f"Expected sample site dimension N={n_sites}, got {flat_samples.shape[1]}"
        )
    return flat_samples


def _diagonal_energies(ham, configurations: jnp.ndarray) -> jnp.ndarray:
    if isinstance(ham, TransverseIsingHamiltonian):
        return jax.vmap(lambda cfg: zz_energy_open(ham, cfg))(configurations)
    if isinstance(ham, LongRangeTransverseIsingHamiltonian):
        return jax.vmap(lambda cfg: zz_energy_lrtfim(ham, cfg))(configurations)
    raise NotImplementedError(f"Unsupported Hamiltonian type: {type(ham)}")


def _flipped_configurations(configurations: jnp.ndarray) -> jnp.ndarray:
    n_sites = configurations.shape[1]
    sites = jnp.arange(n_sites)
    rows = jnp.arange(configurations.shape[0])

    def flip_site(site):
        return configurations.at[rows, site].set(1 - configurations[:, site])

    return jax.vmap(flip_site)(sites)


def estimate_galerkin_matrices(
    ham,
    galerkin_wf: NeuralGalerkinNQS,
    samples: jnp.ndarray,
    *,
    regularization: float = 1e-6,
    density_eps: float = 1e-12,
    hermitize: bool = True,
) -> GalerkinProjectionResult:
    """Estimate S and H in the trained Neural Galerkin basis.

    Samples should be drawn from p(sigma) proportional to sum_i |psi_i(sigma)|^2.
    Because that sampler distribution is normalized internally, both S and H are
    estimated up to the same unknown constant scale; the scale cancels in S^{-1}H.
    """
    if regularization < 0:
        raise ValueError(f"regularization must be >= 0, got {regularization}")

    configurations = _as_flat_samples(samples, ham.N)
    basis_values = galerkin_wf.model.basis_values(configurations)
    density = jnp.sum(jnp.abs(basis_values) ** 2, axis=0)
    inv_density = 1.0 / jnp.maximum(density, density_eps)

    diag = _diagonal_energies(ham, configurations).astype(jnp.complex64)
    flipped = _flipped_configurations(configurations)
    flat_flipped = flipped.reshape((-1, ham.N))
    flipped_values = galerkin_wf.model.basis_values(flat_flipped)
    flipped_values = flipped_values.reshape(
        (basis_values.shape[0], ham.N, configurations.shape[0])
    )

    h_psi = diag[jnp.newaxis, :] * basis_values
    h_psi = h_psi + ham.h * jnp.sum(flipped_values, axis=1)

    S = jnp.einsum(
        "ib,jb,b->ij",
        jnp.conj(basis_values),
        basis_values,
        inv_density,
    ) / configurations.shape[0]
    H = jnp.einsum(
        "ib,jb,b->ij",
        jnp.conj(basis_values),
        h_psi,
        inv_density,
    ) / configurations.shape[0]

    if hermitize:
        S = 0.5 * (S + jnp.conj(S.T))
        H = 0.5 * (H + jnp.conj(H.T))

    eye = jnp.eye(S.shape[0], dtype=S.dtype)
    generator = jnp.linalg.solve(S + regularization * eye, H)
    return GalerkinProjectionResult(
        S=S,
        H=H,
        generator=generator,
        regularization=float(regularization),
    )


def sample_galerkin_basis_mixture(
    galerkin_wf: NeuralGalerkinNQS,
    initial_configurations: jnp.ndarray,
    *,
    n_samples: int,
    burn_in: int,
    thinning: int,
    key: jax.Array,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Sample the Galerkin basis-mixture distribution used for projection."""
    sampler_wf = GalerkinBasisMixtureWavefunction(galerkin_wf)
    samples, stats = metropolis_hastings_sample(
        wf=sampler_wf,
        initial_configurations=initial_configurations,
        t=0.0,
        n_sites=galerkin_wf.N,
        n_samples=n_samples,
        burn_in=burn_in,
        thinning=thinning,
        key=key,
        return_stats=True,
    )
    final_configurations = samples[:, -1, :]
    return samples, stats, final_configurations
