from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.wavefunction import Wavefunction


@dataclass
class TransverseIsingHamiltonian:
    """Transverse-field Ising Hamiltonian with explicit JZZ + hX convention.

    Convention:
      H = J * sum_{i=0}^{N-2} (sigma_i^z sigma_{i+1}^z) + h * sum_{i=0}^{N-1} sigma_i^x

    Spin representation:
      - Configurations are binary bits in {0, 1}.
      - Physical z-spin values are mapped as:
            s_i = 1 - 2 * bit_i
        so bit=0 -> +1, bit=1 -> -1.

    Boundary condition:
      - Open boundary condition (OBC): nearest-neighbor bonds (i, i+1), i=0..N-2.
    """

    J: float
    h: float
    N: int

    @property
    def n_sites(self) -> int:
        return self.N


def _bit_to_sz(configuration: jnp.ndarray) -> jnp.ndarray:
    """Map binary configuration {0,1} to sigma^z eigenvalues {+1,-1}."""
    configuration = configuration.astype(jnp.int32)
    return 1 - 2 * configuration


def zz_energy_open(
    ham: TransverseIsingHamiltonian, configuration: jnp.ndarray
) -> jnp.ndarray:
    """Compute diagonal JZZ energy with open boundaries for one configuration."""
    configuration = jnp.asarray(configuration).astype(jnp.int32)
    if configuration.ndim != 1:
        raise ValueError(f"Expected 1D configuration, got shape {configuration.shape}")
    if configuration.shape[0] != ham.N:
        raise ValueError(f"Expected length {ham.N}, got {configuration.shape[0]}")

    if ham.N <= 1:
        return jnp.asarray(0.0, dtype=jnp.float32)

    s = _bit_to_sz(configuration).astype(jnp.float32)
    # OBC bonds: (0,1), (1,2), ..., (N-2, N-1)
    return ham.J * jnp.sum(s[:-1] * s[1:])


def _transverse_term_single_site_parts(
    site_ind: int,
    logp: jnp.ndarray,
    phi: jnp.ndarray,
    wf: Wavefunction,
    configuration: jnp.ndarray,
    t,
    h: float,
):
    """Single-site hX contribution split into real/imag parts.

    h * Psi(sigma^i)/Psi(sigma) =
      h * exp(0.5 * Δlogp) * [cos(Δphi) + i sin(Δphi)]
    """
    flipped = configuration.at[site_ind].set(1 - configuration[site_ind])
    new_logp, new_phi = wf(flipped, t)

    new_logp = jnp.asarray(new_logp).squeeze()
    new_phi = jnp.asarray(new_phi).squeeze()

    delta_logp = new_logp - logp
    delta_phi = new_phi - phi

    mag = jnp.exp(jnp.clip(0.5 * delta_logp, min=-40.0, max=40.0))
    real_part = h * mag * jnp.cos(delta_phi)
    imag_part = h * mag * jnp.sin(delta_phi)
    return real_part, imag_part


def local_energy(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configuration: jnp.ndarray,
    t,
):
    """Return local energy as (real_part, imag_part), without complex arithmetic.

    E_loc(sigma, t) =
        J * sum_{i=0}^{N-2} s_i s_{i+1}
        + h * sum_{i=0}^{N-1} Psi(sigma^(i), t) / Psi(sigma, t)

    with:
      s_i = 1 - 2*bit_i, bit_i in {0,1}
      sigma^(i): configuration with bit i flipped
    """
    configuration = jnp.asarray(configuration).astype(jnp.int32)
    if configuration.ndim != 1:
        raise ValueError(f"Expected 1D configuration, got shape {configuration.shape}")
    if configuration.shape[0] != ham.N:
        raise ValueError(f"Expected length {ham.N}, got {configuration.shape[0]}")

    zz_real = zz_energy_open(ham, configuration)

    logp, phi = wf(configuration, t)
    logp = jnp.asarray(logp).squeeze()
    phi = jnp.asarray(phi).squeeze()

    sites = jnp.arange(ham.N)

    real_terms, imag_terms = jax.vmap(
        lambda i: _transverse_term_single_site_parts(
            i, logp, phi, wf, configuration, t, ham.h
        )
    )(sites)

    e_real = zz_real + jnp.sum(real_terms)
    e_imag = jnp.sum(imag_terms)
    return e_real, e_imag


def local_energy_batch(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
):
    """Vectorized local energy for a batch of configurations.

    Args:
      ham: Hamiltonian parameters.
      wf: Wavefunction returning (logp, phi) for a configuration.
      configurations: Array of shape (B, N) with bits in {0,1}.
      t: Time scalar (shared across the batch).

    Returns:
      (e_real, e_imag), each with shape (B,).
    """
    configurations = jnp.asarray(configurations).astype(jnp.int32)
    if configurations.ndim != 2:
        raise ValueError(
            f"Expected 2D batched configurations with shape (B, N), got {configurations.shape}"
        )
    if configurations.shape[1] != ham.N:
        raise ValueError(
            f"Expected second dimension N={ham.N}, got {configurations.shape[1]}"
        )

    e_real, e_imag = jax.vmap(lambda config: local_energy(ham, wf, config, t))(
        configurations
    )
    return e_real, e_imag
