from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from src.sampler import metropolis_hastings_sample
from src.wavefunction import Wavefunction


@dataclass
class ObservableEstimates:
    """Monte Carlo or exact expectation values of simple spin observables."""

    z_total: jnp.ndarray
    z_mean: jnp.ndarray
    x_total_real: jnp.ndarray
    x_total_imag: jnp.ndarray
    x_mean_real: jnp.ndarray
    x_mean_imag: jnp.ndarray
    n_samples: int


def _validate_batch_configs(configurations: jnp.ndarray, n_sites: int) -> jnp.ndarray:
    """Validate (B, N) binary configurations."""
    configs = jnp.asarray(configurations).astype(jnp.int32)
    if configs.ndim != 2:
        raise ValueError(
            f"Expected 2D configurations with shape (B, N), got {configs.shape}"
        )
    if configs.shape[1] != n_sites:
        raise ValueError(
            f"Expected configuration length N={n_sites}, got {configs.shape[1]}"
        )
    if not jnp.all((configs == 0) | (configs == 1)):
        raise ValueError("Configurations must be binary bits in {0,1}.")
    return configs


def _bit_to_sz(configurations: jnp.ndarray) -> jnp.ndarray:
    """Map bits {0,1} to sigma^z eigenvalues {+1,-1}."""
    configs = jnp.asarray(configurations).astype(jnp.int32)
    return 1 - 2 * configs


def _local_x_total_single(
    wf: Wavefunction,
    configuration: jnp.ndarray,
    t,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Local estimator for total X = sum_i sigma_i^x on one configuration."""
    config = jnp.asarray(configuration).astype(jnp.int32)
    if config.ndim != 1:
        raise ValueError(f"Expected 1D configuration, got {config.shape}")

    logp, phi = wf(config, t)
    logp = jnp.asarray(logp).squeeze()
    phi = jnp.asarray(phi).squeeze()

    def single_site(site_ind: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        flipped = config.at[site_ind].set(1 - config[site_ind])
        new_logp, new_phi = wf(flipped, t)
        new_logp = jnp.asarray(new_logp).squeeze()
        new_phi = jnp.asarray(new_phi).squeeze()

        delta_logp = new_logp - logp
        delta_phi = new_phi - phi
        magnitude = jnp.exp(jnp.clip(0.5 * delta_logp, min=-40.0, max=40.0))
        return magnitude * jnp.cos(delta_phi), magnitude * jnp.sin(delta_phi)

    real_terms, imag_terms = jax.vmap(single_site)(jnp.arange(config.shape[0]))
    return jnp.sum(real_terms), jnp.sum(imag_terms)


def measure_observables(
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
    *,
    n_sites: int,
) -> ObservableEstimates:
    """Estimate site-averaged and total <Z> / <X> from sampled configurations.

    Convention:
      - Z uses sigma^z eigenvalues s_i = 1 - 2 * bit_i.
      - Returned `z_mean` and `x_mean_*` are site averages, i.e. total / N.
    """
    configs = _validate_batch_configs(configurations, n_sites)
    z_total_per_sample = jnp.sum(_bit_to_sz(configs).astype(jnp.float32), axis=1)
    x_total_real_per_sample, x_total_imag_per_sample = jax.vmap(
        lambda cfg: _local_x_total_single(wf, cfg, t)
    )(configs)

    z_total = jnp.mean(z_total_per_sample)
    x_total_real = jnp.mean(x_total_real_per_sample)
    x_total_imag = jnp.mean(x_total_imag_per_sample)
    n_sites_f = float(n_sites)

    return ObservableEstimates(
        z_total=z_total,
        z_mean=z_total / n_sites_f,
        x_total_real=x_total_real,
        x_total_imag=x_total_imag,
        x_mean_real=x_total_real / n_sites_f,
        x_mean_imag=x_total_imag / n_sites_f,
        n_samples=int(configs.shape[0]),
    )


def sample_and_measure_observables(
    wf: Wavefunction,
    t,
    *,
    n_sites: int,
    initial_configurations: jnp.ndarray,
    n_samples: int,
    burn_in: int = 100,
    thinning: int = 1,
    key: Optional[jax.Array] = None,
    return_samples: bool = False,
):
    """Sample from the wavefunction and estimate <Z> and <X>."""
    samples, sampler_stats = metropolis_hastings_sample(
        wf=wf,
        initial_configurations=initial_configurations,
        t=t,
        n_sites=n_sites,
        n_samples=n_samples,
        burn_in=burn_in,
        thinning=thinning,
        key=key,
        return_stats=True,
    )
    flat_samples = samples.reshape((-1, n_sites))
    estimates = measure_observables(wf, flat_samples, t, n_sites=n_sites)
    final_configurations = samples[:, -1, :]

    if return_samples:
        return estimates, sampler_stats, final_configurations, samples
    return estimates, sampler_stats, final_configurations


def enumerate_binary_configurations(n_sites: int) -> jnp.ndarray:
    """Enumerate all bitstrings of length n_sites in lexicographic order."""
    if n_sites <= 0:
        raise ValueError(f"n_sites must be > 0, got {n_sites}")
    states = jnp.arange(2**n_sites, dtype=jnp.int32)
    bit_positions = jnp.arange(n_sites - 1, -1, -1, dtype=jnp.int32)
    return ((states[:, None] >> bit_positions[None, :]) & 1).astype(jnp.int32)


def normalized_statevector(wf: Wavefunction, n_sites: int, t) -> jnp.ndarray:
    """Enumerate and normalize the wavefunction amplitudes for small systems."""
    configs = enumerate_binary_configurations(n_sites)
    logp = jnp.asarray(wf.log_prob(configs, t), dtype=jnp.float32)
    phi = jnp.asarray(wf.phase(configs, t), dtype=jnp.float32)

    if logp.ndim == 2 and logp.shape[-1] == 1:
        logp = jnp.squeeze(logp, axis=-1)
    if phi.ndim == 2 and phi.shape[-1] == 1:
        phi = jnp.squeeze(phi, axis=-1)

    psi = jnp.exp(0.5 * logp + 1j * phi)
    norm = jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
    if float(norm) == 0.0:
        raise ValueError("Wavefunction amplitudes have zero norm.")
    return psi / norm


def exact_observables_from_wf(
    wf: Wavefunction,
    n_sites: int,
    t,
) -> ObservableEstimates:
    """Compute exact <Z> and <X> by exhaustive enumeration for small systems."""
    psi = normalized_statevector(wf, n_sites, t)
    configs = enumerate_binary_configurations(n_sites)
    probabilities = jnp.abs(psi) ** 2

    z_total_values = jnp.sum(_bit_to_sz(configs).astype(jnp.float32), axis=1)
    z_total = jnp.sum(probabilities * z_total_values)

    basis_indices = jnp.arange(2**n_sites, dtype=jnp.int32)
    x_total = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex64)
    for site in range(n_sites):
        flipped_indices = basis_indices ^ (1 << (n_sites - 1 - site))
        x_total = x_total + jnp.vdot(psi, psi[flipped_indices])

    n_sites_f = float(n_sites)
    return ObservableEstimates(
        z_total=jnp.real(z_total),
        z_mean=jnp.real(z_total) / n_sites_f,
        x_total_real=jnp.real(x_total),
        x_total_imag=jnp.imag(x_total),
        x_mean_real=jnp.real(x_total) / n_sites_f,
        x_mean_imag=jnp.imag(x_total) / n_sites_f,
        n_samples=int(2**n_sites),
    )
