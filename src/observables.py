from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.sampler import metropolis_hastings_sample
from src.wavefunction import Wavefunction


@jax.tree_util.register_pytree_node_class
@dataclass
class ObservableEstimates:
    """Monte Carlo or exact expectation values of simple spin observables."""

    z_total: jnp.ndarray
    z_mean: jnp.ndarray
    x_total_real: jnp.ndarray
    x_total_imag: jnp.ndarray
    x_mean_real: jnp.ndarray
    x_mean_imag: jnp.ndarray
    # Per-site observables
    z_sites: jnp.ndarray  # shape: (N,)
    x_sites_real: jnp.ndarray  # shape: (N,)
    x_sites_imag: jnp.ndarray  # shape: (N,)
    n_samples: int

    def tree_flatten(self):
        children = (
            self.z_total,
            self.z_mean,
            self.x_total_real,
            self.x_total_imag,
            self.x_mean_real,
            self.x_mean_imag,
            self.z_sites,
            self.x_sites_real,
            self.x_sites_imag,
        )
        aux_data = self.n_samples
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, n_samples=aux_data)


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
    if not isinstance(configs, jax.core.Tracer):
        if not jnp.all((configs == 0) | (configs == 1)):
            raise ValueError("Configurations must be binary bits in {0,1}.")
    return configs


def _bit_to_sz(configurations: jnp.ndarray) -> jnp.ndarray:
    """Map bits {0,1} to sigma^z eigenvalues {+1,-1}."""
    configs = jnp.asarray(configurations).astype(jnp.int32)
    return 1 - 2 * configs


def _local_x_sites_single(
    wf: Wavefunction,
    configuration: jnp.ndarray,
    t,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Local estimators for sigma_i^x for each site i on one configuration."""
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

    real_sites, imag_sites = jax.vmap(single_site)(jnp.arange(config.shape[0]))
    return real_sites, imag_sites


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
      - Per-site values are returned in `z_sites`, `x_sites_real`, etc.
    """
    configs = _validate_batch_configs(configurations, n_sites)
    
    # Per-site, per-sample: (B, N)
    z_per_site_per_sample = _bit_to_sz(configs).astype(jnp.float32)
    
    # (B, N), (B, N)
    x_sites_real_per_sample, x_sites_imag_per_sample = jax.vmap(
        lambda cfg: _local_x_sites_single(wf, cfg, t)
    )(configs)

    # Average over batch dimension (B)
    z_sites = jnp.mean(z_per_site_per_sample, axis=0)
    x_sites_real = jnp.mean(x_sites_real_per_sample, axis=0)
    x_sites_imag = jnp.mean(x_sites_imag_per_sample, axis=0)

    # Total and mean (site-averaged)
    z_total = jnp.sum(z_sites)
    x_total_real = jnp.sum(x_sites_real)
    x_total_imag = jnp.sum(x_sites_imag)
    
    n_sites_f = float(n_sites)

    return ObservableEstimates(
        z_total=z_total,
        z_mean=z_total / n_sites_f,
        x_total_real=x_total_real,
        x_total_imag=x_total_imag,
        x_mean_real=x_total_real / n_sites_f,
        x_mean_imag=x_total_imag / n_sites_f,
        z_sites=z_sites,
        x_sites_real=x_sites_real,
        x_sites_imag=x_sites_imag,
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
    logp, phi = wf(configs, t)
    logp = jnp.asarray(logp, dtype=jnp.float32)
    phi = jnp.asarray(phi, dtype=jnp.float32)

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

    # Per-site Z: sum over all states weighted by prob
    z_per_site_all_states = _bit_to_sz(configs).astype(jnp.float32)  # (2^N, N)
    z_sites = jnp.sum(probabilities[:, None] * z_per_site_all_states, axis=0)
    z_total = jnp.sum(z_sites)

    # Per-site X
    basis_indices = jnp.arange(2**n_sites, dtype=jnp.int32)
    x_sites_complex = []
    for site in range(n_sites):
        flipped_indices = basis_indices ^ (1 << (n_sites - 1 - site))
        x_i = jnp.vdot(psi, psi[flipped_indices])
        x_sites_complex.append(x_i)
    
    x_sites_complex = jnp.stack(x_sites_complex)
    x_sites_real = jnp.real(x_sites_complex)
    x_sites_imag = jnp.imag(x_sites_complex)
    
    x_total_real = jnp.sum(x_sites_real)
    x_total_imag = jnp.sum(x_sites_imag)

    n_sites_f = float(n_sites)
    return ObservableEstimates(
        z_total=z_total,
        z_mean=z_total / n_sites_f,
        x_total_real=x_total_real,
        x_total_imag=x_total_imag,
        x_mean_real=x_total_real / n_sites_f,
        x_mean_imag=x_total_imag / n_sites_f,
        z_sites=z_sites,
        x_sites_real=x_sites_real,
        x_sites_imag=x_sites_imag,
        n_samples=int(2**n_sites),
    )
