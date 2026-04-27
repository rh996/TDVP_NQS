from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from src.wavefunction import Wavefunction


@dataclass
class SamplerStats:
    """Diagnostics returned by the Metropolis sampler."""

    acceptance_rate: jnp.ndarray  # shape: (n_chains,)
    proposed_moves: int
    accepted_moves: jnp.ndarray  # shape: (n_chains,)


def _as_batched_configs(configurations: jnp.ndarray, n_sites: int) -> jnp.ndarray:
    """Normalize configurations to shape (n_chains, n_sites)."""
    configs = jnp.asarray(configurations).astype(jnp.int32)

    if configs.ndim == 1:
        if configs.shape[0] != n_sites:
            raise ValueError(
                f"Expected 1D configuration length {n_sites}, got {configs.shape[0]}"
            )
        configs = configs[jnp.newaxis, :]
    elif configs.ndim == 2:
        if configs.shape[1] != n_sites:
            raise ValueError(
                f"Expected 2D configurations second dim {n_sites}, got {configs.shape[1]}"
            )
    else:
        raise ValueError(
            f"Expected configurations of rank 1 or 2, got shape {configs.shape}"
        )

    if not isinstance(configs, jax.core.Tracer):
        if not jnp.all((configs == 0) | (configs == 1)):
            raise ValueError("Configurations must be binary bits in {0,1}.")

    return configs


def _log_prob_batch(wf: Wavefunction, configs: jnp.ndarray, t) -> jnp.ndarray:
    """Evaluate log-probability for a batch of configurations."""
    logp, _ = wf(configs, t)
    logp = jnp.asarray(logp)
    if logp.ndim == 0:
        return logp[jnp.newaxis]
    if logp.ndim == 1:
        return logp
    if logp.ndim == 2 and logp.shape[-1] == 1:
        return jnp.squeeze(logp, axis=-1)
    raise ValueError(f"Unexpected log_prob output shape: {logp.shape}")


def _mh_step(
    wf: Wavefunction,
    configs: jnp.ndarray,
    logp_current: jnp.ndarray,
    t,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One vectorized MH step over all chains."""
    n_chains, n_sites = configs.shape
    key_site, key_u = jax.random.split(key)

    # Propose one spin flip per chain.
    flip_sites = jax.random.randint(
        key_site, shape=(n_chains,), minval=0, maxval=n_sites
    )
    rows = jnp.arange(n_chains)
    proposed = configs.at[rows, flip_sites].set(1 - configs[rows, flip_sites])

    # Acceptance ratio for target p(sigma) ∝ exp(logp(sigma)).
    logp_proposed = _log_prob_batch(wf, proposed, t)
    log_alpha = logp_proposed - logp_current

    # Accept if log(u) < log_alpha.
    u = jax.random.uniform(key_u, shape=(n_chains,), minval=0.0, maxval=1.0)
    accept = jnp.log(u) < jnp.minimum(log_alpha, 0.0)

    new_configs = jnp.where(accept[:, None], proposed, configs)
    new_logp = jnp.where(accept, logp_proposed, logp_current)
    return new_configs, new_logp, accept


def metropolis_hastings_sample(
    wf: Wavefunction,
    initial_configurations: jnp.ndarray,
    t,
    *,
    n_sites: int,
    n_samples: int,
    burn_in: int = 100,
    thinning: int = 1,
    key: Optional[jax.Array] = None,
    return_stats: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Sample from p_theta(sigma,t) ∝ exp(logp_theta(sigma,t)) via MH.

    Args:
        wf: Wavefunction object exposing `log_prob(configurations, t)`.
        initial_configurations: shape (N,) or (C, N), bits in {0,1}.
        t: Time input passed to the wavefunction.
        n_sites: Number of spin sites (N).
        n_samples: Number of retained samples per chain after burn-in/thinning.
        burn_in: Number of initial MH steps to discard.
        thinning: Keep one sample every `thinning` MH steps.
        key: PRNG key. If None, a default deterministic key is used.
        return_stats: Whether to return diagnostics.

    Returns:
        samples: shape (C, n_samples, N), where C is number of chains.
        stats: diagnostics dictionary with acceptance metrics.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if thinning <= 0:
        raise ValueError(f"thinning must be > 0, got {thinning}")

    configs = _as_batched_configs(initial_configurations, n_sites)
    n_chains = configs.shape[0]
    total_steps = burn_in + n_samples * thinning

    if key is None:
        key = jax.random.PRNGKey(0)

    logp = _log_prob_batch(wf, configs, t)

    def scan_body(carry, _):
        k, c_configs, c_logp, c_acc = carry
        k, step_key = jax.random.split(k)
        new_configs, new_logp, accept = _mh_step(wf, c_configs, c_logp, t, step_key)
        return (k, new_configs, new_logp, c_acc + accept.astype(jnp.int32)), None

    carry = (key, configs, logp, jnp.zeros((n_chains,), dtype=jnp.int32))
    
    # 1. Burn-in steps
    carry, _ = jax.lax.scan(scan_body, carry, None, length=burn_in)
    
    # 2. Sampling steps
    def sample_body(carry, _):
        carry, _ = jax.lax.scan(scan_body, carry, None, length=thinning)
        # carry[1] is the new_configs
        return carry, carry[1]

    carry, samples = jax.lax.scan(sample_body, carry, None, length=n_samples)
    
    # Swap axes from (n_samples, n_chains, N) to (n_chains, n_samples, N)
    samples = jnp.swapaxes(samples, 0, 1)

    if not return_stats:
        return samples, {}

    accepted_count = carry[3]
    acceptance_rate = accepted_count.astype(jnp.float32) / float(total_steps)
    stats = {
        "acceptance_rate": acceptance_rate,
        "accepted_moves": accepted_count,
        "proposed_moves": jnp.asarray(total_steps, dtype=jnp.int32),
        "n_chains": jnp.asarray(n_chains, dtype=jnp.int32),
        "n_samples_per_chain": jnp.asarray(n_samples, dtype=jnp.int32),
        "burn_in": jnp.asarray(burn_in, dtype=jnp.int32),
        "thinning": jnp.asarray(thinning, dtype=jnp.int32),
    }
    return samples, stats


def metropolis_hastings_trajectory(
    wf: Wavefunction,
    initial_configurations: jnp.ndarray,
    times: jnp.ndarray,
    *,
    n_sites: int,
    n_samples: int,
    burn_in: int = 100,
    thinning: int = 1,
    key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Sample across a sequence of times, carrying the Markov chain forward.

    This ensures 'warm starting' where each time slice starts from the
    thermalized state of the previous time slice.

    Args:
        wf: Wavefunction object.
        initial_configurations: Starting bits (C, N).
        times: Array of time points (T,).
        n_sites: N.
        n_samples: Samples per chain per time point.
        burn_in: Initial burn-in for the FIRST time point.
        thinning: MCMC thinning.
        key: RNG key.

    Returns:
        all_samples: shape (T, C, n_samples, N)
        all_stats: Dictionary of diagnostics over time.
        final_configs: shape (C, N) from the last time point.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    configs = _as_batched_configs(initial_configurations, n_sites)

    def time_step_scan(carry, t):
        k, current_configs = carry
        k_step, k_next = jax.random.split(k)

        # For the first time step, we use the provided burn_in.
        # For subsequent steps, we could theoretically reduce it, but
        # for simplicity and correctness, we keep it consistent.
        samples, stats = metropolis_hastings_sample(
            wf=wf,
            initial_configurations=current_configs,
            t=t,
            n_sites=n_sites,
            n_samples=n_samples,
            burn_in=burn_in,
            thinning=thinning,
            key=k_step,
            return_stats=True,
        )
        next_configs = samples[:, -1, :]
        return (k_next, next_configs), (samples, stats)

    (_, final_configs), (all_samples, all_stats) = jax.lax.scan(
        time_step_scan, (key, configs), times
    )

    return all_samples, all_stats, final_configs


def autoregressive_trajectory_sample(
    wf: Wavefunction,
    times: jnp.ndarray,
    n_sites: int,
    batch_size: int,
    key: jax.Array,
    n_chains: int = 1,
    use_unique: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Sample an autoregressive trajectory exactly directly without MCMC.
    
    This replaces metropolis_hastings_trajectory for Autoregressive models.
    """
    import flax.nnx as nnx

    if n_chains <= 0:
        raise ValueError(f"n_chains must be > 0, got {n_chains}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if batch_size % n_chains != 0:
        raise ValueError(
            f"batch_size must be divisible by n_chains, got {batch_size} and {n_chains}"
        )

    n_samples_per_chain = batch_size // n_chains
    
    # We must operate directly on the amp_model inside a jitted function
    # but since this function might be called inside a JIT, we should
    # use functional split/merge to be safe with state.
    amp_model = getattr(wf, 'amp_model', getattr(wf, 'model', getattr(getattr(wf, 'model', None), 'amp_model', None)))
    if hasattr(wf.model, 'amp_model'):
        amp_model = wf.model.amp_model
    graphdef, state = nnx.split(amp_model)
    
    def generate_batch(state_val, t_val, prng_val):
        m = nnx.merge(graphdef, state_val)
        
        init_x = jnp.zeros((batch_size, n_sites), dtype=jnp.int32)
        init_kv_cache = m.init_cache(batch_size)

        def step_scan_fn(carry, step_idx):
            x_curr, prng, kv_cache = carry
            prng, subkey = jax.random.split(prng)

            safe_idx = jnp.maximum(step_idx - 1, 0)
            prev_token = jax.lax.dynamic_slice_in_dim(x_curr, safe_idx, 1, axis=1)

            current_token = jnp.where(
                step_idx == 0,
                jnp.zeros((batch_size, 1), dtype=jnp.int32),
                prev_token
            )

            logits, new_kv_cache = m(current_token, t_val, cache=kv_cache, t_index=step_idx)
            v_t = jax.random.categorical(subkey, logits.squeeze(1), axis=-1)
            x_next = x_curr.at[:, step_idx].set(v_t)
            
            return (x_next, prng, new_kv_cache), None

        init_carry = (init_x, prng_val, init_kv_cache)
        final_carry, _ = jax.lax.scan(step_scan_fn, init_carry, jnp.arange(n_sites))
        final_x = final_carry[0]
        return final_x

    def time_scan_fn(carry, t):
        prng = carry
        prng, subk = jax.random.split(prng)
        # Generate raw samples shape: (B, N)
        samples = generate_batch(state, t, subk)
        next_configs = samples.reshape((n_chains, n_samples_per_chain, n_sites))[
            :, -1, :
        ]

        if use_unique:
            samples, sample_counts = jnp.unique(
                samples,
                axis=0,
                return_counts=True,
                size=batch_size,
                fill_value=0,
            )
        else:
            sample_counts = jnp.ones((batch_size,), dtype=jnp.int32)
        
        # Match the MCMC shape contract while preserving dense AR generation.
        samples = samples.reshape((n_chains, n_samples_per_chain, n_sites))
        sample_counts = sample_counts.reshape((n_chains, n_samples_per_chain))
        
        stats = {
            "acceptance_rate": jnp.ones((n_chains,), dtype=jnp.float32),
            "n_chains": jnp.asarray(n_chains, dtype=jnp.int32),
            "n_samples_per_chain": jnp.asarray(n_samples_per_chain, dtype=jnp.int32),
            "sample_counts": sample_counts,
            "unique_count": jnp.sum(sample_counts > 0),
            "unique_fraction": jnp.sum(sample_counts > 0).astype(jnp.float32)
            / jnp.asarray(batch_size, dtype=jnp.float32),
        }
        return prng, (samples, stats, next_configs)

    _, (all_samples, all_stats, all_next_configs) = jax.lax.scan(
        time_scan_fn, key, times
    )
    
    # Final configs shape: (C, N)
    final_configs = all_next_configs[-1]
    
    return all_samples, all_stats, final_configs
