from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from src.hamiltonian import TransverseIsingHamiltonian, local_energy_batch
from src.wavefunction import Wavefunction


@dataclass
class LossDiagnostics:
    """Diagnostics returned with TDVP residual loss."""

    A: jnp.ndarray
    B: jnp.ndarray
    A_mean: jnp.ndarray
    B_mean: jnp.ndarray
    ell: jnp.ndarray
    dlogp_dt: jnp.ndarray
    dphi_dt: jnp.ndarray
    e_real: jnp.ndarray
    e_imag: jnp.ndarray


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


def _as_scalar_like(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize one-sample wf outputs to scalar."""
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 2 and arr.shape == (1, 1):
        return arr[0, 0]
    raise ValueError(f"{name} must be scalar-like for one sample, got {arr.shape}")


def _time_derivatives_single(
    wf: Wavefunction,
    configuration: jnp.ndarray,
    t: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-sample autodiff time derivatives d_t logp and d_t phi."""
    cfg = jnp.asarray(configuration).astype(jnp.int32)
    t = jnp.asarray(t, dtype=jnp.float32)

    def logp_at_time(tt):
        return _as_scalar_like(wf.log_prob(cfg, tt), "log_prob")

    def phi_at_time(tt):
        return _as_scalar_like(wf.phase(cfg, tt), "phase")

    dlogp_dt = jax.grad(logp_at_time)(t)
    dphi_dt = jax.grad(phi_at_time)(t)
    return dlogp_dt, dphi_dt


def time_derivatives_autodiff(
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Batched autodiff time derivatives for (B, N) configurations."""
    configs = jnp.asarray(configurations).astype(jnp.int32)
    if configs.ndim != 2:
        raise ValueError(
            f"Expected 2D configurations with shape (B, N), got {configs.shape}"
        )

    t = jnp.asarray(t, dtype=jnp.float32)
    dlogp_dt, dphi_dt = jax.vmap(lambda c: _time_derivatives_single(wf, c, t))(configs)
    return dlogp_dt, dphi_dt


def tdvp_residual_components(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
):
    """Compute batched residual components A and B.

    A_n = 0.5 * d_t logp_n - Im[E_loc]_n
    B_n = d_t phi_n + Re[E_loc]_n
    """
    configs = _validate_batch_configs(configurations, ham.N)

    dlogp_dt, dphi_dt = time_derivatives_autodiff(wf, configs, t)
    e_real, e_imag = local_energy_batch(ham, wf, configs, t)

    e_real = jnp.asarray(e_real)
    e_imag = jnp.asarray(e_imag)
    if e_real.ndim != 1 or e_imag.ndim != 1:
        raise ValueError(
            f"Expected local energy batch outputs shape (B,), got {e_real.shape} and {e_imag.shape}"
        )

    A = 0.5 * dlogp_dt - e_imag
    B = dphi_dt + e_real

    extras = {
        "dlogp_dt": dlogp_dt,
        "dphi_dt": dphi_dt,
        "e_real": e_real,
        "e_imag": e_imag,
    }
    return A, B, extras


def tdvp_residual_loss(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
    *,
    return_diagnostics: bool = True,
):
    """Compute Monte Carlo TDVP residual variance loss.

    ell_n = (A_n - mean(A))^2 + (B_n - mean(B))^2
    L_hat = mean(ell_n)
    """
    A, B, extras = tdvp_residual_components(ham, wf, configurations, t)

    A_mean = jnp.mean(A)
    B_mean = jnp.mean(B)

    ell = (A - A_mean) ** 2 + (B - B_mean) ** 2
    loss = jnp.mean(ell)

    if not return_diagnostics:
        return loss

    diagnostics = LossDiagnostics(
        A=A,
        B=B,
        A_mean=A_mean,
        B_mean=B_mean,
        ell=ell,
        dlogp_dt=extras["dlogp_dt"],
        dphi_dt=extras["dphi_dt"],
        e_real=extras["e_real"],
        e_imag=extras["e_imag"],
    )
    return loss, diagnostics
