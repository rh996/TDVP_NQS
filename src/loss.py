from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from src.hamiltonian import TransverseIsingHamiltonian, local_energy_batch
from src.wavefunction import Wavefunction


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(self):
        children = (
            self.A,
            self.B,
            self.A_mean,
            self.B_mean,
            self.ell,
            self.dlogp_dt,
            self.dphi_dt,
            self.e_real,
            self.e_imag,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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


def _normalize_sample_weights(
    sample_weights: Optional[jnp.ndarray],
    batch_size: int,
) -> Optional[jnp.ndarray]:
    """Return normalized nonnegative sample weights, or None for uniform weights."""
    if sample_weights is None:
        return None

    weights = jnp.asarray(sample_weights, dtype=jnp.float32)
    if weights.ndim != 1:
        raise ValueError(f"sample_weights must be 1D, got {weights.shape}")
    if weights.shape[0] != batch_size:
        raise ValueError(
            f"sample_weights length must match batch size {batch_size}, got {weights.shape[0]}"
        )
    if not isinstance(weights, jax.core.Tracer):
        if not jnp.all(weights >= 0):
            raise ValueError("sample_weights must be nonnegative.")
        if not jnp.sum(weights) > 0:
            raise ValueError("sample_weights must contain at least one positive entry.")

    total = jnp.sum(weights)
    return weights / total


def _weighted_mean(values: jnp.ndarray, weights: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Mean over the batch axis with optional normalized weights."""
    values = jnp.asarray(values)
    if weights is None:
        return jnp.mean(values)
    return jnp.sum(values * weights)


def _time_derivatives_single(
    wf: Wavefunction,
    configuration: jnp.ndarray,
    t: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-sample autodiff time derivatives d_t logp and d_t phi.

    Optimized to use jax.jvp (Forward-mode AD) to compute derivatives of both
    outputs in a single forward pass of the neural network.
    """
    cfg = jnp.asarray(configuration).astype(jnp.int32)
    t = jnp.asarray(t, dtype=jnp.float32)

    def wf_at_time(tt):
        logp, phi = wf(cfg, tt)
        return _as_scalar_like(logp, "logp"), _as_scalar_like(phi, "phi")

    # jax.jvp computes (f(x), df/dx * v). By setting v=1.0, we get the derivatives directly.
    _, (dlogp_dt, dphi_dt) = jax.jvp(wf_at_time, (t,), (jnp.ones_like(t),))
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
    sample_weights: Optional[jnp.ndarray] = None,
    loss_mode: str = "variance",
):
    """Compute Monte Carlo TDVP residual loss.

    loss_mode="variance":
      ell_n = (A_n - mean(A))^2 + (B_n - mean(B))^2

    loss_mode="schrodinger_l2":
      ell_n = A_n^2 + B_n^2

    loss_mode="phase_speed":
      ell_n = (A_n - mean(A))^2 + B_n^2

    L_hat = mean(ell_n)
    """
    A, B, extras = tdvp_residual_components(ham, wf, configurations, t)

    weights = _normalize_sample_weights(sample_weights, A.shape[0])
    if loss_mode not in ("variance", "schrodinger_l2", "phase_speed"):
        raise ValueError(
            "loss_mode must be one of {'variance', 'schrodinger_l2', 'phase_speed'}, "
            f"got {loss_mode!r}"
        )

    A_mean = _weighted_mean(A, weights)
    B_mean = _weighted_mean(B, weights)

    if loss_mode == "variance":
        ell = (A - A_mean) ** 2 + (B - B_mean) ** 2
    elif loss_mode == "schrodinger_l2":
        ell = A**2 + B**2
    else:
        ell = (A - A_mean) ** 2 + B**2
    loss = _weighted_mean(ell, weights)

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
