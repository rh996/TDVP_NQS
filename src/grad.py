from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from src.hamiltonian import TransverseIsingHamiltonian
from src.loss import tdvp_residual_loss
from src.wavefunction import Wavefunction

PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass
class GradientDiagnostics:
    """Diagnostics for Phase-5 TDVP/VMC gradient estimation."""

    loss: jnp.ndarray
    ell: jnp.ndarray
    ell_mean: jnp.ndarray
    ell_std: jnp.ndarray
    grad_norm_pathwise: jnp.ndarray
    grad_norm_covariance: jnp.ndarray
    grad_norm_total: jnp.ndarray
    ell_var: jnp.ndarray
    finite_loss: jnp.ndarray
    finite_grads: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.loss,
            self.ell,
            self.ell_mean,
            self.ell_std,
            self.grad_norm_pathwise,
            self.grad_norm_covariance,
            self.grad_norm_total,
            self.ell_var,
            self.finite_loss,
            self.finite_grads,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class _ModelWavefunctionView(Wavefunction):
    """Wavefunction adapter that evaluates through a provided NNX model."""

    def __init__(self, model):
        self.model = model

    def __call__(self, configuration, t):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(x)
        if arr.ndim > 1 and arr.shape[-1] == 1:
            return jnp.squeeze(arr, axis=-1)
        return arr


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


def _tree_add(a: PyTree, b: PyTree) -> PyTree:
    """Add two pytrees leaf-wise."""
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(tree: PyTree, scalar: jnp.ndarray) -> PyTree:
    """Scale a pytree by a scalar."""
    return jax.tree_util.tree_map(lambda x: x * scalar, tree)


def _tree_l2_norm(tree: PyTree) -> jnp.ndarray:
    """Compute L2 norm of all leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    sq = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        leaf_f = jnp.asarray(leaf, dtype=jnp.float32)
        sq = sq + jnp.sum(leaf_f * leaf_f)
    return jnp.sqrt(sq)


def _tree_all_finite(tree: PyTree) -> jnp.ndarray:
    """Check if all leaves in a pytree are finite."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    finite_flags = [jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
    result = finite_flags[0]
    for flag in finite_flags[1:]:
        result = jnp.logical_and(result, flag)
    return result


def _as_scalar_like(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize outputs to scalar."""
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

    return weights / jnp.sum(weights)


def _weighted_mean_array(
    values: jnp.ndarray,
    weights: Optional[jnp.ndarray],
) -> jnp.ndarray:
    values = jnp.asarray(values)
    if weights is None:
        return jnp.mean(values, axis=0)
    weight_shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.sum(values * weights.reshape(weight_shape), axis=0)


def _weighted_mean_tree(tree: PyTree, weights: Optional[jnp.ndarray]) -> PyTree:
    return jax.tree_util.tree_map(lambda x: _weighted_mean_array(x, weights), tree)


def _batch_loss_fn(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
    sample_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute scalar phase-4 loss for fixed sampled batch."""
    return tdvp_residual_loss(
        ham,
        wf,
        configurations,
        t,
        return_diagnostics=False,
        sample_weights=sample_weights,
    )


def _per_sample_score_grads(
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
) -> PyTree:
    """Compute per-sample score gradients g_n = ∂_θ log p(σ_n, t).

    Uses nnx.vmap to compute gradients for the batch efficiently.
    Returns a single gradient pytree where each leaf has a leading batch dimension.
    """
    import flax.nnx as nnx

    def logp_fn(m, cfg_i):
        model_wf = _ModelWavefunctionView(m)
        logp, _ = model_wf(cfg_i, t)
        return _as_scalar_like(logp, "log_prob")

    grad_fn = nnx.grad(logp_fn)
    vmap_grad_fn = nnx.vmap(grad_fn, in_axes=(None, 0))
    return vmap_grad_fn(wf.model, configurations)


def tdvp_vmc_gradient(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
    *,
    return_diagnostics: bool = True,
    sample_weights: Optional[jnp.ndarray] = None,
) -> Tuple[PyTree, Dict[str, Any]]:
    """Phase-5 gradient estimator: pathwise + covariance correction.

    Estimator:
      ∂θ L ≈ (1/B) Σ_n ∂θ ell_n + (1/B) Σ_n (ell_n - mean(ell)) (g_n - mean(g))

    where:
      g_n = ∂θ log p(σ_n, t)

    Notes:
      - Samples are treated fixed in the pathwise term.
      - The covariance correction accounts for parameter dependence of the sampling distribution.
    """
    import flax.nnx as nnx

    configs = _validate_batch_configs(configurations, ham.N)
    weights = _normalize_sample_weights(sample_weights, configs.shape[0])

    # 1) Pathwise gradient on fixed sample batch.
    def loss_for_grad(m):
        model_wf = _ModelWavefunctionView(m)
        return _batch_loss_fn(ham, model_wf, configs, t, sample_weights=weights)

    grad_pathwise = nnx.grad(loss_for_grad)(wf.model)

    # 2) Per-sample losses and centered ell.
    _, loss_diag = tdvp_residual_loss(
        ham,
        wf,
        configs,
        t,
        return_diagnostics=True,
        sample_weights=weights,
    )
    ell = jnp.asarray(loss_diag.ell)
    ell_mean = _weighted_mean_array(ell, weights)
    ell_centered = ell - ell_mean

    # 3) Per-sample score gradients g_n = ∂θ log p_n.
    score_grads_batched = _per_sample_score_grads(wf, configs, t)
    score_mean = _weighted_mean_tree(score_grads_batched, weights)

    # 4) Center score gradients leaf-wise.
    score_centered_batched = jax.tree_util.tree_map(
        lambda g, gm: g - gm, score_grads_batched, score_mean
    )

    # 5) Covariance correction: mean_n [ ell_c[n] * score_c[n] ].
    # For each parameter leaf, compute: mean_n [ ell_centered[n] * score_centered[n] ]
    grad_cov = jax.tree_util.tree_map(
        lambda score_c: jnp.mean(
            score_c * ell_centered.reshape(
                (ell_centered.shape[0],) + (1,) * (score_c.ndim - 1)
            ),
            axis=0,
        )
        if weights is None
        else jnp.sum(
            score_c
            * ell_centered.reshape(
                (ell_centered.shape[0],) + (1,) * (score_c.ndim - 1)
            )
            * weights.reshape((weights.shape[0],) + (1,) * (score_c.ndim - 1)),
            axis=0,
        ),
        score_centered_batched,
    )

    # 6) Total gradient.
    grad_total = _tree_add(grad_pathwise, grad_cov)

    if not return_diagnostics:
        return grad_total, {}

    grad_norm_pathwise = _tree_l2_norm(grad_pathwise)
    grad_norm_covariance = _tree_l2_norm(grad_cov)
    grad_norm_total = _tree_l2_norm(grad_total)

    ell_var = _weighted_mean_array(ell_centered * ell_centered, weights)
    finite_loss = jnp.all(jnp.isfinite(ell)) & jnp.isfinite(
        _batch_loss_fn(ham, wf, configs, t, sample_weights=weights)
    )
    finite_grads = (
        _tree_all_finite(grad_pathwise)
        & _tree_all_finite(grad_cov)
        & _tree_all_finite(grad_total)
    )

    diagnostics = GradientDiagnostics(
        loss=_batch_loss_fn(ham, wf, configs, t, sample_weights=weights),
        ell=ell,
        ell_mean=ell_mean,
        ell_std=jnp.sqrt(ell_var),
        grad_norm_pathwise=grad_norm_pathwise,
        grad_norm_covariance=grad_norm_covariance,
        grad_norm_total=grad_norm_total,
        ell_var=ell_var,
        finite_loss=finite_loss,
        finite_grads=finite_grads,
    )

    aux = {
        "loss": diagnostics.loss,
        "ell": diagnostics.ell,
        "ell_mean": diagnostics.ell_mean,
        "ell_std": diagnostics.ell_std,
        "grad_norm_pathwise": diagnostics.grad_norm_pathwise,
        "grad_norm_covariance": diagnostics.grad_norm_covariance,
        "grad_norm_total": diagnostics.grad_norm_total,
        "ell_var": diagnostics.ell_var,
        "finite_loss": diagnostics.finite_loss,
        "finite_grads": diagnostics.finite_grads,
    }
    return grad_total, aux


def tdvp_vmc_trajectory_gradient(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    all_configurations: jnp.ndarray,
    times: jnp.ndarray,
    sample_weights: Optional[jnp.ndarray] = None,
) -> Tuple[PyTree, Dict[str, Any]]:
    """Compute unified TDVP gradient averaged over multiple time slices.

    This version is optimized for TPU memory and JAX trace-safety. It uses
    a functional approach to accumulate gradients slice-by-slice.
    """
    import flax.nnx as nnx

    # Split model into functional components
    graphdef, state = nnx.split(wf.model)

    if sample_weights is not None:
        sample_weights = jnp.asarray(sample_weights, dtype=jnp.float32)
        if sample_weights.ndim != 2:
            raise ValueError(f"sample_weights must have shape (T, B), got {sample_weights.shape}")
        if sample_weights.shape[:2] != all_configurations.shape[:2]:
            raise ValueError(
                "sample_weights shape must match all_configurations leading dimensions "
                f"{all_configurations.shape[:2]}, got {sample_weights.shape}"
            )

    def compute_slice_gradient(state_val, t_val, configs, weights_val):
        """Compute total TDVP gradient for a single slice functionally."""
        weights = _normalize_sample_weights(weights_val, configs.shape[0])
        
        # 1) Reconstruct model view for this slice
        m = nnx.merge(graphdef, state_val)
        model_wf = _ModelWavefunctionView(m)

        # 2) Pathwise piece for this slice
        def pathwise_loss(s):
            m_p = nnx.merge(graphdef, s)
            mw = _ModelWavefunctionView(m_p)
            return _batch_loss_fn(ham, mw, configs, t_val, sample_weights=weights)

        grad_pathwise = jax.grad(pathwise_loss)(state_val)

        # 3) Covariance piece for this slice
        slice_loss, loss_diag = tdvp_residual_loss(
            ham,
            model_wf,
            configs,
            t_val,
            return_diagnostics=True,
            sample_weights=weights,
        )
        ell = jnp.asarray(loss_diag.ell)
        ell_centered = ell - _weighted_mean_array(ell, weights)

        def logp_fn(s, cfg_i):
            m_s = nnx.merge(graphdef, s)
            mw = _ModelWavefunctionView(m_s)
            lp, _ = mw(cfg_i, t_val)
            return _as_scalar_like(lp, "log_prob")

        # Functional score gradients: (Batch, Params...)
        score_grads = jax.vmap(jax.grad(logp_fn), in_axes=(None, 0))(state_val, configs)
        
        score_mean = _weighted_mean_tree(score_grads, weights)
        score_centered = jax.tree_util.tree_map(lambda g, gm: g - gm, score_grads, score_mean)

        grad_cov = jax.tree_util.tree_map(
            lambda sc: jnp.mean(
                sc * ell_centered.reshape((ell_centered.shape[0],) + (1,) * (sc.ndim - 1)),
                axis=0,
            )
            if weights is None
            else jnp.sum(
                sc
                * ell_centered.reshape((ell_centered.shape[0],) + (1,) * (sc.ndim - 1))
                * weights.reshape((weights.shape[0],) + (1,) * (sc.ndim - 1)),
                axis=0,
            ),
            score_centered,
        )

        grad_total = jax.tree_util.tree_map(lambda x, y: x + y, grad_pathwise, grad_cov)
        return grad_total, loss_diag, slice_loss

    if sample_weights is None:
        scan_args = (
            times,
            all_configurations,
            jnp.ones(all_configurations.shape[:2], dtype=jnp.float32),
        )
    else:
        scan_args = (times, all_configurations, sample_weights)

    def trajectory_scan(unused_carry, args):
        t_val, configs, weights_val = args
        g, d, loss_val = compute_slice_gradient(state, t_val, configs, weights_val)
        return None, (g, d, loss_val)

    # Sequential scan over time to keep memory usage constant O(Batch)
    _, (grads_all, diags_all, losses_all) = jax.lax.scan(
        trajectory_scan, None, scan_args
    )

    # Aggregate gradients and diagnostics
    # Use jax.tree_util.tree_map directly to ensure we handle any dict/State hybrid
    # We use jnp.mean to average over the time dimension, preventing the TDVP gradient
    # from overpowering the initial condition Lagrangian penalty when time_steps is large.
    grad_total = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads_all)

    diag_summary = {
        "loss": jnp.mean(losses_all),
        "grad_norm_total": _tree_l2_norm(grad_total),
        "e_real_mean": jnp.mean(diags_all.e_real),
        "e_imag_mean": jnp.mean(diags_all.e_imag),
        "ell_mean": jnp.mean(losses_all),
    }

    return grad_total, diag_summary


def tdvp_vmc_gradient_components(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
) -> Tuple[PyTree, PyTree, PyTree, Dict[str, Any]]:
    """Return (grad_total, grad_pathwise, grad_cov, diagnostics).

    Useful for debugging and analysis of gradient components.
    """
    import flax.nnx as nnx

    configs = _validate_batch_configs(configurations, ham.N)

    def loss_for_grad(m):
        model_wf = _ModelWavefunctionView(m)
        return _batch_loss_fn(ham, model_wf, configs, t)

    grad_pathwise = nnx.grad(loss_for_grad)(wf.model)

    _, loss_diag = tdvp_residual_loss(ham, wf, configs, t, return_diagnostics=True)
    ell = jnp.asarray(loss_diag.ell)
    ell_mean = jnp.mean(ell)
    ell_centered = ell - ell_mean

    score_grads_batched = _per_sample_score_grads(wf, configs, t)
    score_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), score_grads_batched)

    score_centered_batched = jax.tree_util.tree_map(
        lambda g, gm: g - gm, score_grads_batched, score_mean
    )

    grad_cov = jax.tree_util.tree_map(
        lambda score_c: jnp.mean(
            score_c * ell_centered.reshape(
                (ell_centered.shape[0],) + (1,) * (score_c.ndim - 1)
            ),
            axis=0,
        ),
        score_centered_batched,
    )
    grad_total = _tree_add(grad_pathwise, grad_cov)

    diagnostics = {
        "loss": _batch_loss_fn(ham, wf, configs, t),
        "ell": ell,
        "ell_mean": ell_mean,
        "ell_std": jnp.std(ell),
        "grad_norm_pathwise": _tree_l2_norm(grad_pathwise),
        "grad_norm_covariance": _tree_l2_norm(grad_cov),
        "grad_norm_total": _tree_l2_norm(grad_total),
        "finite_loss": jnp.isfinite(_batch_loss_fn(ham, wf, configs, t)),
        "finite_grads": _tree_all_finite(grad_total),
    }

    return grad_total, grad_pathwise, grad_cov, diagnostics
