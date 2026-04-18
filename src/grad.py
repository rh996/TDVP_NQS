from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

from src.hamiltonian import TransverseIsingHamiltonian
from src.loss import tdvp_residual_loss
from src.wavefunction import Wavefunction

PyTree = Any


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


def _tree_add(a: PyTree, b: PyTree) -> PyTree:
    """Add two pytrees leaf-wise."""
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(tree: PyTree, scalar: jnp.ndarray) -> PyTree:
    """Scale a pytree by a scalar."""
    return jax.tree_util.tree_map(lambda x: x * scalar, tree)


def _tree_mean(trees_list: List[PyTree]) -> PyTree:
    """Average a list of pytrees leaf-wise."""
    if not trees_list:
        raise ValueError("Cannot compute mean of empty pytree list.")
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees_list)
    return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), stacked)


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


def _batch_loss_fn(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
) -> jnp.ndarray:
    """Compute scalar phase-4 loss for fixed sampled batch."""
    return tdvp_residual_loss(ham, wf, configurations, t, return_diagnostics=False)


def _per_sample_score_grads(
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
) -> List[PyTree]:
    """Compute per-sample score gradients g_n = ∂_θ log p(σ_n, t).

    Uses a Python loop (not vmap) to avoid NNX trace-level aliasing issues.
    Returns a list of gradient pytrees, one per sample.
    """
    import flax.nnx as nnx

    grads_list = []
    for i in range(configurations.shape[0]):
        cfg_i = configurations[i]

        def logp_fn(m):
            return _as_scalar_like(wf.log_prob(cfg_i, t), "log_prob")

        grad_i = nnx.grad(logp_fn)(wf.model)
        grads_list.append(grad_i)

    return grads_list


def tdvp_vmc_gradient(
    ham: TransverseIsingHamiltonian,
    wf: Wavefunction,
    configurations: jnp.ndarray,
    t,
    *,
    return_diagnostics: bool = True,
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

    # 1) Pathwise gradient on fixed sample batch.
    def loss_for_grad(m):
        return _batch_loss_fn(ham, wf, configs, t)

    grad_pathwise = nnx.grad(loss_for_grad)(wf.model)

    # 2) Per-sample losses and centered ell.
    _, loss_diag = tdvp_residual_loss(ham, wf, configs, t, return_diagnostics=True)
    ell = jnp.asarray(loss_diag.ell)
    ell_mean = jnp.mean(ell)
    ell_centered = ell - ell_mean

    # 3) Per-sample score gradients g_n = ∂θ log p_n.
    score_grads_list = _per_sample_score_grads(wf, configs, t)
    score_mean = _tree_mean(score_grads_list)

    # 4) Center score gradients leaf-wise.
    score_centered_list = [
        jax.tree_util.tree_map(lambda g, gm: g - gm, g, score_mean)
        for g in score_grads_list
    ]

    # 5) Covariance correction: mean_n [ ell_c[n] * score_c[n] ].
    # For each parameter leaf, compute: mean_n [ ell_centered[n] * score_centered[n] ]
    grad_cov = jax.tree_util.tree_map(
        lambda *score_leaves: jnp.mean(
            jnp.stack(score_leaves, axis=0)
            * ell_centered.reshape(
                (ell_centered.shape[0],) + (1,) * score_leaves[0].ndim
            ),
            axis=0,
        ),
        *score_centered_list,
    )

    # 6) Total gradient.
    grad_total = _tree_add(grad_pathwise, grad_cov)

    if not return_diagnostics:
        return grad_total, {}

    grad_norm_pathwise = _tree_l2_norm(grad_pathwise)
    grad_norm_covariance = _tree_l2_norm(grad_cov)
    grad_norm_total = _tree_l2_norm(grad_total)

    ell_var = jnp.mean(ell_centered * ell_centered)
    finite_loss = jnp.all(jnp.isfinite(ell)) & jnp.isfinite(
        _batch_loss_fn(ham, wf, configs, t)
    )
    finite_grads = (
        _tree_all_finite(grad_pathwise)
        & _tree_all_finite(grad_cov)
        & _tree_all_finite(grad_total)
    )

    diagnostics = GradientDiagnostics(
        loss=_batch_loss_fn(ham, wf, configs, t),
        ell=ell,
        ell_mean=ell_mean,
        ell_std=jnp.std(ell),
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
        return _batch_loss_fn(ham, wf, configs, t)

    grad_pathwise = nnx.grad(loss_for_grad)(wf.model)

    _, loss_diag = tdvp_residual_loss(ham, wf, configs, t, return_diagnostics=True)
    ell = jnp.asarray(loss_diag.ell)
    ell_mean = jnp.mean(ell)
    ell_centered = ell - ell_mean

    score_grads_list = _per_sample_score_grads(wf, configs, t)
    score_mean = _tree_mean(score_grads_list)

    score_centered_list = [
        jax.tree_util.tree_map(lambda g, gm: g - gm, g, score_mean)
        for g in score_grads_list
    ]

    grad_cov = jax.tree_util.tree_map(
        lambda *score_leaves: jnp.mean(
            jnp.stack(score_leaves, axis=0)
            * ell_centered.reshape(
                (ell_centered.shape[0],) + (1,) * score_leaves[0].ndim
            ),
            axis=0,
        ),
        *score_centered_list,
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
