import flax.nnx as nnx
import jax.numpy as jnp

from src.grad import (
    tdvp_vmc_gradient,
    tdvp_vmc_gradient_components,
    tdvp_vmc_trajectory_gradient,
)
from src.hamiltonian import TransverseIsingHamiltonian
from src.loss import tdvp_residual_loss
from src.wavefunction import tSpinNQS


def _make_wf(N=5, seed=0):
    return tSpinNQS(
        N=N,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        rngs=nnx.Rngs(seed),
    )


def _make_configs():
    return jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=jnp.int32,
    )


def _assert_tree_all_finite(tree):
    leaves = jax_tree_leaves(tree)
    assert len(leaves) > 0
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        assert jnp.all(jnp.isfinite(arr))


def _tree_l2_norm(tree):
    sq = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in jax_tree_leaves(tree):
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        sq = sq + jnp.sum(arr * arr)
    return jnp.sqrt(sq)


def jax_tree_leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)


def test_tdvp_vmc_gradient_returns_nonempty_finite_pytree_and_diagnostics():
    ham = TransverseIsingHamiltonian(J=1.0, h=0.7, N=5)
    wf = _make_wf(N=5, seed=0)
    configs = _make_configs()
    t = jnp.float32(0.2)

    grad_total, diag = tdvp_vmc_gradient(
        ham=ham,
        wf=wf,
        configurations=configs,
        t=t,
        return_diagnostics=True,
    )

    # Gradient tree sanity
    _assert_tree_all_finite(grad_total)
    assert float(_tree_l2_norm(grad_total)) >= 0.0

    # Diagnostics sanity
    required = {
        "loss",
        "ell",
        "ell_mean",
        "ell_std",
        "grad_norm_pathwise",
        "grad_norm_covariance",
        "grad_norm_total",
        "ell_var",
        "finite_loss",
        "finite_grads",
    }
    assert required.issubset(set(diag.keys()))
    assert jnp.asarray(diag["loss"]).ndim == 0
    assert jnp.asarray(diag["ell"]).shape == (configs.shape[0],)
    assert bool(diag["finite_loss"])
    assert bool(diag["finite_grads"])
    assert jnp.isfinite(diag["grad_norm_total"])
    assert float(diag["grad_norm_total"]) > 0.0
    assert float(diag["grad_norm_pathwise"]) > 0.0
    assert float(diag["grad_norm_covariance"]) > 0.0


def test_tdvp_vmc_gradient_without_diagnostics():
    ham = TransverseIsingHamiltonian(J=0.8, h=1.1, N=5)
    wf = _make_wf(N=5, seed=1)
    configs = _make_configs()
    t = jnp.float32(0.5)

    grad_total, diag = tdvp_vmc_gradient(
        ham=ham,
        wf=wf,
        configurations=configs,
        t=t,
        return_diagnostics=False,
    )

    _assert_tree_all_finite(grad_total)
    assert diag == {}


def test_gradient_components_sum_to_total_and_match_loss_shape():
    ham = TransverseIsingHamiltonian(J=1.3, h=0.4, N=5)
    wf = _make_wf(N=5, seed=2)
    configs = _make_configs()
    t = jnp.float32(0.15)

    grad_total, grad_pathwise, grad_cov, diag = tdvp_vmc_gradient_components(
        ham=ham,
        wf=wf,
        configurations=configs,
        t=t,
    )

    # total == pathwise + covariance leafwise
    import jax

    diffs = jax.tree_util.tree_map(
        lambda gt, gp, gc: gt - (gp + gc),
        grad_total,
        grad_pathwise,
        grad_cov,
    )
    diff_norm = _tree_l2_norm(diffs)
    assert jnp.allclose(diff_norm, 0.0, atol=1e-6)

    # Diagnostics shape/type checks
    assert "loss" in diag and jnp.asarray(diag["loss"]).ndim == 0
    assert "ell" in diag and jnp.asarray(diag["ell"]).shape == (configs.shape[0],)
    assert "ell_mean" in diag and jnp.asarray(diag["ell_mean"]).ndim == 0
    assert "ell_std" in diag and jnp.asarray(diag["ell_std"]).ndim == 0
    assert "finite_loss" in diag
    assert "finite_grads" in diag
    assert bool(diag["finite_loss"])
    assert bool(diag["finite_grads"])


def test_tdvp_vmc_gradient_loss_matches_phase4_loss():
    ham = TransverseIsingHamiltonian(J=0.9, h=0.9, N=5)
    wf = _make_wf(N=5, seed=3)
    configs = _make_configs()
    t = jnp.float32(0.33)

    grad_total, diag = tdvp_vmc_gradient(
        ham=ham,
        wf=wf,
        configurations=configs,
        t=t,
        return_diagnostics=True,
    )
    _assert_tree_all_finite(grad_total)

    loss_phase4 = tdvp_residual_loss(
        ham=ham,
        wf=wf,
        configurations=configs,
        t=t,
        return_diagnostics=False,
    )

    assert jnp.allclose(diag["loss"], loss_phase4, atol=1e-6)


def test_tdvp_vmc_gradient_depends_on_model_parameters():
    ham = TransverseIsingHamiltonian(J=1.0, h=0.7, N=5)
    wf = _make_wf(N=5, seed=5)
    configs = _make_configs()
    t = jnp.float32(0.2)

    grad_total, grad_pathwise, grad_cov, _ = tdvp_vmc_gradient_components(
        ham=ham,
        wf=wf,
        configurations=configs,
        t=t,
    )

    assert float(_tree_l2_norm(grad_total)) > 0.0
    assert float(_tree_l2_norm(grad_pathwise)) > 0.0
    assert float(_tree_l2_norm(grad_cov)) > 0.0


def test_tdvp_vmc_gradient_input_validation_shape_length_binary():
    ham = TransverseIsingHamiltonian(J=1.0, h=0.5, N=5)
    wf = _make_wf(N=5, seed=4)
    t = jnp.float32(0.0)

    # wrong rank
    bad_rank = jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32)
    try:
        _ = tdvp_vmc_gradient(ham, wf, bad_rank, t)
        assert False, "Expected ValueError for non-2D configurations."
    except ValueError as e:
        assert "Expected 2D configurations" in str(e)

    # wrong N dimension
    bad_len = jnp.array([[0, 1, 0], [1, 0, 1]], dtype=jnp.int32)
    try:
        _ = tdvp_vmc_gradient(ham, wf, bad_len, t)
        assert False, "Expected ValueError for wrong configuration length."
    except ValueError as e:
        assert "Expected configuration length" in str(e)

    # non-binary entries
    bad_binary = jnp.array([[0, 1, 2, 0, 1]], dtype=jnp.int32)
    try:
        _ = tdvp_vmc_gradient(ham, wf, bad_binary, t)
        assert False, "Expected ValueError for non-binary configurations."
    except ValueError as e:
        assert "binary bits in {0,1}" in str(e)


def test_tdvp_vmc_trajectory_gradient_returns_valid_output():
    import jax
    N = 4
    ham = TransverseIsingHamiltonian(J=1.0, h=0.5, N=N)
    wf = _make_wf(N=N, seed=42)
    
    T = 3
    times = jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32)
    B = 4
    # (T, B, N)
    all_configs = jax.random.randint(jax.random.PRNGKey(0), (T, B, N), 0, 2)
    
    grad_total, diag = tdvp_vmc_trajectory_gradient(
        ham=ham,
        wf=wf,
        all_configurations=all_configs,
        times=times,
    )
    
    _assert_tree_all_finite(grad_total)
    assert float(_tree_l2_norm(grad_total)) > 0.0
    
    required = {"loss", "grad_norm_total", "e_real_mean", "e_imag_mean", "ell_mean"}
    assert required.issubset(set(diag.keys()))
    assert jnp.isfinite(diag["loss"])
    assert jnp.isfinite(diag["grad_norm_total"])
