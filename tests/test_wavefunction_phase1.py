import flax.nnx as nn
import jax.numpy as jnp
import pytest

from src.wavefunction import (
    AttentionResiduals,
    AutoregressiveNQS,
    AutoregressiveNQS_Z2,
    CausalTransformerLayer,
    FixedCoefficientNeuralGalerkinNQS,
    NeuralGalerkinNQS,
    SimpleSpinNQS,
    TimeFeatureMap,
    odd_silu,
    tSpinNQS,
    tSpinNQS_Z2,
    xsa_output,
)


def _make_model(wf_class, N=6):
    return wf_class(
        N=N,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        rngs=nn.Rngs(0),
    )


def test_xsa_output_projects_out_self_value_direction():
    attn_out = jnp.array(
        [
            [
                [[2.0, 1.0, 0.0], [1.0, -1.0, 2.0]],
                [[0.5, 2.0, -1.0], [3.0, 0.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    v_self = jnp.array(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
            ]
        ],
        dtype=jnp.float32,
    )

    projected = xsa_output(attn_out, v_self)
    residual_dot = jnp.sum(projected * v_self, axis=-1)

    assert projected.shape == attn_out.shape
    assert jnp.allclose(residual_dot, 0.0, atol=1e-5)


def test_causal_transformer_cached_xsa_path_preserves_shape():
    layer = CausalTransformerLayer(
        feature_dim=8,
        num_heads=2,
        head_dim=4,
        out_dim=8,
        rngs=nn.Rngs(0),
    )
    x = jnp.ones((3, 1, 8), dtype=jnp.float32)
    cache = (
        jnp.zeros((3, 5, 2, 4), dtype=jnp.float32),
        jnp.zeros((3, 5, 2, 4), dtype=jnp.float32),
    )

    out, new_cache = layer(x, cache=cache, t_index=2)

    assert out.shape == x.shape
    assert new_cache[0].shape == cache[0].shape
    assert new_cache[1].shape == cache[1].shape


def test_time_feature_map_has_fourier_and_positive_decay_features():
    feature_map = TimeFeatureMap(num_fourier_bands=3, use_exp_decay=True, rngs=nn.Rngs(0))
    features = feature_map(jnp.float32(0.25), batch_dim=4)

    assert features.shape == (4, 8)
    assert jnp.all(jnp.isfinite(features))
    assert feature_map.output_dim == 8
    assert float(feature_map.decay_rate()) > 0.0


def test_attention_residuals_even_logits_preserve_odd_equivariance():
    residuals = AttentionResiduals(
        num_layers=2, feature_dim=4, rngs=nn.Rngs(0), use_even_logits=True
    )
    blocks = [
        jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10.0,
        jnp.ones((2, 3, 4), dtype=jnp.float32),
    ]
    partial = -0.5 * jnp.ones((2, 3, 4), dtype=jnp.float32)

    out = residuals(blocks, partial, layer_index=1)
    flipped_out = residuals([-block for block in blocks], -partial, layer_index=1)

    assert out.shape == partial.shape
    assert jnp.allclose(flipped_out, -out, atol=1e-5)


ALL_WAVEFUNCTIONS = [
    tSpinNQS,
    tSpinNQS_Z2,
    SimpleSpinNQS,
    AutoregressiveNQS,
    AutoregressiveNQS_Z2,
    NeuralGalerkinNQS,
]
Z2_WAVEFUNCTIONS = [tSpinNQS_Z2, SimpleSpinNQS, AutoregressiveNQS_Z2]


@pytest.mark.parametrize("wf_class", ALL_WAVEFUNCTIONS)
def test_tspinnqs_holds_persistent_model_instance(wf_class):
    wf = _make_model(wf_class, N=6)

    # In AutoregressiveNQS, wf.model doesn't exist, it has amp_model and phase_model.
    # We can check if any underlying module has a consistent ID.
    if hasattr(wf, "model"):
        model_id_before = id(wf.model)
        _ = wf(jnp.array([0, 1, 0, 1, 1, 0], dtype=jnp.int32), t=jnp.float32(0.2))
        model_id_after_first_call = id(wf.model)
        _ = wf(jnp.array([1, 0, 1, 0, 0, 1], dtype=jnp.int32), t=jnp.float32(0.3))
        model_id_after_second_call = id(wf.model)
        assert model_id_before == model_id_after_first_call == model_id_after_second_call
    elif hasattr(wf, "amp_model"):
        model_id_before = id(wf.amp_model)
        _ = wf(jnp.array([0, 1, 0, 1, 1, 0], dtype=jnp.int32), t=jnp.float32(0.2))
        model_id_after_first_call = id(wf.amp_model)
        _ = wf(jnp.array([1, 0, 1, 0, 0, 1], dtype=jnp.int32), t=jnp.float32(0.3))
        model_id_after_second_call = id(wf.amp_model)
        assert model_id_before == model_id_after_first_call == model_id_after_second_call


@pytest.mark.parametrize("wf_class", ALL_WAVEFUNCTIONS)
def test_call_output_shapes_for_single_and_batch_inputs(wf_class):
    N = 5
    wf = _make_model(wf_class, N=N)

    # Single configuration input: shape (N,)
    single_config = jnp.array([0, 1, 1, 0, 1], dtype=jnp.int32)
    single_logp, single_phi = wf(single_config, t=jnp.float32(0.1))

    # Wrapper squeezes trailing singleton dim, but keeps batch axis from the model path.
    # For a single input (N,), output is shape (1,).
    assert jnp.asarray(single_logp).shape == (1,)
    assert jnp.asarray(single_phi).shape == (1,)

    # Batched input: shape (B, N)
    batch_config = jnp.array(
        [
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=jnp.int32,
    )
    batch_logp, batch_phi = wf(batch_config, t=jnp.float32(0.1))

    assert batch_logp.shape == (batch_config.shape[0],)
    assert batch_phi.shape == (batch_config.shape[0],)


@pytest.mark.parametrize("wf_class", ALL_WAVEFUNCTIONS)
def test_outputs_are_finite_for_valid_inputs(wf_class):
    N = 7
    wf = _make_model(wf_class, N=N)

    configs = jnp.array(
        [
            [0, 0, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 1, 0],
        ],
        dtype=jnp.int32,
    )
    t = jnp.float32(1.25)

    logp, phi = wf(configs, t)

    assert jnp.all(jnp.isfinite(logp))
    assert jnp.all(jnp.isfinite(phi))


def test_autoregressive_time_gate_is_multiplicative_odd_silu_path():
    wf = _make_model(AutoregressiveNQS_Z2, N=5)
    amp_model = wf.model.amp_model

    batch_dim = 3
    t = jnp.float32(0.25)
    time_features = amp_model.time_features(t, batch_dim)
    expected = odd_silu(amp_model.time_mlp1(time_features))
    expected = odd_silu(amp_model.time_mlp2(expected))
    expected = 1.0 + odd_silu(amp_model.time_mlp3(expected))

    actual = amp_model._time_gate(t, batch_dim)

    assert hasattr(amp_model, "time_mlp3")
    assert actual.shape == (batch_dim, amp_model.emb_dim)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_neural_galerkin_coefficients_are_zero_at_initial_time():
    wf = NeuralGalerkinNQS(
        N=4,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        num_basis=3,
        num_modes=2,
        rngs=nn.Rngs(0),
    )

    coeffs = wf.model.coefficients(jnp.float32(0.0))

    assert coeffs.shape == (3,)
    assert jnp.allclose(coeffs, 0.0, atol=1e-7)


def test_neural_galerkin_is_uniform_unnormalized_at_t0():
    wf = NeuralGalerkinNQS(
        N=4,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        num_basis=2,
        num_modes=2,
        rngs=nn.Rngs(1),
    )
    configs = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=jnp.int32,
    )

    logp, phi = wf(configs, jnp.float32(0.0))

    assert jnp.allclose(logp, 0.0, atol=1e-6)
    assert jnp.allclose(phi, 0.0, atol=1e-6)


def test_neural_galerkin_basis_values_include_uniform_basis():
    wf = NeuralGalerkinNQS(
        N=4,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        num_basis=2,
        num_modes=2,
        rngs=nn.Rngs(1),
    )
    configs = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)

    basis_values = wf.model.basis_values(configs)

    assert basis_values.shape == (3, 2)
    assert jnp.allclose(basis_values[0], 1.0 + 0.0j, atol=1e-6)
    assert jnp.all(jnp.isfinite(jnp.real(basis_values)))
    assert jnp.all(jnp.isfinite(jnp.imag(basis_values)))


def test_fixed_coefficient_neural_galerkin_zero_generator_keeps_initial_state():
    wf = NeuralGalerkinNQS(
        N=4,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        num_basis=2,
        num_modes=2,
        rngs=nn.Rngs(2),
    )
    generator = jnp.zeros((3, 3), dtype=jnp.complex64)
    ode_wf = FixedCoefficientNeuralGalerkinNQS(wf, generator)
    configs = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)

    logp, phi = ode_wf(configs, jnp.float32(0.7))

    assert jnp.allclose(logp, 0.0, atol=1e-6)
    assert jnp.allclose(phi, 0.0, atol=1e-6)


def test_neural_galerkin_outputs_change_after_initial_time():
    wf = NeuralGalerkinNQS(
        N=4,
        Num_boxes=1,
        emb_dim=8,
        num_heads=2,
        head_dim=4,
        num_basis=2,
        num_modes=2,
        rngs=nn.Rngs(2),
    )
    configs = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int32)

    logp0, phi0 = wf(configs, jnp.float32(0.0))
    logp1, phi1 = wf(configs, jnp.float32(0.7))

    assert jnp.all(jnp.isfinite(logp1))
    assert jnp.all(jnp.isfinite(phi1))
    assert not (
        jnp.allclose(logp0, logp1, atol=1e-7)
        and jnp.allclose(phi0, phi1, atol=1e-7)
    )


def test_autoregressive_token_features_are_spin_flip_odd_with_zero_sos():
    wf = _make_model(AutoregressiveNQS_Z2, N=5)
    amp_model = wf.model.amp_model

    inputs = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    features = amp_model._token_features(inputs)

    assert jnp.allclose(features[:, 0, :], -features[:, 1, :], atol=1e-6)
    assert jnp.allclose(features[:, 2, :], 0.0, atol=1e-6)


@pytest.mark.parametrize("wf_class", Z2_WAVEFUNCTIONS)
def test_log_probability_is_global_spin_flip_symmetric(wf_class):
    wf = _make_model(wf_class, N=6)
    configs = jnp.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    flipped_configs = 1 - configs

    logp, _ = wf(configs, t=jnp.float32(0.4))
    flipped_logp, _ = wf(flipped_configs, t=jnp.float32(0.4))

    assert jnp.allclose(logp, flipped_logp, atol=1e-5)


def test_legacy_tspinnqs_is_not_forced_to_be_spin_flip_symmetric():
    wf = _make_model(tSpinNQS, N=6)
    configs = jnp.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    logp, _ = wf(configs, t=jnp.float32(0.4))
    flipped_logp, _ = wf(1 - configs, t=jnp.float32(0.4))

    assert not jnp.allclose(logp, flipped_logp, atol=1e-5)


def test_legacy_autoregressive_nqs_is_not_forced_to_be_spin_flip_symmetric():
    wf = _make_model(AutoregressiveNQS, N=6)
    configs = jnp.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    logp, _ = wf(configs, t=jnp.float32(0.4))
    flipped_logp, _ = wf(1 - configs, t=jnp.float32(0.4))

    assert not jnp.allclose(logp, flipped_logp, atol=1e-5)


def test_autoregressive_conditional_logits_swap_under_spin_flip():
    wf = _make_model(AutoregressiveNQS_Z2, N=6)
    amp_model = wf.model.amp_model
    configs = jnp.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
        ],
        dtype=jnp.int32,
    )

    logits, _ = amp_model(configs, t=jnp.float32(0.7))
    flipped_logits, _ = amp_model(1 - configs, t=jnp.float32(0.7))

    assert jnp.allclose(flipped_logits[..., 0], logits[..., 1], atol=1e-5)
    assert jnp.allclose(flipped_logits[..., 1], logits[..., 0], atol=1e-5)


def test_autoregressive_symmetry_preserving_layers_are_bias_free():
    wf = _make_model(AutoregressiveNQS_Z2, N=5)
    amp_model = wf.model.amp_model
    layer = amp_model.layers[0]

    assert layer.layernorm.bias is None
    assert layer.layernorm2.bias is None
    assert layer.ffn.bias is None
    assert layer.transformer.q_proj.bias is None
    assert layer.transformer.k_proj.bias is None
    assert layer.transformer.v_proj.bias is None
    assert layer.transformer.out_proj.bias is None
    assert amp_model.time_mlp1.bias is None
    assert amp_model.time_mlp2.bias is None
    assert amp_model.time_mlp3.bias is None
    assert amp_model.logits_score.bias is None
    assert not hasattr(amp_model, "pos_embeds")


@pytest.mark.parametrize("wf_class", ALL_WAVEFUNCTIONS)
def test_single_vs_singleton_batch_consistency(wf_class):
    N = 5
    wf = _make_model(wf_class, N=N)

    config = jnp.array([1, 0, 1, 1, 0], dtype=jnp.int32)
    config_batched = config[jnp.newaxis, :]
    t = jnp.float32(0.9)

    logp_single, phi_single = wf(config, t)
    logp_batch, phi_batch = wf(config_batched, t)

    assert logp_batch.shape == (1,)
    assert phi_batch.shape == (1,)
    assert jnp.allclose(logp_single, logp_batch[0], atol=1e-5)
    assert jnp.allclose(phi_single, phi_batch[0], atol=1e-5)
