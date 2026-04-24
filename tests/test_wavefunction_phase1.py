import flax.nnx as nn
import jax.numpy as jnp

from src.wavefunction import tSpinNQS


def _make_model(N=6):
    return tSpinNQS(
        N=N,
        Num_boxes=2,
        emb_dim=16,
        num_heads=2,
        head_dim=8,
        rngs=nn.Rngs(0),
    )


def test_tspinnqs_holds_persistent_model_instance():
    wf = _make_model(N=6)

    model_id_before = id(wf.model)
    _ = wf(jnp.array([0, 1, 0, 1, 1, 0], dtype=jnp.int32), t=jnp.float32(0.2))
    model_id_after_first_call = id(wf.model)
    _ = wf(jnp.array([1, 0, 1, 0, 0, 1], dtype=jnp.int32), t=jnp.float32(0.3))
    model_id_after_second_call = id(wf.model)

    assert model_id_before == model_id_after_first_call == model_id_after_second_call


def test_call_output_shapes_for_single_and_batch_inputs():
    N = 5
    wf = _make_model(N=N)

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


def test_outputs_are_finite_for_valid_inputs():
    N = 7
    wf = _make_model(N=N)

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


def test_single_vs_singleton_batch_consistency():
    N = 5
    wf = _make_model(N=N)

    config = jnp.array([1, 0, 1, 1, 0], dtype=jnp.int32)
    config_batched = config[jnp.newaxis, :]
    t = jnp.float32(0.9)

    logp_single, phi_single = wf(config, t)
    logp_batch, phi_batch = wf(config_batched, t)

    assert logp_batch.shape == (1,)
    assert phi_batch.shape == (1,)
    assert jnp.allclose(logp_single, logp_batch[0], atol=1e-6)
    assert jnp.allclose(phi_single, phi_batch[0], atol=1e-6)
