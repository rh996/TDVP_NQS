from abc import ABC, abstractmethod

import flax.nnx as nn
import jax
import jax.numpy as jnp


def xsa_output(
    attn_out: jnp.ndarray, v_self: jnp.ndarray, eps: float = 1e-6
) -> jnp.ndarray:
    """Exclusive self-attention: remove the component parallel to the self value."""
    coeff = jnp.sum(attn_out * v_self, axis=-1, keepdims=True)
    coeff = coeff / (jnp.sum(v_self * v_self, axis=-1, keepdims=True) + eps)
    return attn_out - coeff * v_self


def odd_silu(x: jnp.ndarray) -> jnp.ndarray:
    """Odd SiLU variant: f(-x) = -f(x), preserving spin-flip equivariance."""
    return x * jax.nn.sigmoid(jnp.abs(x))


def rms_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Parameter-free RMS normalization for attention over residual streams."""
    return x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


class AttentionResiduals(nn.Module):
    """Attention over completed residual blocks and the current partial block."""

    def __init__(
        self,
        num_layers: int,
        feature_dim: int,
        rngs: nn.Rngs,
        use_even_logits: bool = False,
    ):
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.use_even_logits = use_even_logits
        self.w = nn.Param(
            0.02 * jax.random.normal(rngs(), (num_layers + 1, feature_dim))
        )

    def __call__(
        self,
        blocks: list[jnp.ndarray],
        partial_block: jnp.ndarray,
        layer_index: int,
    ) -> jnp.ndarray:
        V = jnp.stack(tuple(blocks + [partial_block]), axis=0)
        K = rms_norm(V)
        if self.use_even_logits:
            K = jnp.abs(K)
        logits = jnp.einsum("d,nbtd->nbt", self.w.get_value()[layer_index], K)
        alpha = nn.softmax(logits, axis=0)
        return jnp.einsum("nbt,nbtd->btd", alpha, V)


def attention_residual_output(
    residuals: AttentionResiduals, blocks: list[jnp.ndarray], partial: jnp.ndarray
) -> jnp.ndarray:
    """Final hidden state after the last residual-attention layer."""
    return residuals(blocks, partial, residuals.num_layers)


class TimeFeatureMap(nn.Module):
    """Fourier time features plus a learnable exponential quench feature."""

    def __init__(
        self,
        num_fourier_bands: int = 4,
        use_exp_decay: bool = True,
        rngs: nn.Rngs | None = None,
    ):
        if num_fourier_bands < 0:
            raise ValueError(f"num_fourier_bands must be >= 0, got {num_fourier_bands}")
        self.num_fourier_bands = num_fourier_bands
        self.use_exp_decay = use_exp_decay
        self.output_dim = 1 + 2 * num_fourier_bands + int(use_exp_decay)
        if use_exp_decay:
            init_raw_a = jnp.log(jnp.expm1(jnp.asarray(1.0, dtype=jnp.float32)))
            self.raw_decay_rate = nn.Param(init_raw_a)

    def decay_rate(self) -> jnp.ndarray:
        if not self.use_exp_decay:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return jax.nn.softplus(self.raw_decay_rate.get_value()) + 1e-6

    def __call__(self, t, batch_dim: int) -> jnp.ndarray:
        t_val = jnp.full((batch_dim, 1), t, dtype=jnp.float32)
        features = [t_val]
        if self.num_fourier_bands > 0:
            freqs = 2.0 ** jnp.arange(self.num_fourier_bands, dtype=t_val.dtype)
            angles = 2.0 * jnp.pi * t_val * freqs[jnp.newaxis, :]
            scale = freqs[jnp.newaxis, :]
            features.extend([jnp.sin(angles) / scale, jnp.cos(angles) / scale])
        if self.use_exp_decay:
            features.append(jnp.exp(-self.decay_rate() * t_val))
        return jnp.concatenate(features, axis=-1)


def apply_rope(x: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary position encoding to the largest even head subspace."""
    head_dim = x.shape[-1]
    rotary_dim = (head_dim // 2) * 2
    if rotary_dim == 0:
        return x

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    half_dim = rotary_dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (jnp.arange(half_dim, dtype=x.dtype) / jnp.asarray(half_dim, dtype=x.dtype))
    )
    angles = positions.astype(x.dtype)[..., None] * inv_freq

    while angles.ndim < x_rot.ndim - 1:
        angles = angles[..., None, :]

    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    x_pair = x_rot.reshape(*x_rot.shape[:-1], half_dim, 2)
    x_even = x_pair[..., 0]
    x_odd = x_pair[..., 1]
    rotated = jnp.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        axis=-1,
    ).reshape(x_rot.shape)
    return jnp.concatenate([rotated, x_pass], axis=-1)


class Wavefunction(ABC):
    """Abstract base class for variational wavefunctions."""

    def __init__(self):
        self.model: nn.Module = None

    @abstractmethod
    def __call__(self, configuration, t):
        return None

    def local_tdvp_estimator(self, configuration):
        return None


class Encoder(nn.Module):
    """Encode 1D chain spin configurations and time t."""

    def __init__(self, seq_dim, embed_dim, rngs: nn.Rngs):
        self.spin_embeds = nn.Param(jax.random.normal(rngs(), (1, embed_dim)))
        self.time_features = TimeFeatureMap(rngs=rngs)
        self.time_mlp1 = nn.Linear(
            self.time_features.output_dim, embed_dim, rngs=rngs, use_bias=False
        )
        self.time_mlp2 = nn.Linear(embed_dim, embed_dim, rngs=rngs, use_bias=False)
        self.seq_dim = seq_dim

    def _time_gate(self, t, batch_dim: int) -> jnp.ndarray:
        time_features = self.time_features(t, batch_dim)
        t_feat = odd_silu(self.time_mlp1(time_features))
        t_feat = odd_silu(self.time_mlp2(t_feat))
        return 1.0 + t_feat

    def __call__(self, configuration: jnp.ndarray, t):
        configuration = configuration.astype(jnp.int32)

        if len(configuration.shape) != 2:
            configuration = jnp.expand_dims(configuration, axis=0)

        if configuration.shape[1] != self.seq_dim:
            raise ValueError(
                f"Expected sequence length {self.seq_dim}, got {configuration.shape[1]}"
            )

        batch_dim = configuration.shape[0]

        spins = 1.0 - 2.0 * configuration
        token_features = spins[:, :, jnp.newaxis] * self.spin_embeds.get_value()

        time_gate = self._time_gate(t, batch_dim)[:, jnp.newaxis, :]
        return token_features * time_gate


class TransformerLayer(nn.Module):
    """Transformer layer for the wavefunction."""

    def __init__(self, feature_dim, num_heads, head_dim, out_dim, rngs: nn.Rngs):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim
        self.q_proj = nn.Linear(
            feature_dim,
            self.proj_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.k_proj = nn.Linear(
            feature_dim,
            self.proj_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.v_proj = nn.Linear(
            feature_dim,
            self.proj_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.out_proj = nn.Linear(
            self.proj_dim,
            out_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )

    def __call__(self, x: jnp.ndarray, positions: jnp.ndarray | None = None):
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        if positions is None:
            positions = jnp.arange(seq)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k)
        attn_logits = attn_logits / jnp.sqrt(self.head_dim)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn = xsa_output(attn, v)
        attn = attn.reshape(batch, seq, -1)

        out = self.out_proj(attn)
        return out


class BoxLayer(nn.Module):
    """"""

    def __init__(self, feature_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layernorm = nn.LayerNorm(self.feature_dim, rngs=rngs, use_bias=False)
        self.layernorm2 = nn.LayerNorm(self.feature_dim, rngs=rngs, use_bias=False)
        self.transformer = TransformerLayer(
            self.feature_dim, self.num_heads, self.head_dim, self.feature_dim, rngs
        )
        self.ffn = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        """Transformer block"""
        return x + self.residual_delta(x)

    def residual_delta(self, x: jnp.ndarray) -> jnp.ndarray:
        x_norm = self.layernorm(x)
        attn_out = self.transformer(x_norm)
        y = x + attn_out
        return attn_out + odd_silu(self.ffn(self.layernorm2(y)))


class OldEncoder(nn.Module):
    """Original encoder with additive position and raw-time MLP features."""

    def __init__(self, seq_dim, embed_dim, rngs: nn.Rngs):
        self.spin_embeds = nn.Param(jax.random.normal(rngs(), (1, embed_dim)))
        self.pos_embeds = nn.Embed(seq_dim, embed_dim, rngs=rngs)
        self.time_mlp1 = nn.Linear(1, embed_dim, rngs=rngs)
        self.time_mlp2 = nn.Linear(embed_dim, embed_dim, rngs=rngs)
        self.time_mlp3 = nn.Linear(embed_dim, embed_dim, rngs=rngs)
        self.seq_dim = seq_dim

    def __call__(self, configuration: jnp.ndarray, t):
        configuration = configuration.astype(jnp.int32)

        if len(configuration.shape) != 2:
            configuration = jnp.expand_dims(configuration, axis=0)

        if configuration.shape[1] != self.seq_dim:
            raise ValueError(
                f"Expected sequence length {self.seq_dim}, got {configuration.shape[1]}"
            )

        batch_dim = configuration.shape[0]
        t_val = jnp.full((batch_dim, 1), t, dtype=jnp.float32)
        t_feat = nn.gelu(self.time_mlp1(t_val))
        t_feat = nn.gelu(self.time_mlp2(t_feat))
        t_feat = nn.gelu(self.time_mlp3(t_feat))[:, jnp.newaxis, :]

        positions = jnp.arange(self.seq_dim)[jnp.newaxis, :].repeat(batch_dim, axis=0)
        spins = 1.0 - 2.0 * configuration
        token_features = spins[:, :, jnp.newaxis] * self.spin_embeds.get_value()
        return token_features + self.pos_embeds(positions) + t_feat


class OldTransformerLayer(nn.Module):
    """Original full self-attention layer, with XSA retained."""

    def __init__(self, feature_dim, num_heads, head_dim, out_dim, rngs: nn.Rngs):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim
        self.q_proj = nn.Linear(
            feature_dim,
            self.proj_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.k_proj = nn.Linear(
            feature_dim,
            self.proj_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.v_proj = nn.Linear(
            feature_dim,
            self.proj_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.out_proj = nn.Linear(
            self.proj_dim,
            out_dim,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
        )

    def __call__(self, x: jnp.ndarray):
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)

        attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k)
        attn_logits = attn_logits / jnp.sqrt(self.head_dim)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn = xsa_output(attn, v)
        attn = attn.reshape(batch, seq, -1)
        return self.out_proj(attn)


class OldBoxLayer(nn.Module):
    """Original Transformer block with standard biased LayerNorm/MLP."""

    def __init__(self, feature_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layernorm = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.layernorm2 = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.transformer = OldTransformerLayer(
            self.feature_dim, self.num_heads, self.head_dim, self.feature_dim, rngs
        )
        self.ffn = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            kernel_init=nn.initializers.kaiming_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        return x + self.residual_delta(x)

    def residual_delta(self, x: jnp.ndarray) -> jnp.ndarray:
        x_norm = self.layernorm(x)
        attn_out = self.transformer(x_norm)
        y = x + attn_out
        return attn_out + nn.gelu(self.ffn(self.layernorm2(y)))


class tNQS(nn.Module):
    """Original t-Spin NQS wavefunction."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.encoder = OldEncoder(self.N, emb_dim, rngs=rngs)
        self.layers = nn.data([])

        for _ in range(Num_boxes):
            self.layers.append(
                OldBoxLayer(self.emb_dim, self.num_heads, self.head_dim, rngs=rngs)
            )
        self.attn_residuals = AttentionResiduals(Num_boxes, self.emb_dim, rngs=rngs)

        self.head_hidden_dim = self.emb_dim
        self.log_amp_head = nn.Linear(
            self.emb_dim,
            self.head_hidden_dim,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.log_amp_out = nn.Linear(
            self.head_hidden_dim,
            1,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.phase_head = nn.Linear(
            self.emb_dim,
            self.head_hidden_dim,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.phase_out = nn.Linear(
            self.head_hidden_dim,
            1,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        x = self.encoder(configuration, t)
        blocks = [x]
        partial = jnp.zeros_like(x)
        for i, layer in enumerate(self.layers):
            h = self.attn_residuals(blocks, partial, i)
            partial = partial + layer.residual_delta(h)
            blocks.append(partial)
            partial = jnp.zeros_like(partial)
        x = attention_residual_output(self.attn_residuals, blocks, partial)
        x = jnp.mean(x, axis=1)
        x1 = self.log_amp_out(nn.tanh(self.log_amp_head(x)))
        x2 = jnp.pi * jax.nn.soft_sign(self.phase_out(nn.tanh(self.phase_head(x))))
        return x1, x2


class tNQS_Z2(nn.Module):
    """Z2-symmetric t-Spin NQS wavefunction."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.encoder = Encoder(self.N, emb_dim, rngs=rngs)

        self.layers = nn.data([])

        for _ in range(Num_boxes):
            self.layers.append(
                BoxLayer(self.emb_dim, self.num_heads, self.head_dim, rngs=rngs)
            )
        self.attn_residuals = AttentionResiduals(
            Num_boxes, self.emb_dim, rngs=rngs, use_even_logits=True
        )

        self.head_hidden_dim = self.emb_dim
        self.log_amp_head = nn.Linear(
            self.emb_dim,
            self.head_hidden_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.log_amp_out = nn.Linear(
            self.head_hidden_dim,
            1,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.phase_head = nn.Linear(
            self.emb_dim,
            self.head_hidden_dim,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.phase_out = nn.Linear(
            self.head_hidden_dim,
            1,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )

    def _even_amp_features(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[1] <= 1:
            return jnp.mean(x * x, axis=1)
        return jnp.mean(x[:, :-1, :] * x[:, 1:, :], axis=1)

    def _log_amp_from_features(self, x: jnp.ndarray) -> jnp.ndarray:
        pooled = self._even_amp_features(x)
        return self.log_amp_out(odd_silu(self.log_amp_head(pooled)))

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        x = self.encoder(configuration, t)
        blocks = [x]
        partial = jnp.zeros_like(x)
        for i, layer in enumerate(self.layers):
            h = self.attn_residuals(blocks, partial, i)
            partial = partial + layer.residual_delta(h)
            blocks.append(partial)
            partial = jnp.zeros_like(partial)
        x = attention_residual_output(self.attn_residuals, blocks, partial)

        phase_features = jnp.mean(x, axis=1)
        x1 = self._log_amp_from_features(x)
        x2 = jnp.pi * jax.nn.soft_sign(
            self.phase_out(odd_silu(self.phase_head(phase_features)))
        )
        return x1, x2


class tSpinNQS(Wavefunction):
    """Persistent wrapper around the original tNQS exposing log_prob/phase API."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rngs = rngs
        self.model = tNQS(
            N=self.N,
            Num_boxes=self.Num_boxes,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rngs=self.rngs,
        )

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x


class tSpinNQS_Z2(Wavefunction):
    """Persistent wrapper around the Z2-symmetric tNQS."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rngs = rngs
        self.model = tNQS_Z2(
            N=self.N,
            Num_boxes=self.Num_boxes,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rngs=self.rngs,
        )

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x


class SimpleEncoder(nn.Module):
    """Encode 1D chain spin configurations and append time t."""

    def __init__(self, seq_dim, embed_dim, rngs: nn.Rngs):
        self.spin_embeds = nn.Param(jax.random.normal(rngs(), (1, embed_dim)))
        self.time_features = TimeFeatureMap(rngs=rngs)
        self.time_mlp1 = nn.Linear(
            self.time_features.output_dim, embed_dim, rngs=rngs, use_bias=False
        )
        self.time_mlp2 = nn.Linear(embed_dim, embed_dim, rngs=rngs, use_bias=False)
        self.seq_dim = seq_dim

    def _time_gate(self, t, batch_dim: int) -> jnp.ndarray:
        time_features = self.time_features(t, batch_dim)
        t_feat = odd_silu(self.time_mlp1(time_features))
        t_feat = odd_silu(self.time_mlp2(t_feat))
        return 1.0 + t_feat

    def __call__(self, configuration: jnp.ndarray, t):
        configuration = configuration.astype(jnp.int32)

        if len(configuration.shape) != 2:
            configuration = jnp.expand_dims(configuration, axis=0)

        if configuration.shape[1] != self.seq_dim:
            raise ValueError(
                f"Expected sequence length {self.seq_dim}, got {configuration.shape[1]}"
            )

        batch_dim = configuration.shape[0]

        spins = 1.0 - 2.0 * configuration
        token_features = spins[:, :, jnp.newaxis] * self.spin_embeds.get_value()

        time_gate = self._time_gate(t, batch_dim)[:, jnp.newaxis, :]
        return token_features * time_gate


class SimpleTNQS(nn.Module):
    """Simple t-Spin NQS wavefunction."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.encoder = SimpleEncoder(self.N, emb_dim, rngs=rngs)

        self.layers = nn.data([])

        for _ in range(Num_boxes):
            self.layers.append(
                BoxLayer(self.emb_dim, self.num_heads, self.head_dim, rngs=rngs)
            )
        self.attn_residuals = AttentionResiduals(
            Num_boxes, self.emb_dim, rngs=rngs, use_even_logits=True
        )

        self.head_hidden_dim = self.emb_dim
        self.log_amp_head = nn.Linear(
            self.emb_dim,
            self.head_hidden_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.log_amp_out = nn.Linear(
            self.head_hidden_dim,
            1,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.phase_head = nn.Linear(
            self.emb_dim,
            self.head_hidden_dim,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        self.phase_out = nn.Linear(
            self.head_hidden_dim,
            1,
            use_bias=False,
            rngs=rngs,
            kernel_init=nn.initializers.kaiming_normal(),
        )

    def _even_amp_features(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[1] <= 1:
            return jnp.mean(x * x, axis=1)
        return jnp.mean(x[:, :-1, :] * x[:, 1:, :], axis=1)

    def _log_amp_from_features(self, x: jnp.ndarray) -> jnp.ndarray:
        pooled = self._even_amp_features(x)
        return self.log_amp_out(odd_silu(self.log_amp_head(pooled)))

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        x = self.encoder(configuration, t)
        blocks = [x]
        partial = jnp.zeros_like(x)
        for i, layer in enumerate(self.layers):
            h = self.attn_residuals(blocks, partial, i)
            partial = partial + layer.residual_delta(h)
            blocks.append(partial)
            partial = jnp.zeros_like(partial)
        x = attention_residual_output(self.attn_residuals, blocks, partial)

        phase_features = jnp.mean(x, axis=1)
        x1 = self._log_amp_from_features(x)
        x2 = jnp.pi * jax.nn.soft_sign(
            self.phase_out(odd_silu(self.phase_head(phase_features)))
        )
        return x1, x2


class SimpleSpinNQS(Wavefunction):
    """Persistent wrapper around SimpleTNQS exposing API."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rngs = rngs
        self.model = SimpleTNQS(
            N=self.N,
            Num_boxes=self.Num_boxes,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rngs=self.rngs,
        )

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x


class CausalTransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, head_dim, out_dim, rngs: nn.Rngs):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim
        self.q_proj = nn.Linear(feature_dim, self.proj_dim, rngs=rngs, use_bias=False)
        self.k_proj = nn.Linear(feature_dim, self.proj_dim, rngs=rngs, use_bias=False)
        self.v_proj = nn.Linear(feature_dim, self.proj_dim, rngs=rngs, use_bias=False)
        self.out_proj = nn.Linear(self.proj_dim, out_dim, rngs=rngs, use_bias=False)

    def __call__(self, x: jnp.ndarray, cache=None, t_index=None):
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)

        if cache is not None:
            q = apply_rope(q, jnp.full((seq,), t_index))
            k = apply_rope(k, jnp.full((seq,), t_index))
            k_cache, v_cache = cache
            k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, t_index, axis=1)
            v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, t_index, axis=1)
            new_cache = (k_cache, v_cache)

            attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k_cache) / jnp.sqrt(
                self.head_dim
            )

            k_indices = jnp.arange(k_cache.shape[1])
            mask = k_indices[None, None, None, :] > t_index
            attn_logits = jnp.where(mask, -1e9, attn_logits)

            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v_cache)
            attn = xsa_output(attn, v)
            attn = attn.reshape(batch, seq, -1)
            out = self.out_proj(attn)
            return out, new_cache
        else:
            positions = jnp.arange(seq)
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)
            attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
            mask = jnp.tril(jnp.ones((seq, seq)))[None, None, :, :]
            attn_logits = jnp.where(mask == 1, attn_logits, -1e9)

            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
            attn = xsa_output(attn, v)
            attn = attn.reshape(batch, seq, -1)
            out = self.out_proj(attn)
            return out, None


class CausalBoxLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.feature_dim = feature_dim
        self.layernorm = nn.LayerNorm(self.feature_dim, rngs=rngs, use_bias=False)
        self.layernorm2 = nn.LayerNorm(self.feature_dim, rngs=rngs, use_bias=False)
        self.transformer = CausalTransformerLayer(
            self.feature_dim, num_heads, head_dim, self.feature_dim, rngs
        )
        self.ffn = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, cache=None, t_index=None):
        x_norm = self.layernorm(x)
        attn_out, new_cache = self.transformer(x_norm, cache, t_index)
        y = x + attn_out
        return y + odd_silu(self.ffn(self.layernorm2(y))), new_cache

    def residual_delta(self, x: jnp.ndarray, cache=None, t_index=None):
        x_norm = self.layernorm(x)
        attn_out, new_cache = self.transformer(x_norm, cache, t_index)
        y = x + attn_out
        return attn_out + odd_silu(self.ffn(self.layernorm2(y))), new_cache


class OldCausalTransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, head_dim, out_dim, rngs: nn.Rngs):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim
        self.q_proj = nn.Linear(feature_dim, self.proj_dim, rngs=rngs, use_bias=False)
        self.k_proj = nn.Linear(feature_dim, self.proj_dim, rngs=rngs, use_bias=False)
        self.v_proj = nn.Linear(feature_dim, self.proj_dim, rngs=rngs, use_bias=False)
        self.out_proj = nn.Linear(self.proj_dim, out_dim, rngs=rngs, use_bias=False)

    def __call__(self, x: jnp.ndarray, cache=None, t_index=None):
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)

        if cache is not None:
            k_cache, v_cache = cache
            k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, t_index, axis=1)
            v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, t_index, axis=1)
            new_cache = (k_cache, v_cache)
            attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k_cache) / jnp.sqrt(
                self.head_dim
            )
            k_indices = jnp.arange(k_cache.shape[1])
            mask = k_indices[None, None, None, :] > t_index
            attn_logits = jnp.where(mask, -1e9, attn_logits)
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v_cache)
            attn = xsa_output(attn, v)
            attn = attn.reshape(batch, seq, -1)
            return self.out_proj(attn), new_cache

        attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((seq, seq)))[None, None, :, :]
        attn_logits = jnp.where(mask == 1, attn_logits, -1e9)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn = xsa_output(attn, v)
        attn = attn.reshape(batch, seq, -1)
        return self.out_proj(attn), None


class OldCausalBoxLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.feature_dim = feature_dim
        self.layernorm = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.layernorm2 = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.transformer = OldCausalTransformerLayer(
            self.feature_dim, num_heads, head_dim, self.feature_dim, rngs
        )
        self.ffn = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            kernel_init=nn.initializers.kaiming_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, cache=None, t_index=None):
        x_norm = self.layernorm(x)
        attn_out, new_cache = self.transformer(x_norm, cache, t_index)
        y = x + attn_out
        return y + jax.nn.gelu(self.ffn(self.layernorm2(y))), new_cache

    def residual_delta(self, x: jnp.ndarray, cache=None, t_index=None):
        x_norm = self.layernorm(x)
        attn_out, new_cache = self.transformer(x_norm, cache, t_index)
        y = x + attn_out
        return attn_out + jax.nn.gelu(self.ffn(self.layernorm2(y))), new_cache


class AutoregressiveAmpModel(nn.Module):
    """Original autoregressive amplitude model without Z2 constraints."""

    def __init__(self, N, Num_boxes, emb_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.N = N
        self.emb_dim = emb_dim
        self.spin_embeds = nn.Embed(3, emb_dim, rngs=rngs)  # 0, 1, 2=SOS
        self.pos_embeds = nn.Embed(N, emb_dim, rngs=rngs)

        self.time_features = TimeFeatureMap(rngs=rngs)
        self.time_mlp1 = nn.Linear(self.time_features.output_dim, emb_dim, rngs=rngs)
        self.time_mlp2 = nn.Linear(emb_dim, emb_dim, rngs=rngs)
        self.time_mlp3 = nn.Linear(emb_dim, emb_dim, rngs=rngs)

        self.layers = nn.data([])
        for _ in range(Num_boxes):
            self.layers.append(
                OldCausalBoxLayer(emb_dim, num_heads, head_dim, rngs=rngs)
            )
        self.attn_residuals = AttentionResiduals(Num_boxes, emb_dim, rngs=rngs)

        self.logits_out = nn.Linear(emb_dim, 2, rngs=rngs)

    def _time_features(self, t, batch_dim: int) -> jnp.ndarray:
        time_features = self.time_features(t, batch_dim)
        t_hidden = jax.nn.gelu(self.time_mlp1(time_features))
        t_feat = jax.nn.gelu(self.time_mlp2(t_hidden))
        t_feat = jax.nn.gelu(self.time_mlp3(t_feat))
        return t_feat + t_hidden

    def __call__(
        self, configuration: jnp.ndarray, t: jnp.float32, cache=None, t_index=None
    ):
        configuration = jnp.asarray(configuration)
        batch_dim = configuration.shape[0]

        if cache is None:
            sos_tokens = jnp.full((batch_dim, 1), 2, dtype=jnp.int32)
            inputs = jnp.concatenate([sos_tokens, configuration[:, :-1]], axis=1)
            positions = jnp.arange(self.N)[jnp.newaxis, :].repeat(batch_dim, axis=0)
        else:
            inputs = jnp.where(
                t_index == 0,
                jnp.full((batch_dim, 1), 2, dtype=jnp.int32),
                configuration,
            )
            positions = jnp.full((batch_dim, 1), t_index, dtype=jnp.int32)

        x = self.spin_embeds(inputs) + self.pos_embeds(positions)
        x = x + self._time_features(t, batch_dim)[:, jnp.newaxis, :]

        blocks = [x]
        partial = jnp.zeros_like(x)
        new_caches = []
        for i, layer in enumerate(self.layers):
            h = self.attn_residuals(blocks, partial, i)
            layer_cache = cache[i] if cache is not None else None
            delta, nc = layer.residual_delta(h, layer_cache, t_index)
            partial = partial + delta
            blocks.append(partial)
            partial = jnp.zeros_like(partial)
            if nc is not None:
                new_caches.append(nc)
        x = attention_residual_output(self.attn_residuals, blocks, partial)

        logits = self.logits_out(x)
        return logits, tuple(new_caches) if cache is not None else None

    def init_cache(self, batch_size):
        caches = []
        for layer in self.layers:
            num_heads = layer.transformer.num_heads
            head_dim = layer.transformer.head_dim
            k_cache = jnp.zeros(
                (batch_size, self.N, num_heads, head_dim), dtype=jnp.float32
            )
            v_cache = jnp.zeros(
                (batch_size, self.N, num_heads, head_dim), dtype=jnp.float32
            )
            caches.append((k_cache, v_cache))
        return tuple(caches)


class AutoregressiveAmpModel_Z2(nn.Module):
    """Z2-constrained autoregressive amplitude model."""

    def __init__(self, N, Num_boxes, emb_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.N = N
        self.emb_dim = emb_dim
        self.spin_embeds = nn.Param(jax.random.normal(rngs(), (1, emb_dim)))

        self.time_features = TimeFeatureMap(rngs=rngs)
        self.time_mlp1 = nn.Linear(
            self.time_features.output_dim, emb_dim, rngs=rngs, use_bias=False
        )
        self.time_mlp2 = nn.Linear(emb_dim, emb_dim, rngs=rngs, use_bias=False)
        self.time_mlp3 = nn.Linear(emb_dim, emb_dim, rngs=rngs, use_bias=False)

        self.layers = nn.data([])
        for _ in range(Num_boxes):
            self.layers.append(CausalBoxLayer(emb_dim, num_heads, head_dim, rngs=rngs))
        self.attn_residuals = AttentionResiduals(
            Num_boxes, emb_dim, rngs=rngs, use_even_logits=True
        )

        self.logits_score = nn.Linear(emb_dim, 1, rngs=rngs, use_bias=False)

    def _time_gate(self, t, batch_dim: int) -> jnp.ndarray:
        time_features = self.time_features(t, batch_dim)
        t_feat = odd_silu(self.time_mlp1(time_features))
        t_feat = odd_silu(self.time_mlp2(t_feat))
        t_feat = odd_silu(self.time_mlp3(t_feat))
        return 1.0 + t_feat

    def _token_features(self, inputs: jnp.ndarray) -> jnp.ndarray:
        spins = 1.0 - 2.0 * inputs
        token_features = spins[:, :, jnp.newaxis] * self.spin_embeds.get_value()
        return jnp.where(inputs[:, :, jnp.newaxis] == 2, 0.0, token_features)

    def __call__(
        self, configuration: jnp.ndarray, t: jnp.float32, cache=None, t_index=None
    ):
        configuration = jnp.asarray(configuration)
        batch_dim = configuration.shape[0]

        if cache is None:
            sos_tokens = jnp.full((batch_dim, 1), 2, dtype=jnp.int32)
            inputs = jnp.concatenate([sos_tokens, configuration[:, :-1]], axis=1)
        else:
            inputs = jnp.where(
                t_index == 0,
                jnp.full((batch_dim, 1), 2, dtype=jnp.int32),
                configuration,
            )

        x = self._token_features(inputs)
        time_gate = self._time_gate(t, batch_dim)[:, jnp.newaxis, :]
        x = x * time_gate

        blocks = [x]
        partial = jnp.zeros_like(x)
        new_caches = []
        for i, layer in enumerate(self.layers):
            h = self.attn_residuals(blocks, partial, i)
            layer_cache = cache[i] if cache is not None else None
            delta, nc = layer.residual_delta(h, layer_cache, t_index)
            partial = partial + delta
            blocks.append(partial)
            partial = jnp.zeros_like(partial)
            if nc is not None:
                new_caches.append(nc)
        x = attention_residual_output(self.attn_residuals, blocks, partial)

        score = self.logits_score(x)
        logits = jnp.concatenate([score, -score], axis=-1)
        return logits, tuple(new_caches) if cache is not None else None

    def init_cache(self, batch_size):
        caches = []
        for layer in self.layers:
            num_heads = layer.transformer.num_heads
            head_dim = layer.transformer.head_dim
            k_cache = jnp.zeros(
                (batch_size, self.N, num_heads, head_dim), dtype=jnp.float32
            )
            v_cache = jnp.zeros(
                (batch_size, self.N, num_heads, head_dim), dtype=jnp.float32
            )
            caches.append((k_cache, v_cache))
        return tuple(caches)


class AutoregressiveNQSModel(nn.Module):
    def __init__(self, N, Num_boxes, emb_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.amp_model = AutoregressiveAmpModel(
            N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs
        )
        self.phase_model = tNQS(N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs)

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        configuration = jnp.asarray(configuration)
        is_single = configuration.ndim == 1
        if is_single:
            configuration = configuration[None, ...]

        logits, _ = self.amp_model(configuration, t, cache=None)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected_log_probs = jnp.take_along_axis(
            log_probs, configuration[..., None], axis=-1
        ).squeeze(-1)
        logp = jnp.sum(selected_log_probs, axis=1)

        _, phi = self.phase_model(configuration, t)

        return logp, phi


class AutoregressiveNQSModel_Z2(nn.Module):
    def __init__(self, N, Num_boxes, emb_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.amp_model = AutoregressiveAmpModel_Z2(
            N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs
        )
        self.phase_model = tNQS(N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs)

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        configuration = jnp.asarray(configuration)
        is_single = configuration.ndim == 1
        if is_single:
            configuration = configuration[None, ...]

        logits, _ = self.amp_model(configuration, t, cache=None)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected_log_probs = jnp.take_along_axis(
            log_probs, configuration[..., None], axis=-1
        ).squeeze(-1)
        logp = jnp.sum(selected_log_probs, axis=1)

        _, phi = self.phase_model(configuration, t)

        return logp, phi


class AutoregressiveNQS(Wavefunction):
    """Original autoregressive wavefunction without Z2 constraints."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rngs = rngs

        self.model = AutoregressiveNQSModel(
            N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs
        )

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x


class AutoregressiveNQS_Z2(Wavefunction):
    """Z2-constrained autoregressive wavefunction."""

    def __init__(
        self,
        N: int,
        Num_boxes: int,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        rngs: nn.Rngs,
    ):
        self.N = N
        self.Num_boxes = Num_boxes
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rngs = rngs

        self.model = AutoregressiveNQSModel_Z2(
            N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs
        )

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x
