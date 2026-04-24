from abc import ABC, abstractmethod

import flax.nnx as nn
import jax.numpy as jnp


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
        self.vocab = 2
        self.token_embeds = nn.Embed(self.vocab, embed_dim, rngs=rngs)
        self.pos_embeds = nn.Embed(seq_dim, embed_dim, rngs=rngs)
        # Time MLP: (1, D) -> GeLU -> (D, D) -> GeLU
        self.time_mlp1 = nn.Linear(1, embed_dim, rngs=rngs)
        self.time_mlp2 = nn.Linear(embed_dim, embed_dim, rngs=rngs)
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

        # Time feature vector: (B, 1) -> (B, D)
        t_val = jnp.full((batch_dim, 1), t)
        t_feat = nn.gelu(self.time_mlp1(t_val))
        t_feat = nn.gelu(self.time_mlp2(t_feat))
        
        # Reshape for broadcasting: (B, 1, D)
        t_feat = t_feat[:, jnp.newaxis, :]

        positions = jnp.arange(self.seq_dim)[jnp.newaxis, :].repeat(
            configuration.shape[0], axis=0
        )

        x = self.token_embeds(configuration) + self.pos_embeds(positions)
        # Additive time feature
        x = x + t_feat

        return x


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

    def __call__(self, x: jnp.ndarray):
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)

        attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k)
        attn_logits = attn_logits / jnp.sqrt(self.head_dim)
        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn = attn.reshape(batch, seq, -1)

        out = self.out_proj(attn)
        return out


class BoxLayer(nn.Module):
    """"""

    def __init__(self, feature_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layernorm = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.layernorm2 = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.transformer = TransformerLayer(
            self.feature_dim, self.num_heads, self.head_dim, self.feature_dim, rngs
        )
        self.ffn = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            kernel_init=nn.initializers.kaiming_normal(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray):
        """Transformer block"""
        x = self.layernorm(x)
        x = x + self.transformer(x)
        x = self.layernorm2(x)
        x = x + nn.tanh(self.ffn(x))

        return x


class tNQS(nn.Module):
    """t-Spin NQS wavefunction."""

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
        for layer in self.layers:
            x = layer(x)
        x = jnp.mean(x, axis=1)
        x1 = self.log_amp_out(nn.tanh(self.log_amp_head(x)))
        x2 = self.phase_out(nn.tanh(self.phase_head(x)))
        return x1, x2


class tSpinNQS(Wavefunction):
    """Persistent wrapper around tNQS exposing log_prob/phase API."""

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

    def log_prob(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, _ = self(configuration, t)
        return logp

    def phase(self, configuration: jnp.ndarray, t: jnp.float32):
        _, phi = self(configuration, t)
        return phi

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x
