from abc import ABC, abstractmethod

import flax.nnx as nn
import jax
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
        self.spin_embeds = nn.Param(jax.random.normal(rngs(), (1, embed_dim)))
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

        spins = 1.0 - 2.0 * configuration
        token_features = spins[:, :, jnp.newaxis] * self.spin_embeds.get_value()

        x = token_features + self.pos_embeds(positions)
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
        x2 = jnp.pi * jax.nn.soft_sign(self.phase_out(nn.tanh(self.phase_head(x))))
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

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x


class SimpleEncoder(nn.Module):
    """Encode 1D chain spin configurations and append time t."""

    def __init__(self, seq_dim, embed_dim, rngs: nn.Rngs):
        self.spin_embeds = nn.Param(jax.random.normal(rngs(), (1, embed_dim)))
        self.pos_embeds = nn.Embed(seq_dim, embed_dim, rngs=rngs)
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

        positions = jnp.arange(self.seq_dim)[jnp.newaxis, :].repeat(
            configuration.shape[0], axis=0
        )

        spins = 1.0 - 2.0 * configuration
        token_features = spins[:, :, jnp.newaxis] * self.spin_embeds.get_value()

        x = token_features + self.pos_embeds(positions)

        # append the time to embedding vector
        t = jnp.full((batch_dim, self.seq_dim, 1), t)
        x = jnp.concatenate([x, t], axis=2)

        return x


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
        self.emb_dim += 1

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
        x2 = jnp.pi * jax.nn.soft_sign(self.phase_out(nn.tanh(self.phase_head(x))))
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
            k_cache, v_cache = cache
            k_cache = jax.lax.dynamic_update_slice_in_dim(k_cache, k, t_index, axis=1)
            v_cache = jax.lax.dynamic_update_slice_in_dim(v_cache, v, t_index, axis=1)
            new_cache = (k_cache, v_cache)
            
            attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k_cache) / jnp.sqrt(self.head_dim)
            
            k_indices = jnp.arange(k_cache.shape[1])
            mask = k_indices[None, None, None, :] > t_index
            attn_logits = jnp.where(mask, -1e9, attn_logits)
            
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v_cache)
            attn = attn.reshape(batch, seq, -1)
            out = self.out_proj(attn)
            return out, new_cache
        else:
            attn_logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(self.head_dim)
            mask = jnp.tril(jnp.ones((seq, seq)))[None, None, :, :]
            attn_logits = jnp.where(mask == 1, attn_logits, -1e9)
            
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
            attn = attn.reshape(batch, seq, -1)
            out = self.out_proj(attn)
            return out, None


class CausalBoxLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.feature_dim = feature_dim
        self.layernorm = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.layernorm2 = nn.LayerNorm(self.feature_dim, rngs=rngs)
        self.transformer = CausalTransformerLayer(
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
        x = x + attn_out
        x = x + jax.nn.tanh(self.ffn(self.layernorm2(x)))
        return x, new_cache


class AutoregressiveAmpModel(nn.Module):
    def __init__(self, N, Num_boxes, emb_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.N = N
        self.emb_dim = emb_dim
        self.spin_embeds = nn.Embed(3, emb_dim, rngs=rngs) # 0, 1, 2=SOS
        self.pos_embeds = nn.Embed(N, emb_dim, rngs=rngs)
        
        self.time_mlp1 = nn.Linear(1, emb_dim, rngs=rngs)
        self.time_mlp2 = nn.Linear(emb_dim, emb_dim, rngs=rngs)
        self.time_mlp3 = nn.Linear(emb_dim, emb_dim, rngs=rngs)
        
        self.layers = nn.data([])
        for _ in range(Num_boxes):
            self.layers.append(CausalBoxLayer(emb_dim, num_heads, head_dim, rngs=rngs))
            
        self.logits_out = nn.Linear(emb_dim, 2, rngs=rngs)

    def _time_features(self, t, batch_dim: int) -> jnp.ndarray:
        t_val = jnp.full((batch_dim, 1), t)
        t_hidden = jax.nn.gelu(self.time_mlp1(t_val))
        t_feat = jax.nn.gelu(self.time_mlp2(t_hidden))
        t_feat = jax.nn.gelu(self.time_mlp3(t_feat))
        return t_feat + t_hidden

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32, cache=None, t_index=None):
        configuration = jnp.asarray(configuration)
        batch_dim = configuration.shape[0]
        
        if cache is None:
            sos_tokens = jnp.full((batch_dim, 1), 2, dtype=jnp.int32)
            inputs = jnp.concatenate([sos_tokens, configuration[:, :-1]], axis=1)
            positions = jnp.arange(self.N)[jnp.newaxis, :].repeat(batch_dim, axis=0)
        else:
            inputs = jnp.where(t_index == 0, jnp.full((batch_dim, 1), 2, dtype=jnp.int32), configuration)
            positions = jnp.full((batch_dim, 1), t_index, dtype=jnp.int32)
            
        x = self.spin_embeds(inputs) + self.pos_embeds(positions)
        
        t_feat = self._time_features(t, batch_dim)[:, jnp.newaxis, :]
        x = x + t_feat
        
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, nc = layer(x, layer_cache, t_index)
            if nc is not None:
                new_caches.append(nc)
                
        logits = self.logits_out(x)
        return logits, tuple(new_caches) if cache is not None else None

    def init_cache(self, batch_size):
        caches = []
        for layer in self.layers:
            num_heads = layer.transformer.num_heads
            head_dim = layer.transformer.head_dim
            k_cache = jnp.zeros((batch_size, self.N, num_heads, head_dim), dtype=jnp.float32)
            v_cache = jnp.zeros((batch_size, self.N, num_heads, head_dim), dtype=jnp.float32)
            caches.append((k_cache, v_cache))
        return tuple(caches)


class AutoregressiveNQSModel(nn.Module):
    def __init__(self, N, Num_boxes, emb_dim, num_heads, head_dim, rngs: nn.Rngs):
        self.amp_model = AutoregressiveAmpModel(N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs)
        self.phase_model = tNQS(N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs)

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        configuration = jnp.asarray(configuration)
        is_single = configuration.ndim == 1
        if is_single:
            configuration = configuration[None, ...]

        logits, _ = self.amp_model(configuration, t, cache=None)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected_log_probs = jnp.take_along_axis(log_probs, configuration[..., None], axis=-1).squeeze(-1)
        logp = jnp.sum(selected_log_probs, axis=1)
        
        _, phi = self.phase_model(configuration, t)
        
        return logp, phi

class AutoregressiveNQS(Wavefunction):
    """Autoregressive Wavefunction using AR for amplitude and Static for phase."""

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
        
        self.model = AutoregressiveNQSModel(N, Num_boxes, emb_dim, num_heads, head_dim, rngs=rngs)

    def __call__(self, configuration: jnp.ndarray, t: jnp.float32):
        logp, phi = self.model(configuration, t)
        return self._squeeze_last_dim(logp), self._squeeze_last_dim(phi)

    @staticmethod
    def _squeeze_last_dim(x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1 and x.shape[-1] == 1:
            return jnp.squeeze(x, axis=-1)
        return x
