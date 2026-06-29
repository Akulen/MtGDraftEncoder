import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from models._types import MLPLayer, ModelLayer

class PackAttention(eqx.Module):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    W_o: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    num_heads: int=eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        d_model: int=64,
        num_heads: int=2,
        dropout_p: float=0.1
    ):
        key_q, key_k, key_v, key_o = jax.random.split(key, 4)
        self.num_heads = num_heads
        d_qk = d_model // num_heads
        d_vo = d_model // num_heads
        self.W_q = eqx.nn.Linear(
            in_features=d_model,
            out_features=num_heads*d_qk,
            key=key_q
        )
        self.W_k = eqx.nn.Linear(
            in_features=d_model,
            out_features=num_heads*d_qk,
            key=key_k
        )
        self.W_v = eqx.nn.Linear(
            in_features=d_model,
            out_features=num_heads*d_vo,
            key=key_v
        )
        self.W_o = eqx.nn.Linear(
            in_features=num_heads*d_vo,
            out_features=d_model,
            key=key_o
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        key: PRNGKeyArray,
        query: Float[Array, "45 d_model"],
        key_: Float[Array, "45 15 d_model"],
        value: Float[Array, "45 15 d_model"],
        mask: Optional[Bool[Array, "45 15"]]=None,
        inference: bool=False
    ) -> Float[Array, "45 d_model"]:
        d_model = query.shape[-1]
        assert(query.shape == (45, d_model))
        assert(key_.shape == (45, 15, d_model))
        assert(value.shape == (45, 15, d_model))
        Q = jax.vmap(self.W_q)(query).reshape((45, self.num_heads, -1))
        K = jax.vmap(jax.vmap(self.W_k))(
            key_
        ).reshape((45, 15, self.num_heads, -1))
        V = jax.vmap(jax.vmap(self.W_v))(
            value
        ).reshape((45, 15, self.num_heads, -1))

        logits = jnp.einsum("qhd,qkhd->hqk", Q, K) / jnp.sqrt(Q.shape[-1])
        if mask is not None:
            assert(mask.shape == logits.shape[1:])
            logits = jnp.where(
                jnp.tile(mask, (self.num_heads, 1, 1)),
                logits,
                jnp.finfo(logits.dtype).min
            )
        weights = jax.nn.softmax(logits, axis=-1)
        weights = self.dropout(weights, inference=inference, key=key)
        attn = jnp.einsum("hqk,qkhd->qhd", weights, V)
        attn = attn.reshape((45, -1))
        attn = jax.vmap(self.W_o)(attn)
        return attn

class MultiheadAttentionLayer(ModelLayer):
    attention: eqx.nn.MultiheadAttention

    def __init__(self,
        key: PRNGKeyArray,
        num_heads: int=2,
        d_model: int=64,
        dropout_p: float=0.1,
        nn: Optional[eqx.nn.MultiheadAttention]=None
    ):
        if nn is None:
            self.attention = eqx.nn.MultiheadAttention(
                num_heads=num_heads,
                query_size=d_model,
                use_query_bias=True,
                use_key_bias=True,
                use_value_bias=True,
                use_output_bias=True,
                dropout_p=dropout_p,
                key=key
            )
        else:
            self.attention = nn

    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, "q_len d_model"],
        *args,
        inference: bool=False,
        mask: Optional[Bool[Array, "q_len kv_len"]]=None,
        process_heads: Optional[Callable]=None,
        key_: Optional[Float[Array, "kv_len d_model"]]=None,
        value: Optional[Float[Array, "kv_len d_model"]]=None,
        **kwargs
    ) -> Float[Array, "q_len d_model"]:
        del args, kwargs
        if key_ is None:
            key_ = x
        if value is None:
            value = x
        return self.attention(
            query=x,
            key_=key_,
            value=value,
            mask=mask,
            inference=inference,
            process_heads=process_heads,
            key=key
        )

class AlternatingResidualAttention(eqx.Module):
    lns_h: List[eqx.nn.LayerNorm]
    lns_s: List[eqx.nn.LayerNorm]
    att_h: MultiheadAttentionLayer
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    mlp_h: MLPLayer
    att_s: PackAttention
    mlp_s: MLPLayer
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        ff_depth: int=1,
        ff_activation=jax.nn.relu
    ):
        key_h, key_s = jax.random.split(key)
        key_att, key_mlp = jax.random.split(key_h)
        self.att_h = MultiheadAttentionLayer(
            key=key_att,
            num_heads=num_heads,
            d_model=d_model,
            dropout_p=dropout_p
        )
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=d_model // num_heads
        )
        self.mlp_h = MLPLayer(
            in_size=d_model,
            out_size=d_model,
            width_size=d_model,
            depth=ff_depth,
            activation=ff_activation,
            key=key_mlp
        )
        key_att, key_mlp = jax.random.split(key_s)
        self.att_s = PackAttention(
            key=key_att,
            d_model=d_model,
            num_heads=num_heads,
            dropout_p=dropout_p
        )
        self.mlp_s = MLPLayer(
            in_size=d_model,
            out_size=d_model,
            width_size=d_model,
            depth=ff_depth,
            activation=ff_activation,
            key=key_mlp
        )
        self.lns_h = [
            eqx.nn.LayerNorm(d_model) for _ in range(2)
        ]
        self.lns_s = [
            eqx.nn.LayerNorm(d_model) for _ in range(2)
        ]
        self.dropout = eqx.nn.Dropout(dropout_p)

    def process_heads(
        self,
        query_heads: Float[Array, "l_q num_heads qk_size"],
        key_heads: Float[Array, "l_kv num_heads qk_size"],
        value_heads: Float[Array, "l_kv num_heads vo_size"]
    ) -> Tuple[
        Float[Array, "l_q num_heads qk_size"],
        Float[Array, "l_kv num_heads qk_size"],
        Float[Array, "l_kv num_heads vo_size"]
    ]:
        query_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(
            query_heads
        )
        key_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(
            key_heads
        )
        return query_heads, key_heads, value_heads

    def __call__(self,
        key: PRNGKeyArray,
        h: Float[Array, "45 15 d_model"],
        s: Float[Array, "45 d_model"],
        pack_mask: Bool[Array, "45 15"],
        picks_pos: Int[Array, "45"],
        *args,
        inference: bool=False,
        **kwargs,
    ) -> Tuple[
        Float[Array, "45 15 d_model"],
        Float[Array, "45 d_model"]
    ]:
        del args, kwargs
        h_0 = h
        h = jax.vmap(self.lns_h[0])(
            h.reshape(-1, h.shape[-1])
        ).reshape(h_0.shape).transpose(1, 0, 2)
        mask = jnp.tri(45)
        key, key_att, key_do = jax.random.split(key, 3)
        h = jax.vmap(eqx.Partial(
            self.att_h,
            inference=inference,
            mask=mask,
            process_heads=self.process_heads,
            key_=s,
            value=s
        ), in_axes=(None, 0))(
            key_att,
            h,
        ).transpose(1, 0, 2)
        h = h_0 + self.dropout(h, inference=inference, key=key_do)
        h_0 = h
        h = jax.vmap(self.lns_h[1])(
            h.reshape(-1, h.shape[-1])
        )
        key, key_mlp, key_do = jax.random.split(key, 3)
        h = self.mlp_h(
            key_mlp,
            h,
            inference=inference
        ).reshape(h_0.shape)
        h = h_0 + self.dropout(h, inference=inference, key=key_do)

        s_0 = s
        s = jax.vmap(self.lns_s[0])(s)
        key, key_att, key_do = jax.random.split(key, 3)
        s = self.att_s(
            key_att,
            query=h[jnp.arange(h.shape[0]), picks_pos],
            key_=h,
            value=h,
            mask=pack_mask,
            inference=inference
        )
        s = s_0 + self.dropout(s, inference=inference, key=key_do)
        s_0 = s
        s = jax.vmap(self.lns_s[1])(s)
        key, key_mlp, key_do = jax.random.split(key, 3)
        s = self.mlp_s(
            key_mlp,
            s,
            inference=inference
        )
        s = s_0 + self.dropout(s, inference=inference, key=key_do)
        return h, s

