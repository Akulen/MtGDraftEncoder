from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from jaxtyping import Array, Int, Float, PRNGKeyArray

from models._types import ModelLayer

def segment_softmax(
    logits: Float[Array, "n"],
    segment_ids: Int[Array, "n"],
    num_segments: Optional[int]=None,
) -> Float[Array, "n"]:
    maxs = jax.ops.segment_max(
        logits, segment_ids, num_segments
    )
    logits = logits - maxs[segment_ids]
    logits = jnp.exp(logits)
    normalizers = jax.ops.segment_sum(
        logits, segment_ids, num_segments
    )
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax

class GATLayer(ModelLayer):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(self,
        key: PRNGKeyArray,
        d_model: int=64,
        **kwargs
    ):
        del kwargs
        key1, key2, key3 = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(d_model, d_model, key=key1)
        self.linear2 = eqx.nn.Linear(d_model, 1, key=key2)
        self.linear3 = eqx.nn.Linear(d_model, d_model, key=key3)

    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, "n_nodes d_model"],
        senders: Int[Array, "n_edges"],
        receivers: Int[Array, "n_edges"],
        *args,
        inference: bool=False,
        **kwargs
    ) -> Float[Array, "n_nodes d_model"]:
        del key, args, inference, kwargs

        nodes = x
        num_nodes = nodes.shape[0]
        num_edges = senders.shape[0]

        nodes_proj = jax.vmap(self.linear1)(nodes)
        attn_weights = jax.nn.leaky_relu(
            jax.vmap(self.linear2)(
                nodes_proj[senders] + nodes_proj[receivers]
            )
        ).reshape(num_edges)

        attn_coeff = segment_softmax(
            attn_weights,
            receivers,
            num_segments=num_nodes
        )
        messages = (
            attn_coeff[:, None]
            * jax.vmap(self.linear3)(nodes)[senders]
        )
        new_nodes = jax.ops.segment_sum(
            messages,
            receivers,
            num_segments=num_nodes
        )

        return new_nodes

class GATLayerV2(ModelLayer):
    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    W_o: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    num_heads: int=eqx.field(static=True)

    def __init__(self,
        key: PRNGKeyArray,
        d_model: int=64,
        num_heads: int=2,
        dropout_p: float=0.1,
        **kwargs
    ):
        del kwargs
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

    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, "n_nodes d_model"],
        senders: Int[Array, "n_edges"],
        receivers: Int[Array, "n_edges"],
        *args,
        inference: bool=False,
        **kwargs
    ) -> Float[Array, "n_nodes d_model"]:
        del args, kwargs

        nodes = x
        num_nodes = nodes.shape[0]
        num_edges = senders.shape[0]

        nodes_query = jax.vmap(self.W_q)(
            nodes
        ).reshape((num_nodes, self.num_heads, -1))
        nodes_key = jax.vmap(self.W_k)(
            nodes
        ).reshape((num_nodes, self.num_heads, -1))
        nodes_value = jax.vmap(self.W_v)(
            nodes
        ).reshape((num_nodes, self.num_heads, -1))

        attn_weights = jnp.einsum(
            "mhd,mhd->hm", nodes_query[senders], nodes_key[receivers]
        ) / jnp.sqrt(nodes_query.shape[-1])
        # attn_weights = jax.nn.leaky_relu(
        #     jax.vmap(self.linear2)(
        #         nodes_proj[senders] + nodes_proj[receivers]
        #     )
        # ).reshape(num_edges)

        attn_coeff = jax.vmap(partial(segment_softmax,
            segment_ids=receivers,
            num_segments=num_nodes
        ))(attn_weights)
        attn_coeff = self.dropout(attn_coeff, inference=inference, key=key)
        messages = jnp.einsum(
            'hm,mhd->hmd',
            attn_coeff,
            nodes_value[senders]
        )
        new_nodes = jax.vmap(partial(jax.ops.segment_sum,
            segment_ids=receivers,
            num_segments=num_nodes
        ))(messages)
        new_nodes = jnp.transpose(
            new_nodes,
            (1, 0, 2)
        ).reshape((num_nodes, -1))
        new_nodes = jax.vmap(self.W_o)(new_nodes)

        return new_nodes

class GINLayer(eqx.Module):
    nn: eqx.nn.MLP
    epsilon: float
    self_edge: float

    def __init__(
        self, key: PRNGKeyArray, nn: eqx.nn.MLP, epsilon: float=0.0,
        self_edge: bool=True
    ):
        del key
        self.nn = nn
        self.epsilon = epsilon
        self.self_edge = 1 if self_edge else 0

    def __call__(
        self,
        key: PRNGKeyArray,
        nodes: Float[Array, "n_nodes d_model"],
        senders: Int[Array, "n_edges"],
        receivers: Int[Array, "n_edges"],
        # state: eqx.nn.State,
        inference: bool=False
    ) -> Float[Array, "n_nodes d_model"]:
        x = nodes
        x = x * (self.self_edge + self.epsilon) + jax.ops.segment_sum(
            x[senders],
            receivers,
            num_segments=x.shape[0]
        )
        # x, state = self.nn(x, key, state, training)
        x = jax.vmap(self.nn)(x)
        return x

class GINWrapper(ModelLayer):
    gin: GINLayer

    def __init__(self,
        key: PRNGKeyArray,
        d_model: int=64,
        nn: Optional[eqx.nn.MLP]=None,
        epsilon: float=0.0,
        self_edge: bool=True
    ):
        key_mlp, key_gin = jax.random.split(key, 2)
        if nn is None:
            nn = eqx.nn.MLP(
                in_size=d_model,
                out_size=d_model,
                width_size=d_model,
                depth=1,
                key=key_mlp,
            )
        self.gin = GINLayer(key_gin, nn, epsilon, self_edge)

    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, 'n d_model'], 
        senders: Int[Array, 'n_edges'],
        receivers: Int[Array, 'n_edges'],
        *args,
        inference: bool=False,
        **kwargs
    ) -> Float[Array, 'n d_model']:
        return self.gin(
            key,
            nodes=x,
            senders=senders,
            receivers=receivers,
            inference=inference
        )
