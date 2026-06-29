import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from data_types import Cards, Sets, Drafts

class DraftWRPredictor(eqx.Module, ABC):
    @abstractmethod
    def __call__(self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        pass

class ModelLayer(eqx.Module, ABC):
    @abstractmethod
    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, "n d_model"],
        *args,
        inference: bool=False,
        **kwargs
    ) -> Float[Array, "n d_model"]:
        pass

class MLPLayer(ModelLayer):
    nn: eqx.nn.MLP

    def __init__(self,
        key: PRNGKeyArray,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation=jax.nn.relu,
        nn: Optional[eqx.nn.MLP]=None
    ):
        if nn is None:
            self.nn = eqx.nn.MLP(
                in_size=in_size,
                out_size=out_size,
                width_size=width_size,
                depth=depth,
                activation=activation,
                key=key
            )
        else:
            self.nn = nn

    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, "n d_model"],
        *args,
        **kwargs
    ) -> Float[Array, "45"]:
        del key, args, kwargs
        return jax.vmap(self.nn)(x)

class ResidualLayer(eqx.Module):
    lns: List[eqx.nn.LayerNorm]
    layers: List[ModelLayer]
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        layers: List[ModelLayer],
        dropout_p: float=0.1,
        d_model: int=64,
        ff_depth: Optional[int]=None,
        ff_activation=jax.nn.relu
    ):
        self.layers = layers
        if ff_depth is not None:
            self.layers.append(MLPLayer(
                in_size=d_model,
                out_size=d_model,
                width_size=d_model,
                depth=ff_depth,
                activation=ff_activation,
                key=key
            ))
        self.lns = [eqx.nn.LayerNorm(d_model) for _ in range(len(layers))]
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(self,
        key: PRNGKeyArray,
        x: Float[Array, "n d_model"],
        *args,
        inference: bool=False,
        **kwargs,
    ) -> Float[Array, "n d_model"]:
        for ln, layer in zip(self.lns, self.layers):
            x_0 = x
            x = jax.vmap(ln)(x)
            key, key_layer, key_dropout = jax.random.split(key, 3)
            x = layer(key_layer, x, *args, inference=inference, **kwargs)
            x = x_0 + self.dropout(x, inference=inference, key=key_dropout)
        return x

class PackStateAttention(eqx.Module):
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
        query: Float[Array, "45 15 d_model"],
        key_: Float[Array, "45 d_model"],
        value: Float[Array, "45 d_model"],
        mask: Optional[Bool[Array, "45 15"]]=None,
        inference: bool=False
    ) -> Float[Array, "45 d_model"]:
        d_model = query.shape[-1]
        assert(query.shape == (45, 15, d_model))
        assert(key_.shape == (45, d_model))
        assert(value.shape == (45, d_model))
        Q = jax.vmap(jax.vmap(self.W_q))(
            query
        ).reshape((45, 15, self.num_heads, -1))
        K = jax.vmap(self.W_k)(
            key_
        ).reshape((45, self.num_heads, -1))
        V = jax.vmap(self.W_v)(
            value
        ).reshape((45, self.num_heads, -1))

        logits = jnp.einsum("qkhd,qhd->hqk", Q, K) / jnp.sqrt(Q.shape[-1])
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
