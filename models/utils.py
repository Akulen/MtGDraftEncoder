import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional
from jaxtyping import Float, Array, Bool, PRNGKeyArray

from data_types import Cards, Sets, Drafts
import utils

class CardPreProcess(eqx.Module):
    reduce: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        d_model: int=64,
        dropout_p: float=0.1
    ):
        self.reduce = eqx.nn.Linear(
            in_features=cards.textual_features.shape[1]
                      + cards.numeric_features.shape[1],
            out_features=d_model,
            key=key
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        inference: bool=False
    ) -> Float[Array, "set_size_max d_model"]:
        card_features = jnp.concat([
            jax.vmap(utils.layer_norm)(cards.textual_features),
            jax.vmap(utils.layer_norm)(cards.numeric_features)
        ], axis=-1)
        set_cards = sets.card_ids[drafts.set_id]
        h_cards_0 = card_features[set_cards]
        h_cards_0 = jax.vmap(self.reduce)(h_cards_0)
        h_cards_0 = jax.nn.leaky_relu(h_cards_0)
        h_cards_0 = self.dropout(h_cards_0, inference=inference, key=key)
        return h_cards_0

class AttentionUpdater(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2
    ):
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
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        key: PRNGKeyArray,
        query: Float[Array, "q_len d_model"],
        key_: Float[Array, "kv_len d_model"],
        value: Float[Array, "kv_len d_model"],
        mask: Optional[Bool[Array, "q_len kv_len"]]=None,
        process_heads: Optional[Callable]=None,
        inference: bool=False,
    ) -> Float[Array, "q_len d_model"]:
        key_attention, key_dropout = jax.random.split(key)
        update = self.attention(
            query=query,
            key_=key_,
            value=value,
            mask=mask,
            inference=inference,
            process_heads=process_heads,
            key=key_attention
        )
        x = jax.vmap(utils.layer_norm)(query + update)
        x = jax.nn.leaky_relu(x)
        x = self.dropout(x, inference=inference, key=key_dropout)
        return x
