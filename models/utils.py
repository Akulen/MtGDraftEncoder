import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Tuple
from jaxtyping import Float, Array, PRNGKeyArray

from models._types import ResidualLayer
from data_types import Cards, Sets, Drafts
import utils

class CardPreProcess(eqx.Module):
    use_meta: bool=eqx.field(static=True)
    reduce: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        use_meta: bool,
        d_model: int=64,
        dropout_p: float=0.1
    ):
        self.use_meta = use_meta
        n_features = cards.textual_features.shape[1]
        if use_meta:
            n_features += cards.numeric_features.shape[1]
        self.reduce = eqx.nn.Linear(
            in_features=n_features,
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
        features = [cards.textual_features]
        if self.use_meta:
            features.append(cards.numeric_features)
        card_features = jnp.concat([
            jax.vmap(utils.layer_norm)(feature)
            for feature in features
        ], axis=-1)
        set_cards = sets.card_ids[drafts.set_id]
        h_cards_0 = card_features[set_cards]
        h_cards_0 = jax.vmap(self.reduce)(h_cards_0)
        h_cards_0 = jax.nn.gelu(h_cards_0) # jax.nn.leaky_relu(h_cards_0)
        h_cards_0 = self.dropout(h_cards_0, inference=inference, key=key)
        return h_cards_0

class SetContext(eqx.Module):
    process_cards: CardPreProcess
    context_film: eqx.nn.MLP
    layers: List[ResidualLayer]

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        use_meta: bool,
        d_model: int=64,
        dropout_p: float=0.1,
        n_layers: int=3,
        layer_fns: List[Callable]=[]
    ):
        key_process, key_film, key_layers = jax.random.split(key, 3)
        self.process_cards = CardPreProcess(
            key=key_process,
            cards=cards,
            use_meta=use_meta,
            d_model=d_model,
            dropout_p=dropout_p
        )
        self.context_film = eqx.nn.MLP(
            in_size=d_model,
            out_size=2 * d_model,
            width_size=d_model,
            depth=1,
            activation=jax.nn.gelu, #jax.nn.relu,
            key=key_film
        )
        keys = jax.random.split(key_layers, (n_layers, 1+len(layer_fns)))
        self.layers = [
            ResidualLayer(
                key=layer_keys[0],
                layers=[
                    layer_fn(subkey)
                    for subkey, layer_fn in zip(layer_keys[1:], layer_fns)
                ],
                dropout_p=dropout_p,
                d_model=d_model,
                ff_depth=1
            )
            for layer_keys in keys
        ]

    def __call__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        context: Float[Array, "d_model"],
        state: eqx.nn.State,
        *args,
        inference: bool=False
    ) -> Tuple[Float[Array, "set_size_max d_model"], eqx.nn.State]:
        gamma, beta = jnp.split(self.context_film(context), 2, axis=-1)
        gamma = 1.0 + 0.1 * jnp.tanh(gamma)

        key, subkey = jax.random.split(key)
        h_cards_0 = gamma * self.process_cards(
            key=subkey,
            cards=cards,
            sets=sets,
            drafts=drafts,
            inference=inference
        ) + beta

        card_mask = jnp.arange(
            sets.card_ids.shape[1]
        ) < sets.set_size[drafts.set_id]

        keys = jax.random.split(key, len(self.layers))
        for subkey, layer in zip(keys, self.layers):
            h_cards_0 = layer(
                key,
                h_cards_0,
                *args,
                inference=inference,
                mask=jnp.outer(card_mask, card_mask)
            )
        return h_cards_0, state
