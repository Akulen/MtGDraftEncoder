import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from models._types import DraftWRPredictor
from data_types import Cards, Sets, Drafts

class LinearRegression(DraftWRPredictor):
    linear: List[eqx.nn.Linear]

    def __init__(self, key: PRNGKeyArray, cards: Cards):
        d_t = cards.textual_features.shape[1]
        d_n = cards.numeric_features.shape[1]
        key, subkey = jax.random.split(key)
        self.linear = [
            eqx.nn.Linear(
                in_features=(i+1)*(d_t+d_n),
                out_features=1,
                key=subkey
            )
            for i in range(45)
        ]

    def __call__(self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        del key
        del sets
        del inference
        assert jnp.issubdtype(cards.textual_features.dtype, jnp.floating)
        picked_cards = drafts.picks # 45
        picked_data = jnp.where(
            (picked_cards == 0).reshape((-1, 1)),
            0,
            jnp.concat([
                cards.textual_features[picked_cards],
                cards.numeric_features[picked_cards]
            ], axis=1)
        )
        x = [
            self.linear[i-1](picked_data[:i].reshape((-1))).squeeze()
            for i in range(1, 46)
        ]
        return jnp.stack(x), state
