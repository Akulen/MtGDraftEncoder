import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from data_types import Cards, Drafts

class LinearRegression(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key: PRNGKeyArray, cards: Cards):
        d_t = cards.textual_features.shape[1]
        d_n = cards.numeric_features.shape[1]
        key, subkey = jax.random.split(key)
        self.linear = eqx.nn.Linear(
            in_features=45*(d_t+d_n),
            out_features=1,
            key=subkey
        )

    def __call__(self,
        key: PRNGKeyArray, cards: Cards, drafts: Drafts, state: eqx.nn.State
    ) -> Tuple[Float[Array, ""], eqx.nn.State]:
        del key
        assert jnp.issubdtype(cards.textual_features.dtype, jnp.floating)
        picked_cards = drafts.picks # batch_size x 45
        x = jnp.concat([
            cards.textual_features[picked_cards],
            cards.numeric_features[picked_cards]
        ], axis=1).reshape((-1))
        return self.linear(x).squeeze(), state
