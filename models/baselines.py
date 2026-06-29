import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from models._types import DraftWRPredictor
from data_types import Cards, Sets, Drafts

class LinearRegression(DraftWRPredictor):
    use_meta: bool=eqx.field(static=True)
    rank_embedding: eqx.nn.Embedding
    linear: List[eqx.nn.Linear]

    def __init__(self, key: PRNGKeyArray, cards: Cards, use_meta: bool=False):
        self.use_meta = use_meta
        d_t = cards.textual_features.shape[1]
        d_n = cards.numeric_features.shape[1] if use_meta else 0
        key, subkey = jax.random.split(key)
        self.rank_embedding = eqx.nn.Embedding(
            num_embeddings=7,
            embedding_size=64,
            key=subkey
        )
        subkeys = jax.random.split(subkey, 45)
        self.linear = [
            eqx.nn.Linear(
                in_features=(i+1)*(d_t+d_n)+64,
                out_features=1,
                key=subkeys[i]
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
        features = cards.textual_features[picked_cards]
        if self.use_meta:
            features = jnp.concatenate([
                features,
                cards.numeric_features[picked_cards]
            ], axis=1)
        picked_data = jnp.where(
            (picked_cards == 0).reshape((-1, 1)),
            0,
            features
        )
        rank_embedding = self.rank_embedding(drafts.rank)
        x = jnp.stack([
            self.linear[i-1](jnp.concatenate([
                picked_data[:i].reshape((-1)),
                rank_embedding
            ])).squeeze()
            for i in range(1, 46)
        ])
        return jax.nn.sigmoid(x), state

class Constant(DraftWRPredictor):
    output: Float[Array, ""]

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        use_meta: bool=False
    ):
        del key, cards, use_meta
        self.output = jnp.array(0.5)

    def __call__(self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        return jnp.repeat(self.output, 45), state

class Heuristic(DraftWRPredictor):
    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        use_meta: bool=False
    ):
        del key, cards
        if not use_meta:
            raise AttributeError('Heuristic model requires access to meta stats.')

    def __call__(self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        picks_deck_wr = cards.numeric_features[drafts.picks][:,3]

        idx = jnp.arange(45)
        masked = jnp.where(
            idx[None, :] <= idx[:, None],
            picks_deck_wr[None, :],
            -jnp.inf
        )
        sorted_vals = jnp.sort(masked, axis=-1)[:, -23:] # Top 23 cards
        valid = sorted_vals != -jnp.inf

        return jnp.where(
            valid, sorted_vals, 0.0
        ).sum(axis=1) / valid.sum(axis=1), state
