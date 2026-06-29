import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from models._types import DraftWRPredictor
from models.utils import SetContext
from models.attention import AlternatingResidualAttention
from data_types import Cards, Sets, Drafts
from models.gnn import GINWrapper


class DraftGraph2(DraftWRPredictor):
    rank_embedding: eqx.nn.Embedding
    set_context: SetContext
    ln: eqx.nn.LayerNorm
    layers: List[AlternatingResidualAttention]
    predictor_head: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        use_meta: bool=False,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        n_set_layers: int=3,
        n_layers: int=5,
        pred_n_layers: int=2
    ):
        (
            key_emb, key_context, key_predict, key_layers
        ) = jax.random.split(key, 4)
        self.rank_embedding = eqx.nn.Embedding(
            num_embeddings=7,
            embedding_size=d_model,
            key=key_emb
        )
        self.set_context = SetContext(
            key=key_context,
            cards=cards,
            use_meta=use_meta,
            d_model=d_model,
            dropout_p=dropout_p,
            n_layers=n_set_layers,
            layer_fns=[
                lambda subkey: GINWrapper(
                    key=subkey,
                    d_model=d_model,
                )
            ]
        )
        self.ln = eqx.nn.LayerNorm(d_model)
        keys = jax.random.split(key_layers, n_layers)
        self.layers = [
            AlternatingResidualAttention(
                subkey,
                dropout_p,
                d_model,
                num_heads,
                ff_depth=1
            )
            for subkey in keys
        ]
        self.predictor_head = eqx.nn.MLP(
            in_size=2*d_model,
            out_size='scalar',
            width_size=2*d_model,
            depth=pred_n_layers,
            activation=jax.nn.leaky_relu,
            final_activation=jax.nn.sigmoid,
            key=key_predict
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        set_cards = sets.card_ids[drafts.set_id]
        rev_idx = (
            jnp.full(cards.card_id.shape[0], -1, dtype=jnp.int32)
               .at[set_cards].set(jnp.arange(set_cards.shape[0]))
        ).at[0].set(-2) # Mark padding card specially
        card_mask = jnp.arange(
            sets.card_ids.shape[1]
        ) < sets.set_size[drafts.set_id]

        rank_embedding = self.rank_embedding(drafts.rank)

        senders, receivers = sets.graph[drafts.set_id]
        key, subkey = jax.random.split(key)
        h_cards_set, state = self.set_context(
            subkey, cards, sets, drafts, rank_embedding, state, senders,
            receivers, inference=inference
        )

        packs = rev_idx[drafts.packs]
        picks = rev_idx[drafts.picks]
        pick_positions = jnp.argmax(packs == picks[:, None], axis=1)
        h_cards_set = jax.vmap(self.ln)(h_cards_set)
        h = h_cards_set[packs]
        s = h_cards_set[picks]

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, len(self.layers))
        for subkey, layer in zip(keys, self.layers):
            h, s = layer(
                subkey,
                h,
                s,
                drafts.packs > 0,
                pick_positions,
                inference=inference
            )

        y = jax.vmap(self.predictor_head)(jnp.concatenate([
            s,
            jnp.tile(rank_embedding, (45, 1))
        ], axis=1))

        return y, state
