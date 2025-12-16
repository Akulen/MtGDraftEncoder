from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Bool, Int, Float, PRNGKeyArray

from models._types import DraftWRPredictor
from models.utils import AttentionUpdater
from data_types import Cards, Sets, Drafts
import utils

class DraftGraph(DraftWRPredictor):
    linear_reduce: eqx.nn.Linear
    card_attention: eqx.nn.MultiheadAttention
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    card_updater: AttentionUpdater
    pack_updater: AttentionUpdater
    state_updater: AttentionUpdater
    predictor_head: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        pred_n_layers: int=2
    ):
        (
            key_reduce, key_att, key_card, key_pack, key_state, key_predict
        )= jax.random.split(key, 6)
        self.linear_reduce = eqx.nn.Linear(
            in_features=cards.textual_features.shape[1]
                      + cards.numeric_features.shape[1],
            out_features=d_model,
            key=key_reduce
        )
        self.card_attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_p,
            key=key_att
        )
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=d_model // num_heads
        )
        self.card_updater = AttentionUpdater(
            key=key_card,
            dropout_p=dropout_p,
            d_model=d_model,
            num_heads=num_heads
        )
        self.pack_updater = AttentionUpdater(
            key=key_pack,
            dropout_p=dropout_p,
            d_model=d_model,
            num_heads=num_heads
        )
        self.state_updater = AttentionUpdater(
            key=key_state,
            dropout_p=dropout_p,
            d_model=d_model,
            num_heads=num_heads
        )
        self.predictor_head = eqx.nn.MLP(
            in_size=d_model,
            out_size=1,
            width_size=2*d_model,
            depth=pred_n_layers,
            activation=jax.nn.leaky_relu,
            final_activation=jax.nn.sigmoid,
            key=key_predict,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)
        # for module in self.__dict__:
        #     print(module, sum(x.size for x in jax.tree_util.tree_leaves(
        #         eqx.filter(getattr(self, module), eqx.is_inexact_array)
        #     )))

    def __call__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        def process_heads(
            query_heads: Float[Array, "l_q num_heads qk_size"],
            key_heads: Float[Array, "l_kv num_heads qk_size"],
            value_heads: Float[Array, "l_kv num_heads vo_size"],
        ) -> Tuple[
            Float[Array, "l_q num_heads qk_size"],
            Float[Array, "l_kv num_heads qk_size"],
            Float[Array, "l_kv num_heads vo_size"]
        ]:
            key_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(
                key_heads
            )
            return query_heads, key_heads, value_heads

        card_features = jnp.concat([
            jax.vmap(utils.layer_norm)(cards.textual_features),
            jax.vmap(utils.layer_norm)(cards.numeric_features)
        ], axis=1)

        set_cards = sets.card_ids[drafts.set_id]
        rev_idx = (
            jnp.full(card_features.shape[0], -1, dtype=jnp.int32)
               .at[set_cards].set(jnp.arange(set_cards.shape[0]))
        ).at[0].set(-2) # Mark padding card specially
        h_cards_0 = card_features[set_cards]
        h_cards_0 = jax.vmap(self.linear_reduce)(h_cards_0)
        h_cards_0 = jax.nn.leaky_relu(h_cards_0)
        key, subkey = jax.random.split(key)
        h_cards_0 = self.dropout(h_cards_0, inference=inference, key=subkey)
        card_mask = jnp.arange(set_cards.shape[0])<sets.set_size[drafts.set_id]

        key, key_att, key_drop = jax.random.split(key, 3)
        draft_vectors = jnp.zeros(
            (46, h_cards_0.shape[1]),
            dtype=h_cards_0.dtype
        )
        init_state = self.card_attention(
            query=jax.vmap(
                partial(utils.masked_mean, mask=card_mask), in_axes=1
            )(h_cards_0).reshape((1, -1)),
            key_=h_cards_0,
            value=h_cards_0,
            mask=card_mask.reshape((1, -1)),
            inference=inference,
            key=key_att
        ).squeeze()
        init_state = utils.layer_norm(init_state)
        init_state = jax.nn.leaky_relu(init_state)
        init_state = self.dropout(init_state, inference=inference, key=key_drop)
        draft_vectors = draft_vectors.at[0].set(init_state)

        # @partial(jax.jit, donate_argnums=(0,))
        def pick_update(
            carry: Tuple[
                Float[Array, "46 d_model"],           # draft states
                # Float[Array, "set_size_max d_model"], # h_cards
                PRNGKeyArray                          # key
            ],
            data: Tuple[
                Int[Array, ""],   # i_pick
                Float[Array, "15 d_model"], # pack
                Bool[Array, "15"],
                Int[Array, ""]    # pick
            ]
        ) -> Tuple[
            Tuple[
                Float[Array, "45 d_model"],           # draft states
                # Float[Array, "set_size_max d_model"], # h_cards
                PRNGKeyArray                          # key
            ],
            None
        ]:
            draft_vectors, key = carry
            i_pick, h_pack_cards_0, pack_mask, pick_pos = data

            # TODO: GNN on interaction graph restricted to pool
            key, key_upd = jax.random.split(key)
            h_pack_cards = self.card_updater(
                key=key_upd,
                query=h_pack_cards_0,
                key_=draft_vectors[i_pick].reshape((1, -1)),
                value=draft_vectors[i_pick].reshape((1, -1)),
                inference=inference,
            )

            key, key_upd = jax.random.split(key)
            h_pack = self.pack_updater(
                key=key_upd,
                query=h_pack_cards[pick_pos].reshape((1, -1)),
                key_=h_pack_cards_0,
                value=h_pack_cards,
                mask=pack_mask.reshape((1, -1)),
                inference=inference,
            ).squeeze()

            key, key_upd = jax.random.split(key)
            draft_state = self.state_updater(
                key=key_upd,
                query=h_pack.reshape((1, -1)),
                key_=draft_vectors[:45],
                value=draft_vectors[:45],
                mask=(jnp.arange(45) <= i_pick).reshape((1, -1)),
                inference=inference,
                process_heads=process_heads,
            ).squeeze()
            draft_vectors = draft_vectors.at[i_pick+1].set(draft_state)

            return (draft_vectors, key), None

        packs = rev_idx[drafts.packs]
        picks = rev_idx[drafts.picks]
        pick_positions = jnp.argmax(packs == picks[:, None], axis=1)
        (draft_vectors, key), _ = jax.lax.scan(
            pick_update,
            (draft_vectors, key),
            (jnp.arange(45), h_cards_0[packs], card_mask[packs], pick_positions),
            unroll=5
        )
        
        # One prediction for each pick, using only previous context
        y = jax.vmap(self.predictor_head)(draft_vectors[1:]).squeeze()
        return y, state
