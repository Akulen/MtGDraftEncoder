from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Int, Float, PRNGKeyArray

from models._types import DraftWRPredictor
from data_types import Cards, Sets, Drafts
import utils

class DraftGraph(DraftWRPredictor):
    linear_reduce: eqx.nn.Linear
    card_attention: eqx.nn.MultiheadAttention
    pack_attention: eqx.nn.MultiheadAttention
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding
    pick_attention: eqx.nn.MultiheadAttention
    # card_linear: eqx.nn.Linear
    card_attention2: eqx.nn.MultiheadAttention
    linear_predict: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2
    ):
        (
            key_reduce, key_catt, key_patt, key_pick, key_catt2, key_predict
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
            key=key_catt
        )
        self.pack_attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_p,
            key=key_patt
        )
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=d_model // num_heads
        )
        self.pick_attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_p,
            key=key_pick
        )
        # self.card_linear = eqx.nn.Linear(
        #     in_features=2*d_model,
        #     out_features=d_model,
        #     key=key_catt2
        # )
        self.card_attention2 = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_p,
            key=key_catt2
        )
        self.linear_predict = eqx.nn.Linear(
            in_features=d_model,
            out_features=1,
            key=key_predict
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
            value_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(
                value_heads
            )
            return query_heads, key_heads, value_heads

        card_features = jnp.concat([
            cards.textual_features,
            cards.numeric_features
        ], axis=1)

        set_cards = sets.card_ids[drafts.set_id]
        rev_idx = (
            jnp.full(card_features.shape[0], -1, dtype=jnp.int32)
               .at[set_cards].set(jnp.arange(set_cards.shape[0]))
        ).at[0].set(-2) # Mark padding card specially
        h_cards_0 = card_features[set_cards]
        h_cards_0 = jax.vmap(utils.layer_norm)(h_cards_0.T).T
        h_cards_0 = jax.vmap(self.linear_reduce)(h_cards_0)
        h_cards_0 = jax.nn.relu(h_cards_0)
        key, subkey = jax.random.split(key)
        h_cards_0 = self.dropout(h_cards_0, inference=inference, key=subkey)
        # h_cards = h_cards_0
        card_mask = jnp.arange(set_cards.shape[0])<sets.set_size[drafts.set_id]

        key, key_att, key_drop = jax.random.split(key, 3)
        draft_vectors = jnp.zeros(
            (46, h_cards_0.shape[1]),
            dtype=h_cards_0.dtype
        )
        init_state = self.dropout(self.card_attention(
            query=jax.vmap(
                partial(utils.masked_mean, mask=card_mask), in_axes=1
            )(h_cards_0).reshape((1, -1)),
            key_=h_cards_0,
            value=h_cards_0,
            mask=card_mask.reshape((1, -1)),
            inference=inference,
            key=key_att
        ).squeeze(), inference=inference, key=key_drop)
        draft_vectors = draft_vectors.at[0].set(init_state)

        def pick_update(
            carry: Tuple[
                Float[Array, "46 d_model"],           # draft states
                # Float[Array, "set_size_max d_model"], # h_cards
                PRNGKeyArray                          # key
            ],
            data: Tuple[
                Int[Array, ""],   # i_pick
                Int[Array, "15"], # pack
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
            i_pick, pack, pick = data

            key, key_att, key_drop = jax.random.split(key, 3)
            h_cards = self.card_attention2(
                query=h_cards_0[pack],
                key_=draft_vectors,
                value=draft_vectors,
                mask=jnp.tile(jnp.arange(46) <= i_pick, (pack.shape[0], 1)),
                inference=inference,
                process_heads=process_heads,
                key=key_att
            )
            h_cards = utils.layer_norm(h_cards)
            h_cards = jax.nn.relu(h_cards)
            h_cards = self.dropout(h_cards, inference=inference, key=key_drop)

            key, key_att, key_drop = jax.random.split(key, 3)
            pack_mask = card_mask[pack]
            pick_id = jnp.argwhere(pack == pick, size=1, fill_value=-1)[0, 0]
            pack = self.pack_attention(
                query=h_cards[pick_id].reshape((1, -1)),
                key_=h_cards_0[pack],
                value=h_cards, #[pack],
                mask=pack_mask.reshape((1, -1)),
                inference=inference,
                key=key_att
            ).squeeze()
            pack = utils.layer_norm(pack)
            pack = jax.nn.relu(pack)
            pack = self.dropout(pack, inference=inference, key=key_drop)

            key, key_att, key_drop = jax.random.split(key, 3)
            draft_state = self.pick_attention(
                query=pack.reshape((1, -1)),
                key_=draft_vectors,
                value=draft_vectors,
                mask=(jnp.arange(46) <= i_pick).reshape((1, -1)),
                inference=inference,
                process_heads=process_heads,
                key=key_att
            ).squeeze()
            draft_state = utils.layer_norm(draft_state)
            draft_state = jax.nn.relu(draft_state)
            draft_state = self.dropout(
                draft_state, inference=inference, key=key_drop
            )
            draft_vectors = draft_vectors.at[i_pick+1].set(draft_state)

            # key, key_att, key_drop = jax.random.split(key, 3)
            # # Idea: Only update h_card for cards actually in packs, and from h_0
            # h_cards_new = jax.vmap(self.card_linear)(jnp.concatenate([
            #     h_cards,
            #     jnp.tile(draft_state, (h_cards.shape[0], 1))
            # ], axis=1))
            # h_cards_new = jax.nn.relu(h_cards_new)
            # # h_cards_new = self.card_attention2(
            # #     query=h_cards,
            # #     key_=draft_vectors,
            # #     value=draft_vectors,
            # #     mask=jnp.tile(jnp.arange(46)<=i_pick+1, (h_cards.shape[0], 1)),
            # #     inference=inference,
            # #     process_heads=process_heads,
            # #     key=key_att
            # # )
            # h_cards_new = self.dropout(
            #     h_cards_new, inference=inference, key=key_drop
            # )
            # h_cards += h_cards_new
            # h_cards = jax.vmap(utils.layer_norm)(h_cards)
            # # h_cards = jnp.where(
            # #     i_pack < 3 * sets[drafts.set_id].pack_size,
            # #     g(h_cards, h_cards_0, h_pick),
            # #     h_cards
            # # )

            return (draft_vectors, key), None

        packs = rev_idx[drafts.packs]
        picks = rev_idx[drafts.picks]
        (draft_vectors, key), _ = jax.lax.scan(
            pick_update,
            (draft_vectors, key),
            (jnp.arange(45), packs, picks),
            unroll=3
        )
        
        # One prediction for each pick, using only previous context
        y = jax.vmap(self.linear_predict)(draft_vectors[1:]).squeeze()
        y = jax.nn.sigmoid(y)
        return y, state
