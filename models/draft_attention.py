from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional, Tuple
from jaxtyping import Array, Bool, Int, Float, PRNGKeyArray

from models._types import DraftWRPredictor
from models.utils import AttentionUpdater, CardPreProcess
from data_types import Cards, Sets, Drafts
import utils

class DraftAttention(DraftWRPredictor):
    process_cards: CardPreProcess
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
            key_process, key_att, key_card, key_pack, key_state, key_predict
        )= jax.random.split(key, 6)
        self.process_cards = CardPreProcess(
            key=key_process,
            cards=cards,
            d_model=d_model,
            dropout_p=dropout_p
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

        set_cards = sets.card_ids[drafts.set_id]
        rev_idx = (
            jnp.full(cards.card_id.shape[0], -1, dtype=jnp.int32)
               .at[set_cards].set(jnp.arange(set_cards.shape[0]))
        ).at[0].set(-2) # Mark padding card specially
        card_mask = jnp.arange(set_cards.shape[0])<sets.set_size[drafts.set_id]

        key, subkey = jax.random.split(key)
        h_cards_0 = self.process_cards(
            key=subkey,
            cards=cards,
            sets=sets,
            drafts=drafts,
            inference=inference
        )

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

class PackAttention(eqx.Module):
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
        query: Float[Array, "45 d_model"],
        key_: Float[Array, "45 15 d_model"],
        value: Float[Array, "45 15 d_model"],
        mask: Optional[Bool[Array, "45 15"]]=None,
        inference: bool=False
    ) -> Float[Array, "45 d_model"]:
        d_model = query.shape[-1]
        assert(query.shape == (45, d_model))
        assert(key_.shape == (45, 15, d_model))
        assert(value.shape == (45, 15, d_model))
        Q = jax.vmap(self.W_q)(query).reshape((45, self.num_heads, -1))
        K = jax.vmap(jax.vmap(self.W_k))(key_).reshape((45, 15, self.num_heads, -1))
        V = jax.vmap(jax.vmap(self.W_v))(value).reshape((45, 15, self.num_heads, -1))

        logits = jnp.einsum("qhd,qkhd->hqk", Q, K) / jnp.sqrt(Q.shape[-1])
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

class TransformerLayer(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    feed_forward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        ff_depth: int=1
    ):
        key_att, key_mlp = jax.random.split(key, 2)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=d_model,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_p,
            key=key_att
        )
        self.feed_forward = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=d_model,
            depth=ff_depth,
            activation=jax.nn.relu,
            key=key_mlp
        )
        self.dropout = eqx.nn.Dropout(dropout_p)
    
    def __call__(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "45 d_model"],
        mask: Optional[Bool[Array, "45 45"]]=None,
        inference: bool=False
    ) -> Float[Array, "45 d_model"]:
        key, key_att, key_drop = jax.random.split(key, 3)
        x_0 = x
        x = jax.vmap(utils.layer_norm)(x)
        x = self.attention(
            x, x, x,
            mask=mask,
            inference=inference,
            key=key_att
        )
        x = x_0 + self.dropout(x, inference=inference, key=key_drop)

        x_0 = x
        x = jax.vmap(utils.layer_norm)(x)
        x = jax.vmap(self.feed_forward)(x)
        x = x_0 + self.dropout(x, inference=inference, key=key_drop)

        return x

class DraftTransformer(DraftWRPredictor):
    process_cards: CardPreProcess
    pack_att: PackAttention
    set_layers: List[TransformerLayer]
    layers: List[TransformerLayer]
    predictor_head: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        n_set_layers: int=3,
        n_layers: int=5,
        pred_n_layers: int=2
    ):
        (
            key_process, key_pack, key_set, key_layers, key_predict
        ) = jax.random.split(key, 5)
        self.process_cards = CardPreProcess(
            key=key_process,
            cards=cards,
            d_model=d_model,
            dropout_p=dropout_p
        )
        self.pack_att = PackAttention(key_pack, d_model, num_heads, dropout_p)
        keys = jax.random.split(key_set, n_set_layers)
        self.set_layers = [
            TransformerLayer(
                key=subkey,
                dropout_p=dropout_p,
                d_model=d_model,
                num_heads=num_heads
            )
            for subkey in keys
        ]
        keys = jax.random.split(key_layers, n_layers)
        self.layers = [
            TransformerLayer(
                key=subkey,
                dropout_p=dropout_p,
                d_model=d_model,
                num_heads=num_heads
            )
            for subkey in keys
        ]
        self.predictor_head = eqx.nn.MLP(
            in_size=d_model,
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
        card_mask = jnp.arange(set_cards.shape[0])<sets.set_size[drafts.set_id]

        key, subkey = jax.random.split(key)
        h_cards_0 = self.process_cards(
            key=subkey,
            cards=cards,
            sets=sets,
            drafts=drafts,
            inference=inference
        )

        keys = jax.random.split(subkey, len(self.set_layers))
        for subkey, layer in zip(keys, self.set_layers):
            h_cards_0 = layer(
                subkey,
                h_cards_0,
                mask=jnp.outer(card_mask, card_mask),
                inference=inference
            )

        packs = rev_idx[drafts.packs]
        picks = rev_idx[drafts.picks]
        key, subkey = jax.random.split(key)
        h_picks_0 = self.pack_att(
            key=subkey,
            query=h_cards_0[picks], 
            key_=h_cards_0[packs],
            value=h_cards_0[packs],
            mask=drafts.packs > 0,
            inference=inference
        )

        mask = jnp.tri(45)
        h_picks = h_picks_0
        keys = jax.random.split(subkey, len(self.layers))
        for subkey, layer in zip(keys, self.layers):
            h_picks = layer(subkey, h_picks, mask, inference)

        y = jax.vmap(self.predictor_head)(h_picks)

        return y, state

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

class GraphLayer(eqx.Module):
    gnn: GINLayer
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        ff_depth: int=1
    ):
        key_att, key_mlp = jax.random.split(key, 2)
        self.gnn = GINLayer(
            key=key_att,
            nn=eqx.nn.MLP(
                in_size=d_model,
                out_size=d_model,
                width_size=d_model,
                depth=1,
                activation=jax.nn.relu,
                key=key_att
            )
        )
        # self.feed_forward = eqx.nn.MLP(
        #     in_size=d_model,
        #     out_size=d_model,
        #     width_size=d_model,
        #     depth=ff_depth,
        #     activation=jax.nn.relu,
        #     key=key_mlp
        # )
        self.dropout = eqx.nn.Dropout(dropout_p)
    
    def __call__(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "n_nodes d_model"],
        senders: Int[Array, "n_edges"],
        receivers: Int[Array, "n_edges"],
        # mask: Optional[Bool[Array, "45 45"]]=None,
        inference: bool=False
    ) -> Float[Array, "45 d_model"]:
        key, key_att, key_drop = jax.random.split(key, 3)
        x_0 = x
        x = jax.vmap(utils.layer_norm)(x)
        x = self.gnn(
            key_att,
            x, senders, receivers,
            # mask=mask,
            inference=inference,
        )
        x = x_0 + self.dropout(x, inference=inference, key=key_drop)

        # x_0 = x
        # x = jax.vmap(utils.layer_norm)(x)
        # x = jax.vmap(self.feed_forward)(x)
        # x = x_0 + self.dropout(x, inference=inference, key=key_drop)

        return x

class DraftGraph(DraftWRPredictor):
    process_cards: CardPreProcess
    pack_att: PackAttention
    set_layers: List[GraphLayer]
    layers: List[TransformerLayer]
    predictor_head: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        key: PRNGKeyArray,
        cards: Cards,
        dropout_p: float=0.1,
        d_model: int=64,
        num_heads: int=2,
        n_set_layers: int=3,
        n_layers: int=5,
        pred_n_layers: int=2
    ):
        (
            key_process, key_pack, key_set, key_layers, key_predict
        ) = jax.random.split(key, 5)
        self.process_cards = CardPreProcess(
            key=key_process,
            cards=cards,
            d_model=d_model,
            dropout_p=dropout_p
        )
        self.pack_att = PackAttention(key_pack, d_model, num_heads, dropout_p)
        keys = jax.random.split(key_set, n_set_layers)
        self.set_layers = [
            GraphLayer(
                key=subkey,
                dropout_p=dropout_p,
                d_model=d_model,
                num_heads=num_heads
            )
            for subkey in keys
        ]
        keys = jax.random.split(key_layers, n_layers)
        self.layers = [
            TransformerLayer(
                key=subkey,
                dropout_p=dropout_p,
                d_model=d_model,
                num_heads=num_heads
            )
            for subkey in keys
        ]
        self.predictor_head = eqx.nn.MLP(
            in_size=d_model,
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
        card_mask = jnp.arange(set_cards.shape[0])<sets.set_size[drafts.set_id]

        key, subkey = jax.random.split(key)
        h_cards_0 = self.process_cards(
            key=subkey,
            cards=cards,
            sets=sets,
            drafts=drafts,
            inference=inference
        )
        senders, receivers = rev_idx[sets.graph[drafts.set_id]]
        jax.debug.print("{x}", x=senders)

        keys = jax.random.split(subkey, len(self.set_layers))
        for subkey, layer in zip(keys, self.set_layers):
            h_cards_0 = layer(
                subkey,
                h_cards_0,
                senders,
                receivers,
                #mask=jnp.outer(card_mask, card_mask),
                inference=inference
            )

        packs = rev_idx[drafts.packs]
        picks = rev_idx[drafts.picks]
        key, subkey = jax.random.split(key)
        h_picks_0 = self.pack_att(
            key=subkey,
            query=h_cards_0[picks], 
            key_=h_cards_0[packs],
            value=h_cards_0[packs],
            mask=drafts.packs > 0,
            inference=inference
        )

        mask = jnp.tri(45)
        h_picks = h_picks_0
        keys = jax.random.split(subkey, len(self.layers))
        for subkey, layer in zip(keys, self.layers):
            h_picks = layer(subkey, h_picks, mask, inference)

        y = jax.vmap(self.predictor_head)(h_picks)

        return y, state
