import jax
import equinox as eqx
from jaxtyping import Array, Bool, Int, Float, Real

class BatchedData(eqx.Module):
    def __getitem__(self, idx: Int[Array, "*bs"]) -> "Drafts":
        return jax.tree.map(lambda x: x[idx], self)


class Cards(BatchedData):
    card_id: Int[Array, "n_cards"]
    textual_features: Real[Array, "n_cards d_t"]
    numeric_features: Float[Array, "n_cards d_n"]

class Sets(BatchedData):
    card_ids: Int[Array, "n_sets set_size_max"]
    set_size: Int[Array, "n_sets"]
    pack_size: Int[Array, "n_sets"]
    graph: Int[Array, "n_sets 2 n_edges"]
    adjacency: Bool[Array, "n_sets n_nodes n_nodes"]

class Drafts(BatchedData):
    set_id: Int[Array, "n_drafts"]
    packs: Int[Array, "n_drafts 45 15"] # TODO: A factor of 2 can be gained here
    picks: Int[Array, "n_drafts 45"]
    game_outcome: Int[Array, "n_drafts 2"]
    rank: Int[Array, ""]
    player_wr: Float[Array, "n_drafts"]
    weight: Int[Array, "n_drafts"]

# Deprecated
# class TextGraph(NamedTuple):
#     n_nodes: Int[Array, "batch_size"]
#     # n_edges: Int[Array, "batch_size"]
#     node_features: Int[Array, "n_nodes"]
#     edge_features: Int[Array, "n_edges 59"]
#     edges_u: Int[Array, "n_edges"]
#     edges_v: Int[Array, "n_edges"]
#
# class Graph(NamedTuple):
#     n_nodes: Int[Array, "batch_size"]
#     # n_edges: Int[Array, "batch_size"]
#     node_features: Float[Array, "n_nodes d_model"]
#     edge_features: Float[Array, "n_edges d_model"]
#     edges_u: Int[Array, "n_edges"]
#     edges_v: Int[Array, "n_edges"]
