import jax
import equinox as eqx
from jaxtyping import jaxtyped, Array, Int, Float, Real
from beartype import beartype as typechecker

class BatchedData(eqx.Module):
    def __getitem__(self, idx: Int[Array, "*bs"]) -> "Drafts":
        return jax.tree.map(lambda x: x[idx], self)


@jaxtyped(typechecker=typechecker)
class Cards(BatchedData):
    card_id: Int[Array, "n"]
    textual_features: Real[Array, "n d_t"]
    numeric_features: Float[Array, "n d_n"]

@jaxtyped(typechecker=typechecker)
class Drafts(BatchedData):
    set_id: Int[Array, "n"]
    packs: Int[Array, "n 45 15"] # TODO: A factor of 2 can be gained here
    picks: Int[Array, "n 45"]
    win_rate: Float[Array, "n"]
    player_wr: Float[Array, "n"]
    weight: Int[Array, "n"]

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
