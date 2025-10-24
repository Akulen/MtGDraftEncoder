from collections import Counter
from functools import cache, partial
import json
import numpy as np
import os
import polars as pl
import re
import torch
from torch.utils.data import Dataset
from typing import (
    cast, Any, Callable, Generator, Generic, List, Optional, Tuple, TypeVar
)
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Mapping, NamedTuple

import time

from card_utils import (
    get_all_cards, make_oracle, get_draft_data, get_card_stats
)

os.makedirs('data', exist_ok=True)

@cache
def get_set_config() -> Mapping[str, Mapping[str, Any]]:
    with open('data/set_config.json', 'r') as file:
        return json.load(file)

class DL17Lands(Dataset):
    def __init__(self, format='OTJ', include_ext=False, verbose=True):
        st = time.time()
        self.all_cards = get_all_cards()
        if verbose:
            print(f"Loading {format}")
            print('-' * 11)
            print(f"Loading card data: {time.time() - st:.6f}s")
            st = time.time()

        try:
            set_config = get_set_config()[format]
        except KeyError:
            raise NotImplementedError(
                f"Format '{format}' is not (yet) implemented."
            )
        self.cards, self.drafts = self.collect_format(**set_config)
        if verbose:
            print(f"Making extension dataframe: {time.time() - st:.6f}s")
            print('#' * 50)

    def collect_format(
        self, stlands, expansions, exclude=None, special_guests=None,
        thelist=None, include_ext=False, pack_size=None, pad_id=0
    ):
        if exclude is None:
            exclude = []
        if special_guests is None:
            special_guests = []
        if thelist is None:
            thelist = []
        assert(len(self.all_cards.filter(pl.col('id') == pad_id)) == 0)

        # We drop cards that only appear in set or collector boosters
        df_cards = self.all_cards.filter(pl.col('is_booster') == True).filter(
            (
                pl.col('expansion').is_in(expansions)
                & ~pl.col('name').is_in(exclude)
            )
            | (
                (pl.col('expansion') == 'SPG')
                &  pl.col('name').is_in(special_guests)
            )
            | (
                pl.col('name').is_in(thelist)
            )
        )
        # Split cards are listed 3 times (once fully, then once for each half).
        # so we drop the half cards
        split_cards = (
            df_cards.filter(pl.col('name').str.contains('//'))['name']
            .str.split(by=' // ')
        ).explode()
        df_cards = df_cards.filter(
            ~pl.col('name').is_in(split_cards.implode())
        )
        # Drop duplicate basics
        # Stopped the assert as list cards can be duplicates
        # dup_rarities = df_cards.filter(
        #     pl.col('name').is_duplicated()
        # )['rarity'].unique()
        # assert(len(dup_rarities) == 1 and dup_rarities[0] == 'basic')
        df_cards = df_cards.unique(subset='name')

        df_cards = df_cards.with_columns(
            oracle=make_oracle(df_cards)
        )

        df_drafts = get_draft_data(df_cards, stlands, pack_size, pad_id)

        df_cards = get_card_stats(df_cards, stlands, include_ext)

        return df_cards, df_drafts

class simplestr(type):
    def __str__(self):
        return self.__name__

class SimpleTokenizer(metaclass=simplestr):
    def __init__(self, data=None, max_seq_length=150, device=None):
        self.max_seq_length = max_seq_length
        self.device = device
        if data is not None:
            self.build_vocab(data)

    # investigate vs investigates
    # Tokenizer and Vocabulary Building
    def tokenize(self, text):
        return re.findall(
            r'[a-zA-Z0-9+\-<>\']+'
            # symbols
            r'|{[a-zA-Z0-9]+}'
            # punctuation
            r'|[:.,()—/]|\\\\',
            text.lower()
        )

    # Build vocabulary from the training data
    def build_vocab(self, texts: pl.Series):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        self.vocab = {
            word: idx
            for idx, (word, _) in enumerate(counter.items(), start=2)
        }
        self.vocab["<unk>"] = 1  # Add unknown token
        self.vocab["<pad>"] = 0  # Add padding token

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, text, dtype=torch.int16):
        assert(torch.iinfo(dtype).max >= max(self.vocab.values()))
        tokens = self.tokenize(text)
        return torch.tensor(
            [
                [self.vocab.get(token, self.vocab["<unk>"]), 1]
                for token in tokens
            ] + [
                [self.vocab['<pad>'], 0]
            ] * (self.max_seq_length - len(tokens)),
            dtype=torch.int16
        ).transpose(-1, -2) \
         .to(self.device)


T = TypeVar('T', torch.Tensor, jnp.ndarray)

@jax.pmap
def device_select(x: T, i) -> T:
    return x[i]

class CardDataset(Dataset, Generic[T]):
    def __init__(
        self,
        oracle: List[str],
        target: torch.Tensor,
        weight: torch.Tensor,
        tokenizer: Callable[[str], torch.Tensor],
        use_torch: bool=False
    ):
        # tokenizer should return a tensor of shape (?, seq_len)
        # Currently Bert return tokens, pad_mask
        # But simpletokenizer returns tokens without padding
        # TODO:check if jnp works
        encodings = torch.stack(
            list(map(tokenizer, oracle))
        )
        def convert(x: torch.Tensor) -> T:
            if use_torch:
                return x # type: ignore
            return jnp.array(x) # type: ignore
        self.encodings: T = convert(encodings)
        self.target: T = convert(target)
        self.weight: T = convert(weight)
        self.n_devices: Optional[int] = None
        self.n_samples = encodings.shape[0]
        assert(self.n_samples == self.target.shape[0])
        assert(self.n_samples == self.weight.shape[0])

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return (
            self.encodings[idx],
            self.target[idx],
            self.weight[idx],
        )

    def to_devices(self, devices: List):
        assert(isinstance(self.encodings, jnp.ndarray))
        self.encodings = jax.device_put_replicated(self.encodings, devices)
        self.target = jax.device_put_replicated(self.target, devices)
        self.weight = jax.device_put_replicated(self.weight, devices)
        self.n_devices = len(devices)

    def batch(
        self,
        batch_size: int,
        shuffle: bool=True,
        seed: int=0
    ) -> Tuple[Callable[[], Generator[Tuple[T, T, T], None, None]], int]:
        assert(self.n_devices is None or batch_size % self.n_devices == 0)
        def generator() -> Generator[Tuple[T, T, T], None, None]:
            indices = np.arange(self.n_samples)
            if shuffle:
                rng = np.random.default_rng(seed)
                rng.shuffle(indices)
            if self.n_samples % batch_size != 0:
                # Add first few samples to the last batch so each batch has the
                # same size
                # TODO: It might be better to only add to be a multiple of
                # n_devices
                indices = np.concatenate([
                    indices,
                    indices[:batch_size-len(indices) % batch_size]
                ])

            # Create batches
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                x: T
                y: T
                z: T
                if self.n_devices is not None:
                    batch_indices = batch_indices.reshape((self.n_devices, -1))
                    x, y, z = map(partial(device_select, i=batch_indices),
                        (self.encodings, self.target, self.weight)
                    )
                else:
                    x = self.encodings[batch_indices]
                    y = self.target[batch_indices]
                    z = self.weight[batch_indices]
                yield (x, y, z)

        return generator, (self.n_samples + batch_size - 1) // batch_size

class TextGraph(NamedTuple):
    n_nodes: Int[Array, "batch_size"]
    # n_edges: Int[Array, "batch_size"]
    node_features: Int[Array, "n_nodes"]
    edge_features: Int[Array, "n_edges 59"]
    edges_u: Int[Array, "n_edges"]
    edges_v: Int[Array, "n_edges"]

class Graph(NamedTuple):
    n_nodes: Int[Array, "batch_size"]
    # n_edges: Int[Array, "batch_size"]
    node_features: Float[Array, "n_nodes d_model"]
    edge_features: Float[Array, "n_edges d_model"]
    edges_u: Int[Array, "n_edges"]
    edges_v: Int[Array, "n_edges"]

class JaxDraftDataset:
    def __init__(
        self,
        drafts: Mapping[str, Array],
        targets: Array,
        set_size: Int[Array, "1+n_sets"],
        pack_size: int=15,
        n_devices: int=1,
        device_batch_size: Optional[int]=None,
        batch_size: Optional[int]=None,
        shuffle: bool=True,
        seed: int=42
    ):
        if batch_size is None and device_batch_size is None:
            raise ValueError("batch_size or device_batch_size must be set")
        if batch_size is not None and device_batch_size is not None:
            raise ValueError(
                "batch_size and device_batch_size cannot be set together"
            )
        if device_batch_size is not None:
            self.batch_size = device_batch_size * n_devices
            self.device_batch_size = device_batch_size
        elif batch_size is not None:
            if batch_size % n_devices != 0:
                raise ValueError("batch_size must be a multiple of n_devices")
            self.batch_size = batch_size
            self.device_batch_size = batch_size // n_devices

        self.drafts = drafts
        # self.targets = drafts['set_id'].astype(jnp.float32)-1
        self.targets = targets
        self.set_size = set_size
        self.set_size_cum = set_size.cumsum()
        self.max_set_size = set_size.max().item()
        self.pack_size = pack_size
        self.n_devices = n_devices
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        def build_batch_nodes(
            padding_nodes: Int,
            sets: Int[Array, "batch_size"],
            n_nodes: Int,
            max_size: Int
        ) -> Int[Array, "n_nodes"]:
            # print("compiling", n_nodes, max_size)
            res = jnp.zeros(n_nodes+max_size, dtype=jnp.int32)

            def apply_set(
                carry: Tuple[Int, Int[Array, "res_size"]],
                set_id: Int
            ) -> Tuple[
                Tuple[Int, Int[Array, "res_size"]],
                None
            ]:
                pos, out = carry
                out = jax.lax.dynamic_update_slice(
                    out,
                    jnp.zeros(1, dtype=jnp.int32),
                    [pos]
                )
                out = jax.lax.dynamic_update_slice(
                    out,
                    jnp.arange(max_size) + self.set_size_cum[set_id-1],
                    [pos+1]
                )
                return ((pos + 1 + set_size[set_id], out), None)

            (_, res), _ = jax.lax.scan(apply_set, (padding_nodes, res), sets)

            return res[:n_nodes]

        def build_edges(
            set_id: Int,
            set_size: Int,
            pack_size: Int,
            picks_id: Int[Array, "3*pack_size"],
            packs_ids: Int[Array, "3*pack_size pack_size"],
            offset: Int
        ) -> Tuple[
            Int[Array, "max_set_size+draft_edges"],
            Int[Array, "max_set_size+draft_edges"],
            Int[Array, "max_set_size+draft_edges 4*pack_size-1"]
        ]:
            indexes = jnp.arange(self.max_set_size, dtype=jnp.int32)
            global_u = cast(Int[Array, "max_set_size"], jnp.where(
                indexes < set_size,
                offset, # Global node
                0       # Padding node
            ))
            global_v = cast(Int[Array, "max_set_size"], jnp.where(
                indexes < set_size,
                offset + 1 + indexes, # Card node
                0                     # Padding node
            ))
            global_edges = jnp.full(
                (self.max_set_size, 4*pack_size-1),
                0,
                dtype=jnp.int32
            )

            tri_i, tri_j = jnp.triu_indices(pack_size)
            packs_ids_tri = (
                packs_ids.reshape(
                            (3, pack_size, pack_size)
                        )[:, :, ::-1][:, tri_i, tri_j]
                         .ravel()
            )
            picks = (
                jnp.arange(3*pack_size)
                   .repeat(pack_size)
                   .reshape((3*pack_size, pack_size))
            )
            picks = (
                picks.reshape(
                        (3, pack_size, pack_size)
                    )[:, :, ::-1][:, tri_i, tri_j]
                     .ravel()
            )
            draft_u = cast(Int[Array, "draft_edges"], jnp.where(
                packs_ids_tri > 0,
                offset + 1 + packs_ids_tri - self.set_size_cum[set_id-1],
                0
            ))
            draft_v = cast(Int[Array, "draft_edges"], jnp.where(
                packs_ids_tri > 0,
                offset + 1 + picks_id[picks] - self.set_size_cum[set_id-1],
                0
            ))
            draft_edges = jnp.repeat(
                jnp.concat([
                    jnp.tril(jnp.tile(picks_id[:-1], (3*pack_size, 1)), k=-1),
                    packs_ids
                ], axis=1),
                jnp.tile(jnp.arange(pack_size)[::-1]+1, 3),
                axis=0,
                total_repeat_length=3*pack_size*(pack_size+1)//2
            )

            return (
                jnp.concat([global_u, draft_u]),
                jnp.concat([global_v, draft_v]),
                jnp.concat([global_edges, draft_edges], axis=0)
            )
        build_batch_edges = jax.vmap(build_edges, in_axes=(0, 0, None, 0, 0, 0))

        from beartype import beartype as typechecker
        from jaxtyping import jaxtyped

        @partial(jax.jit, static_argnums=(1, 3, 4))
        @partial(jax.vmap, in_axes=(0, None, 0, None, None))
        @jaxtyped(typechecker=typechecker)
        def build_graph(
            idxs: Int[Array, "batch_size"],
            n_nodes: int,
            # n_edges: Int,
            set_sizes: Int[Array, "batch_size"],
            max_size: int,
            pack_size: int=15
        ) -> Tuple[
            TextGraph,
            Float[Array, "batch_size"],
            Float[Array, "batch_size"]
        ]:
            padding_nodes = (
                  n_nodes
                - (1 + set_sizes).sum()
            )
            # padding_edges = (
            #       n_edges
            #     - set_sizes.sum()
            #     - device_batch_size * draft_edges
            # )

            n_node = jnp.concat([
                padding_nodes.reshape((1,)),
                1 + set_sizes
            ])
            # n_edge = jnp.concat([
            #     padding_edges.reshape((1,)),
            #     set_sizes + draft_edges
            # ])

            node_features = build_batch_nodes(
                padding_nodes,
                drafts['set_id'][idxs],
                n_nodes,
                max_size
            )
            edges_u, edges_v, edge_features = build_batch_edges(
                drafts['set_id'][idxs],
                set_size[drafts['set_id'][idxs]],
                pack_size,
                drafts['picks_id'][idxs],
                drafts['packs_ids'][idxs],
                n_node.cumsum()[:-1]
            )
            edges_u, edges_v = edges_u.ravel(), edges_v.ravel()
            edge_features = edge_features.reshape((-1, 4*pack_size-1))

            return (
                TextGraph(
                    n_nodes=n_node,
                    # n_edges=n_edge,
                    node_features=node_features,
                    edge_features=edge_features,
                    edges_u=edges_u,
                    edges_v=edges_v
                ),
                # (drafts['set_id'].astype(jnp.float32)-1)[idxs],
                self.targets[idxs],
                drafts['weight'][idxs]
            )
        self.build_graph = build_graph

    def make_batches(
        self
    ) -> Generator[Tuple[
        TextGraph,
        Float[Array, "n_devices batch_size"],
        Float[Array, "n_devices batch_size"]
    ], None, None]:
        n_drafts = self.drafts['set_id'].shape[0]
        n_drafts += (
            self.n_devices - n_drafts % self.n_devices
        ) % self.n_devices
        indices = np.arange(n_drafts) % self.drafts['set_id'].shape[0]
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = jnp.asarray(indices)

        # draft_edges = 3 * (pack_size * (pack_size + 1)) // 2
        # batch_edges = self.device_batch_size * (self.max_set_size+draft_edges)

        batch_size = self.n_devices * self.device_batch_size
        for start_idx in range(0, len(indices), batch_size):
            idxs = indices[start_idx:start_idx + batch_size]
            idxs = idxs.reshape((self.n_devices, -1))
            batch_nodes = 1 + idxs.shape[1] * (1+self.max_set_size)

            yield self.build_graph(
                idxs,
                batch_nodes,
                # batch_edges,
                self.set_size[self.drafts['set_id'][idxs]],
                self.max_set_size,
                self.pack_size
            )

class JaxDraftMultiDataset:
    def __init__(
        self,
        drafts: Mapping[str, Array],
        targets: Array,
        set_size: Int[Array, "1+n_sets"],
        pack_size: int=15,
        n_devices: int=1,
        device_batch_size: Optional[int]=None,
        batch_size: Optional[int]=None,
        shuffle: bool=True,
        seed: int=42
    ):
        if batch_size is None and device_batch_size is None:
            raise ValueError("batch_size or device_batch_size must be set")
        if batch_size is not None and device_batch_size is not None:
            raise ValueError(
                "batch_size and device_batch_size cannot be set together"
            )
        if device_batch_size is not None:
            self.batch_size = device_batch_size * n_devices
            self.device_batch_size = device_batch_size
        elif batch_size is not None:
            if batch_size % n_devices != 0:
                raise ValueError("batch_size must be a multiple of n_devices")
            self.batch_size = batch_size
            self.device_batch_size = batch_size // n_devices

        self.n_sets = drafts['set_id'].max()
        assert(drafts['set_id'].min() > 0)

        self.drafts = [
            {
                field: drafts[field][drafts['set_id'] == i_set+1]
                for field in drafts
            }
            for i_set in range(self.n_sets)
        ]
        self.targets = [
            targets[drafts['set_id'] == i_set+1]
            for i_set in range(self.n_sets)
        ]
        self.set_size = set_size
        self.set_size_cum = set_size.cumsum()
        self.max_set_size = set_size.max().item()
        self.pack_size = pack_size
        self.n_devices = n_devices
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        def build_edges(
            set_id: Int,
            set_size: Int,
            pack_size: Int,
            picks_id: Int[Array, "3*pack_size"],
            packs_ids: Int[Array, "3*pack_size pack_size"],
            offset: Int
        ) -> Tuple[
            Int[Array, "max_set_size+draft_edges"],
            Int[Array, "max_set_size+draft_edges"],
            Int[Array, "max_set_size+draft_edges 4*pack_size-1"]
        ]:
            indexes = jnp.arange(self.max_set_size, dtype=jnp.int32)
            global_u = cast(Int[Array, "max_set_size"], jnp.where(
                indexes < set_size,
                offset, # Global node
                0       # Padding node
            ))
            global_v = cast(Int[Array, "max_set_size"], jnp.where(
                indexes < set_size,
                offset + 1 + indexes, # Card node
                0                     # Padding node
            ))
            global_edges = jnp.full(
                (self.max_set_size, 4*pack_size-1),
                0,
                dtype=jnp.int32
            )

            tri_i, tri_j = jnp.triu_indices(pack_size)
            packs_ids_tri = (
                packs_ids.reshape(
                            (3, pack_size, pack_size)
                        )[:, :, ::-1][:, tri_i, tri_j]
                         .ravel()
            )
            picks = (
                jnp.arange(3*pack_size)
                   .repeat(pack_size)
                   .reshape((3*pack_size, pack_size))
            )
            picks = (
                picks.reshape(
                        (3, pack_size, pack_size)
                    )[:, :, ::-1][:, tri_i, tri_j]
                     .ravel()
            )
            draft_u = cast(Int[Array, "draft_edges"], jnp.where(
                packs_ids_tri > 0,
                offset + 1 + packs_ids_tri - self.set_size_cum[set_id-1],
                0
            ))
            draft_v = cast(Int[Array, "draft_edges"], jnp.where(
                packs_ids_tri > 0,
                offset + 1 + picks_id[picks] - self.set_size_cum[set_id-1],
                0
            ))
            draft_edges = jnp.repeat(
                jnp.concat([
                    jnp.tril(jnp.tile(picks_id[:-1], (3*pack_size, 1)), k=-1),
                    packs_ids
                ], axis=1),
                jnp.tile(jnp.arange(pack_size)[::-1]+1, 3),
                axis=0,
                total_repeat_length=3*pack_size*(pack_size+1)//2
            )

            return (
                jnp.concat([global_u, draft_u]),
                jnp.concat([global_v, draft_v]),
                jnp.concat([global_edges, draft_edges], axis=0)
            )
        build_batch_edges = jax.vmap(build_edges, in_axes=(None, None, None, 0, 0, 0))

        from beartype import beartype as typechecker
        from jaxtyping import jaxtyped

        @partial(jax.jit, static_argnums=(1, 2, 3))
        @partial(jax.vmap, in_axes=(0, None, None, None))
        @jaxtyped(typechecker=typechecker)
        def build_graph(
            idxs: Int[Array, "batch_size"],
            # n_nodes: int,
            # n_edges: Int,
            set_size: int,
            set_id: int,
            pack_size: int=15
        ) -> Tuple[
            Int[Array, 'n_cards'],
            TextGraph,
            Float[Array, "batch_size"],
            Float[Array, "batch_size"]
        ]:
            # print("compiling", n_nodes, max_size)

            # padding_nodes = (
            #       n_nodes
            #     - (1 + set_sizes).sum()
            # )
            # padding_edges = (
            #       n_edges
            #     - set_sizes.sum()
            #     - device_batch_size * draft_edges
            # )

            n = idxs.shape[0]
            n_node = jnp.array([1] + [1+set_size] * n, dtype=jnp.int32)

            node_features = jnp.concat([
                jnp.zeros(1, dtype=jnp.int32),
                jnp.tile(jnp.concat([
                    jnp.zeros(1, dtype=jnp.int32),
                    jnp.arange(set_size) + 1
                ]), n)
            ])
            edges_u, edges_v, edge_features = build_batch_edges(
                set_id,
                set_size,
                pack_size,
                self.drafts[set_id-1]['picks_id'][idxs],
                self.drafts[set_id-1]['packs_ids'][idxs],
                n_node.cumsum()[:-1]
            )
            edges_u, edges_v = edges_u.ravel(), edges_v.ravel()
            edge_features = edge_features.reshape((-1, 4*pack_size-1))
            edge_features = jnp.where(
                edge_features == 0,
                0,
                edge_features - self.set_size_cum[set_id-1] + 1
            )

            return (
                jnp.concat([
                    jnp.zeros(1, dtype=jnp.int32),
                    jnp.arange(set_size) + self.set_size_cum[set_id-1]
                ]),
                TextGraph(
                    n_nodes=n_node,
                    # n_edges=n_edge,
                    node_features=node_features,
                    edge_features=edge_features,
                    edges_u=edges_u,
                    edges_v=edges_v
                ),
                self.targets[set_id-1][idxs],
                self.drafts[set_id-1]['weight'][idxs]
            )
        self.build_graph = build_graph

    def make_set_batches(self, i_set, batch_size):
        n_drafts = self.drafts[i_set]['set_id'].shape[0]
        n_batches = (n_drafts + batch_size - 1) // batch_size
        extra_drafts = n_batches * batch_size - n_drafts
        indices = np.arange(n_drafts)
        self.rng.shuffle(indices)
        indices = np.concatenate([indices, indices[:extra_drafts]])
        if self.shuffle:
            self.rng.shuffle(indices)
        return jnp.asarray(indices).reshape((n_batches, batch_size))

    def n_batches(self):
        batch_size = self.n_devices * self.device_batch_size
        batches = [
            self.make_set_batches(i_set, batch_size)
            for i_set in range(self.n_sets)
        ]
        return sum([len(set_batches) for set_batches in batches])

    def make_batches(
        self
    ) -> Generator[Tuple[
        Int[Array, 'set_size+1'],
        TextGraph,
        Float[Array, "n_devices batch_size"],
        Float[Array, "n_devices batch_size"]
    ], None, None]:
        batch_size = self.n_devices * self.device_batch_size
        batches = [
            self.make_set_batches(i_set, batch_size)
            for i_set in range(self.n_sets)
        ]
        batch_order = sum([
            list(zip([i_set] * len(set_batches), range(len(set_batches))))
            for i_set, set_batches in enumerate(batches)
        ], [])
        if self.shuffle:
            self.rng.shuffle(batch_order)

        # draft_edges = 3 * (pack_size * (pack_size + 1)) // 2
        # batch_edges = self.device_batch_size * (self.max_set_size+draft_edges)

        for i_set, i_batch in batch_order:
            idxs = batches[i_set][i_batch]
            idxs = idxs.reshape((self.n_devices, -1))
            batch_nodes = 1 + idxs.shape[1] * (1+self.set_size[i_set+1])

            yield self.build_graph(
                idxs,
                # batch_nodes,
                # batch_edges,
                self.set_size[i_set+1].item(),
                i_set+1,
                self.pack_size
            )

def train_test_split(df, test_size=0.2, seed=0):
    return df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=seed)
        .gt(pl.len() * test_size)
        .alias('split')
    ).partition_by('split', include_key=False)

def test_set(ext):
    print("#" * 50)
    print(f"Test {ext}")
    print("=" * 10)
    dataloader= DL17Lands(ext, verbose=True)
    print(dataloader.all_cards.shape)
    print(dataloader.cards.shape)
    with pl.Config(tbl_cols=-1):
        print(dataloader.cards.head())
        print(dataloader.drafts.collect().shape)
        print(dataloader.drafts.head().collect())
    print(dataloader.cards.estimated_size())
    print(dataloader.drafts.collect().estimated_size())

def main():
    test_set('ONE')
    test_set('MOM')
    test_set('WOE')
    test_set('LCI')
    test_set('MKM')
    test_set('OTJ')
    test_set('BLB')
    test_set('DSK')
    test_set('FDN')
    test_set('DFT')

    print("#" * 50)

if __name__ == "__main__":
    main()
