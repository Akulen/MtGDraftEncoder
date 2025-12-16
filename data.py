import os
from gpu_management import set_gpus

if __name__ == "__main__":
    set_gpus(2, forcing=True)
    print(f"XLA_PYTHON_CLIENT_ALLOCATOR set to: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}")
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import argparse
from functools import cache, partial
import hashlib
import datetime
import humanfriendly
import json
import numpy as np
import pickle
import polars as pl
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Tuple
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Array, ArrayLike, Float, Int, PRNGKeyArray
from typing import Literal, Mapping, Optional
import equinox as eqx

import time

from models import DraftWRPredictor
from nlp import Gemma, NLPProcessor
from data_types import Cards, Sets, Drafts
from card_utils import (
    get_all_cards, make_oracle, fix_split_names, get_draft_data, get_card_stats
)
from training import Trainer

os.makedirs('data', exist_ok=True)
CSI = "\x1b[" #]

@cache
def get_set_config() -> Mapping[str, Mapping[str, Any]]:
    with open('data/set_config.json', 'r') as file:
        return json.load(file)

class DL17Lands(Dataset):
    def __init__(self, format='OTJ', include_ext=False, verbose=True):
        st = time.time()
        self.format = format
        self.all_cards = get_all_cards()
        self.verbose = verbose
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
        self.cards, self.drafts = self.collect_format(
            **set_config, include_ext=include_ext
        )
        self.pack_size = set_config.get('pack_size', 15)
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
        # so we drop the half cards, then rename the full to keep only the
        # first half
        split_cards = (
            df_cards.filter(pl.col('name').str.contains('//'))['name']
            .str.split(by=' // ')
        ).explode()
        df_cards = df_cards.filter(
            ~pl.col('name').is_in(split_cards.implode())
        )
        df_cards = df_cards.with_columns(
            name=pl.col('name').str.split(by=' // ').list.get(0)
        )
        # Drop duplicate basics
        # Stopped the assert as list cards can be duplicates
        # dup_rarities = df_cards.filter(
        #     pl.col('name').is_duplicated()
        # )['rarity'].unique()
        # assert(len(dup_rarities) == 1 and dup_rarities[0] == 'basic')
        df_cards = df_cards.unique(subset='name')

        oracle = make_oracle(df_cards)
        df_cards = df_cards.with_columns(
            oracle=oracle['oracle'],
            full_name=oracle['name']
        )
        df_cards = fix_split_names(df_cards, stlands, verbose=self.verbose)

        df_drafts = get_draft_data(df_cards, stlands, pack_size, pad_id)

        df_cards = get_card_stats(df_cards, stlands, include_ext)

        df_cards = df_cards.sort(by="id")
        df_drafts = df_drafts.sort(by="draft_id")

        return df_cards, df_drafts


def collate_drafts(
    set_id: int,
    dl: DL17Lands,
    time_offset: int=0,
    time_window: int=7,
    first_n: int=-1,
    pad_id: int=0,
    verbose: bool= False,
) -> Drafts:
    cache = f'drafts_{dl.format}'
    if time_offset > 0:
        cache += f'_skip-{time_offset}'
    cache += f'_{time_window}-days'
    if first_n > 0:
        cache += f'_first-{first_n}'
    if pad_id != 0:
        cache += f'_pad-{pad_id}'
    if os.path.exists(f'cache/{cache}.pickle'):
        return pickle.load(open(f'cache/{cache}.pickle', 'rb'))
    assert(len(dl.all_cards.filter(pl.col('id') == pad_id)) == 0)
    PERIOD = datetime.timedelta(days=time_window)
    draft_start = (dl.drafts
        .select(pl.col('draft_time'))
        .min()
        .collect()['draft_time']
        .str.to_datetime()[0]
    )
    last_draft = (dl.drafts
        .select(pl.col('draft_time'))
        .max()
        .collect()['draft_time']
        .str.to_datetime()[0]
    )
    first_draft = last_draft - PERIOD
    if verbose:
        print(
            f"All drafts:         {CSI}31m{draft_start}{CSI}0m"
            f" to {CSI}32m{last_draft}{CSI}0m"
        )
    OFFSET = datetime.timedelta(days=time_offset)
    first_draft = first_draft - OFFSET
    last_draft = last_draft - OFFSET
    if verbose:
        print(
            f"Picking drafts from {CSI}34m{first_draft}{CSI}0m"
            f" to {CSI}32m{last_draft}{CSI}0m"
        )
    if draft_start > first_draft + datetime.timedelta(days=1):
        print(f'{CSI}33mWarning: {CSI}34mDraft period starts ({CSI}32m{first_draft}{CSI}34m) significantly earlier than first draft ({CSI}32m{draft_start}{CSI}34m).{CSI}0m')
    n_picks = dl.pack_size * 3
    drafts = (dl.drafts
        .filter(pl.col('draft_time').str.to_datetime().is_between(
            first_draft, last_draft
        ))
        .sort(['pack_number', 'pick_number'])
        .group_by('draft_id').agg(
            pl.col('pick_id'),
            pl.col('pack')
              .alias('packs_ids'),
            pl.col('event_match_wins').max(),
            pl.col('event_match_losses').max(),
            pl.col('user_game_win_rate_bucket').mean()
              .alias('player_win_rate'),
            pl.col('user_n_games_bucket').max()
              .alias('weight'),
        )
        # .filter(pl.col('win_rate').is_not_nan())
        .with_columns(
            pl.col('pick_id').list.to_array(n_picks),
            pl.col('packs_ids').list.to_array(n_picks),
        )
    )
    n_incomplete = drafts.filter(
        (pl.col('event_match_wins') < 7) & (pl.col('event_match_losses') < 3)
    ).select(pl.len()).collect().item()
    assert(n_incomplete == 0)
    if first_n > 0:
        drafts = drafts.head(n=first_n)
    drafts = drafts.collect()
    if verbose:
        select_drafts = len(drafts)
        all_drafts = dl.drafts.select(pl.len()).collect().item() // n_picks
        print(
            f'{CSI}34m{select_drafts} drafts{CSI}0m out of '
            f'{CSI}31m{all_drafts}{CSI}0m '
            f'({100 * select_drafts // all_drafts}%)'
        )
    drafts = Drafts(
        set_id=jnp.full((len(drafts),), set_id, dtype=jnp.int16),
        packs=jnp.array(np.pad(
            drafts['packs_ids'].to_numpy(),
            ((0, 0), (0, 45 - n_picks), (0, 0)),
            constant_values=pad_id
        )),
        picks=jnp.array(np.pad(
            drafts['pick_id'].to_numpy(),
            ((0, 0), (0, 45 - n_picks)),
            constant_values=pad_id
        )),
        game_outcome=jnp.stack((
            drafts['event_match_wins'].to_numpy(),
            drafts['event_match_losses'].to_numpy()
        ), axis=1),
        # win_rate=jnp.array(drafts['win_rate'].to_numpy()),
        player_wr=jnp.array(drafts['player_win_rate'].to_numpy()),
        weight=jnp.array(drafts['weight'].to_numpy()),
    )
    pickle.dump(drafts, open(f'cache/{cache}.pickle', 'wb'))
    return drafts

def collate_cards(
    dt: DL17Lands, nlp_processor: NLPProcessor, use_meta: bool=False,
    prompt: Optional[str]=None
) -> Cards:
    card_ids = jnp.array(dt.cards['id'].to_numpy())
    return Cards(
        card_id=card_ids,
        textual_features=jnp.array(nlp_processor(
            dt.cards['oracle'].to_list(), prompt=prompt
        )),
        numeric_features=jnp.transpose(jnp.array([
            dt.cards[category]
            for category in [
                'opening_hand', 'drawn', 'tutored', 'deck', 'sideboard', 'GIH'
            ]
        ])) if use_meta else jnp.zeros((card_ids.shape[0], 0))
    )

def make_reverse_dict(l: Int[Array, "n"], offset=0) -> Int[Array, "m"]:
    assert(l.min() > 0)
    d = np.full(l.max() + 1, -1, dtype=jnp.int32)
    d[0] = 0
    d[l] = offset + np.arange(l.shape[0])
    return jnp.array(d)

def make_graph_knn(
    sims: Float[Array, "n n"], density: float=0.2
) -> Int[Array, "2 m"]:
    n = sims.shape[0]
    k = int((n-1) * density)
    U = []
    V = []
    for u in range(n):
        U.extend([u] * (k+1))
        V.extend(np.argsort(sims[u])[-k-1:])
    return jnp.stack([jnp.array(U), jnp.array(V)])

def make_graph_global(
    sims: Float[Array, "n n"], density: float=0.2
) -> Int[Array, "2 m"]:
    n = sims.shape[0]
    values = sims.flatten()
    values.sort()
    values = values[:-n]
    limit = values[-int(len(values)*density)]
    adj_matrix = sims > limit
    return jnp.argwhere(adj_matrix).T

def make_graph(
    sims: Float[Array, "n n"], density: float=0.1, local: bool=False
) -> Int[Array, "2 m"]:
    if local:
        return make_graph_knn(sims, density)
    return make_graph_global(sims, density)

class JaxDraftDataset:
    def __init__(
        self,
        dataloaders: List[DL17Lands],
        nlp_processor: NLPProcessor,
        time_offset: int=0,
        time_window: int=7,
        use_meta: bool=False,
        graph_density: float=0.1,
        graph_type: Literal['knn', 'global']='knn',
        pad_id: int=0,
        batch_size: int=32,
        shuffle: bool=True,
        seed: int=42,
        verbose: bool=False
    ):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        cards = []
        sets = []
        drafts = []
        offset = 1
        for i, dl in enumerate(dataloaders):
            st = time.time()
            cards.append(collate_cards(
                dl, nlp_processor, use_meta,
                prompt="task: Magic the Gathering card selection in a draft | card: "
            ))
            id_map = make_reverse_dict(cards[-1].card_id, offset)
            graph = jnp.zeros((1, 2, 0), jnp.int32)
            if isinstance(nlp_processor, Gemma):
                h = torch.from_numpy(np.array(cards[-1].textual_features))
                graph = make_graph(
                    nlp_processor.model.similarity(h, h).numpy(), #type: ignore
                    density=graph_density,
                    local=graph_type=='knn'
                ) + offset
            sets.append(Sets(
                card_ids=id_map[cards[-1].card_id],
                set_size=cards[-1].card_id.shape[0],
                pack_size=dl.pack_size,
                graph=graph
            ))
            cur_drafts = collate_drafts(
                i, dl, time_offset=time_offset, time_window=time_window,
                pad_id=pad_id, verbose=verbose
            )
            cur_drafts = eqx.tree_at(
                lambda d: d.packs,
                cur_drafts,
                id_map[cur_drafts.packs]
            )
            cur_drafts = eqx.tree_at(
                lambda d: d.picks,
                cur_drafts,
                id_map[cur_drafts.picks]
            )
            drafts.append(cur_drafts)
            offset += cards[-1].card_id.shape[0]
            if verbose:
                print(f"Collating {dl.format} took {time.time() - st:.6f}s")
        d_t = cards[-1].textual_features.shape[1]
        d_n = cards[-1].numeric_features.shape[1]
        cards.insert(0, Cards(
            card_id=jnp.array([0]),
            textual_features=jnp.zeros((1, d_t)),
            numeric_features=jnp.zeros((1, d_n), dtype=jnp.float32)
        ))
        max_set_size = max(s.set_size for s in sets)
        max_graph_size = max(s.graph.shape[1] for s in sets)
        for i in range(len(sets)):
            sets[i] = eqx.tree_at(
                lambda s: s.card_ids,
                sets[i],
                jnp.pad(
                    sets[i].card_ids,
                    ((0, max_set_size - sets[i].set_size),),
                    constant_values=pad_id
                )
            )
            sets[i] = eqx.tree_at(
                lambda s: s.graph,
                sets[i],
                jnp.pad(
                    sets[i].graph,
                    ((0, 0), (0, max_graph_size - sets[i].graph.shape[1])),
                    constant_values=pad_id
                )
            )
        self.cards = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            *cards
        )
        self.sets = jax.tree.map(
            lambda *x: jnp.stack(x),
            *sets
        )
        self.drafts = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            *drafts
        )
        assert(self.drafts.packs.min() == 0 and self.drafts.picks.min() == 0)
        n_samples = self.drafts.picks.shape[0]
        if 2 * n_samples < self.batch_size:
            raise ValueError(
                f"Batch size is too large for the dataset (n={n_samples})"
            )
        if n_samples % self.batch_size != 0:
            duplicate = self.rng.choice(
                jnp.arange(n_samples),
                size=self.batch_size - (n_samples % self.batch_size),
                replace=False
            )
            self.drafts = jax.tree.map(
                lambda leaf: jnp.concatenate((leaf, leaf[duplicate]), axis=0),
                self.drafts
            )

        self.card_bytes = sum(
            jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.cards))
        )
        self.draft_bytes = sum(
            jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.drafts))
        )
        if verbose:
            print(f'Cards use:  {humanfriendly.format_size(self.card_bytes):>9}')
            print(f'Drafts use: {humanfriendly.format_size(self.draft_bytes):>9}')

        self.shuffle = shuffle
        self.step_fn = None
        self.trainer = None
        self.compiled = False

    def n_steps(self) -> int:
        return self.drafts.picks.shape[0] // self.batch_size

    def shard_data(self, trainer: Optional[Trainer]=None):
        if trainer is not None:
            self.trainer = trainer
        if self.trainer is not None:
            self.cards, self.sets = self.trainer.shard_model(
                self.cards, self.sets
            )
            # self.drafts = trainer.shard_data(self.drafts)

    def set_step_function(self,
        static: DraftWRPredictor,
        step_fn: Callable[
            [
                DraftWRPredictor, eqx.nn.State, Any, Any,
                Cards, Sets, Drafts, PRNGKeyArray
            ],
            Tuple[DraftWRPredictor, eqx.nn.State, Any, Float[Array, "..."]]
        ]
    ):
        def foo(
            carry: Tuple[
                DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray, Cards, Sets
            ],
            batch: Drafts,
            static: DraftWRPredictor,
        ) -> Tuple[
            Tuple[DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray, Cards, Sets],
            Float[Array, "..."]
        ]:
            params, state, opt_state, lr_transform_state, key, cards, sets = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            model, state, opt_state, output = step_fn(
                model, state, opt_state, lr_transform_state,
                cards, sets, batch, subkey
            )
            params, _ = eqx.partition(model, eqx.is_array)
            return (params, state, opt_state, lr_transform_state, key, cards, sets), output
        self.step_fn = jax.jit(
            partial(
                foo, static=static,
            ),
            donate_argnums=(0, 1)
        )
        self.scan_fn = partial(jax.lax.scan, f=self.step_fn)
        self.compiled = False

    def precompile(
        self,
        params: DraftWRPredictor,
        state: eqx.nn.State,
        opt_state: Any, #optax.OptState,
        lr_transform_state: Any,
        key: PRNGKeyArray
    ):
        if self.compiled:
            return
        indices = (
            jnp.arange(self.drafts.picks.shape[0])
               .reshape((-1, self.batch_size))
        )
        drafts = self.drafts[indices]
        if self.trainer is not None:
            drafts = self.trainer.shard_data(drafts)
        self.scan_fn = jax.jit(self.scan_fn).trace( #type: ignore
            init=(params, state, opt_state, lr_transform_state, key, self.cards, self.sets),
            xs=drafts
        ).lower().compile()
        with open('tmp_hlo.txt', 'w') as f:
            for module in self.scan_fn.runtime_executable().hlo_modules():
                f.write(module.to_string())
        # assert False
        self.compiled = True

    @jaxtyped(typechecker=typechecker)
    def run_batches(
        self,
        params: DraftWRPredictor,
        state: eqx.nn.State,
        opt_state: Any, #optax.OptState,
        lr_transform_state: Any,
        key: PRNGKeyArray
    ) -> Tuple[
        DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray,
        Float[Array, "n_batches ..."]
    ]:
        if self.step_fn is None:
            raise ValueError("step_fn is not set")

        n_samples = self.drafts.picks.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = jnp.array(indices).reshape((-1, self.batch_size))

        drafts = self.drafts[indices]
        if self.trainer is not None:
            drafts = self.trainer.shard_data(drafts)

        (params, state, opt_state, lr_transform_state, key, _, _), outputs = self.scan_fn(
            init=(params, state, opt_state, lr_transform_state, key, self.cards, self.sets),
            xs=drafts
        )
        self.compiled = True
        return params, state, opt_state, lr_transform_state, key, outputs

    def to_serializable(self):
        def to_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        serial = {
            'scalar': {
                'batch_size': self.batch_size,
                'card_bytes': self.card_bytes,
                'draft_bytes': self.draft_bytes,
                'shuffle': self.shuffle,
            },
            'eqx_data': {
                'cards': jax.tree.map(to_host, self.cards),
                'drafts': jax.tree.map(to_host, self.drafts),
                'sets': jax.tree.map(to_host, self.sets),
            },
            'rng': self.rng.bit_generator.state
        }
        return serial

    @classmethod
    def from_serialized(cls, serial):
        """Create an instance quickly from serial (dict of numpy arrays)."""
        obj = object.__new__(cls)

        def from_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, np.ndarray):
                return jnp.array(x)
            return x

        for k, v in serial['scalar'].items():
            setattr(obj, k, v)
        obj.cards = jax.tree.map(from_host, serial['eqx_data']['cards'])
        obj.drafts = jax.tree.map(from_host, serial['eqx_data']['drafts'])
        obj.sets = jax.tree.map(from_host, serial['eqx_data']['sets'])
        bit_rng = np.random.PCG64()
        bit_rng.state = serial['rng']
        obj.rng = np.random.Generator(bit_rng)
        obj.step_fn = None
        obj.trainer = None
        obj.compiled = False
        return obj

def make_dataset_from_args(args: argparse.Namespace) -> Tuple[
    JaxDraftDataset, JaxDraftDataset, JaxDraftDataset
]:
    return make_datasets(
        train_set=args.train_set,
        val_set=args.val_set,
        test_set=args.test_set,
        temporal_split=args.temporal_split,
        time_window=args.time_window,
        use_meta=args.use_meta,
        graph_density=args.graph_density,
        graph_type=args.graph_type,
        batch_size=args.batch_size,
        seed=args.seed,
        encoder_name=args.encoder,
        verbose=args.verbose
    )

def make_datasets(
    train_set: List[str],
    val_set: List[str],
    test_set: List[str],
    temporal_split: bool=False,
    time_window: int=7,
    use_meta: bool=False,
    graph_density: float=0.1,
    graph_type: Literal['knn', 'global']='knn',
    batch_size: int=32,
    seed: int=42,
    encoder_name: str='gemma',
    verbose: bool=False,
    cache: bool=True
) -> Tuple[
    JaxDraftDataset, JaxDraftDataset, JaxDraftDataset
]:
    args = locals()
    hs = None
    if cache:
        hs = hashlib.sha256(
            json.dumps(args, sort_keys=True).encode('utf-8')
        ).hexdigest()
        if os.path.exists(f'cache/datasets/{hs}.pkl'):
            with open(f'cache/datasets/{hs}.pkl', 'rb') as f:
                _args, (data_train, data_val, data_test) = pickle.load(f)
                if args != _args:
                    raise ValueError('Cached arguments do not match')
                return (
                    JaxDraftDataset.from_serialized(data_train),
                    JaxDraftDataset.from_serialized(data_val),
                    JaxDraftDataset.from_serialized(data_test)
                )

    dataloaders_train = [
        DL17Lands(ext, verbose=verbose) for ext in train_set
    ]
    if temporal_split:
        dataloaders_test = dataloaders_train
        dataloaders_val = dataloaders_train
    else:
        dataloaders_test = [
            DL17Lands(ext, verbose=verbose) for ext in test_set
        ]
        dataloaders_val = [
            DL17Lands(ext, verbose=verbose) for ext in val_set
        ]

    match encoder_name:
        case 'gemma':
            from nlp import Gemma
            encoder = Gemma()
        case _:
            raise NotImplementedError(f'Unknown encoder {encoder_name}')

    data_train = JaxDraftDataset(
        dataloaders_train,
        encoder,
        time_offset=2*time_window if temporal_split else 0,
        time_window=time_window,
        use_meta=use_meta,
        graph_density=graph_density,
        graph_type=graph_type,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        verbose=verbose
    )
    data_val = JaxDraftDataset(
        dataloaders_val,
        encoder,
        time_offset=time_window if temporal_split else 0,
        time_window=time_window,
        use_meta=use_meta,
        graph_density=graph_density,
        graph_type=graph_type,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        verbose=verbose
    )
    data_test = JaxDraftDataset(
        dataloaders_test,
        encoder,
        time_window=time_window,
        use_meta=use_meta,
        graph_density=graph_density,
        graph_type=graph_type,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        verbose=verbose
    )
    # constant_bytes = sum(
    #     d.card_bytes + d.draft_bytes
    #     for d in [data_train, data_val, data_test]
    # ) 
    # jax.config.update(
    #     "jax_captured_constants_warn_bytes", int(constant_bytes*1.01)
    # )
    if cache:
        assert hs is not None
        os.makedirs('cache/datasets', exist_ok=True)
        with open(f'cache/datasets/{hs}.pkl', 'wb') as f:
            pickle.dump((args, (
                data_train.to_serializable(),
                data_val.to_serializable(),
                data_test.to_serializable()
            )), f)
    return data_train, data_val, data_test

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
