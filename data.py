import os
from gpu_management import set_gpus

if __name__ == "__main__":
    set_gpus(2, forcing=True)
    print(f"XLA_PYTHON_CLIENT_ALLOCATOR set to: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}")
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from functools import cache, partial
import datetime
import humanfriendly
import json
import numpy as np
import pickle
import polars as pl
from torch.utils.data import Dataset
from typing import Any, Callable, List, Tuple
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Array, Float, Int, PRNGKeyArray
from typing import Mapping
import equinox as eqx

import time

from models import DraftWRPredictor
from nlp import NLPProcessor
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
            (pl.col('event_match_wins').max() / (
                  pl.col('event_match_wins').max()
                + pl.col('event_match_losses').max()
            )).alias('win_rate'),
            pl.col('user_game_win_rate_bucket').mean()
              .alias('player_win_rate'),
            pl.col('user_n_games_bucket').max()
              .alias('weight'),
        )
        .filter(pl.col('win_rate').is_not_nan())
        .with_columns(
            pl.col('pick_id').list.to_array(n_picks),
            pl.col('packs_ids').list.to_array(n_picks),
        )
    )
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
        win_rate=jnp.array(drafts['win_rate'].to_numpy()),
        player_wr=jnp.array(drafts['player_win_rate'].to_numpy()),
        weight=jnp.array(drafts['weight'].to_numpy()),
    )
    pickle.dump(drafts, open(f'cache/{cache}.pickle', 'wb'))
    return drafts

def collate_cards(dt: DL17Lands, nlp_processor: NLPProcessor) -> Cards:
    return Cards(
        card_id=jnp.array(dt.cards['id'].to_numpy()),
        textual_features=jnp.array(nlp_processor(dt.cards['oracle'].to_list())),
        numeric_features=jnp.transpose(jnp.array([
            dt.cards[category]
            for category in [
                'opening_hand', 'drawn', 'tutored', 'deck', 'sideboard', 'GIH'
            ]
        ])),
    )

def make_reverse_dict(l: Int[Array, "n"], offset=0) -> Int[Array, "m"]:
    assert(l.min() > 0)
    d = np.full(l.max() + 1, -1, dtype=jnp.int32)
    d[0] = 0
    d[l] = offset + np.arange(l.shape[0])
    return jnp.array(d)

class JaxDraftDataset:
    def __init__(
        self,
        dataloaders: List[DL17Lands],
        nlp_processor: NLPProcessor,
        time_offset: int=0,
        time_window: int=7,
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
            cards.append(collate_cards(dl, nlp_processor))
            id_map = make_reverse_dict(cards[-1].card_id, offset)
            sets.append(Sets(
                card_ids=id_map[cards[-1].card_id],
                set_size=cards[-1].card_id.shape[0],
                pack_size=dl.pack_size
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

        if verbose:
            print(f'Cards use:  {humanfriendly.format_size(sum(
                jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.cards))
            )):>9}')
            print(f'Drafts use: {humanfriendly.format_size(sum(
                jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.drafts))
            )):>9}')

        self.shuffle = shuffle
        self.step_fn = None
        self.trainer = None
        self.compiled = False

    def n_steps(self) -> int:
        return self.drafts.picks.shape[0] // self.batch_size

    def shard_data(self, trainer: Trainer):
        self.trainer = trainer
        self.cards, self.sets = trainer.shard_model(
            self.cards, self.sets
        )
        self.drafts = trainer.shard_data(self.drafts)

    def set_step_function(self,
        static: DraftWRPredictor,
        step_fn: Callable[
            [
                DraftWRPredictor, eqx.nn.State, Any, Any,
                Cards, Sets, Drafts, PRNGKeyArray
            ],
            Tuple[DraftWRPredictor, eqx.nn.State, Any, Float[Array, ""]]
        ]
    ):
        def foo(
            carry: Tuple[
                DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray
            ],
            idx: Int[Array, "bs"],
            static: DraftWRPredictor,
            cards: Cards, sets: Sets, drafts: Drafts
        ) -> Tuple[
            Tuple[DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray],
            Float[Array, "1"]
        ]:
            params, state, opt_state, lr_transform_state, key = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            model, state, opt_state, output = step_fn(
                model, state, opt_state, lr_transform_state,
                cards, sets, drafts[idx], subkey
            )
            params, _ = eqx.partition(model, eqx.is_array)
            return (params, state, opt_state, lr_transform_state, key), output
        self.step_fn = partial(
            foo, static=static,
            cards=self.cards, sets=self.sets, drafts=self.drafts
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
        if self.trainer is not None:
            indices = self.trainer.shard_model(indices)
        self.scan_fn = jax.jit(self.scan_fn).trace( #type: ignore
            init=(params, state, opt_state, lr_transform_state, key),
            xs=indices
        ).lower().compile()
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
        DraftWRPredictor, eqx.nn.State, Any, PRNGKeyArray,
        Float[Array, "n_batches"]
    ]:
        if self.step_fn is None:
            raise ValueError("step_fn is not set")

        n_samples = self.drafts.picks.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = jnp.array(indices).reshape((-1, self.batch_size))
        if self.trainer is not None:
            indices = self.trainer.shard_model(indices)

        (params, state, opt_state, _, key), outputs = self.scan_fn(
            init=(params, state, opt_state, lr_transform_state, key),
            xs=indices
        )
        self.compiled = True
        return params, state, opt_state, key, outputs

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
