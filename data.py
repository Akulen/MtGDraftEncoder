import os
from gpu_management import set_gpus

if __name__ == "__main__":
    set_gpus(2, forcing=True)
    print(f"XLA_PYTHON_CLIENT_ALLOCATOR set to: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}")
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from functools import cache
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
import optax
import equinox as eqx

import time

from data_types import Cards, Drafts
from card_utils import (
    get_all_cards, make_oracle, get_draft_data, get_card_stats
)

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


ONE_WEEK = datetime.timedelta(days=7)
def collate_drafts(
    set_id: int,
    dl: DL17Lands,
    s2l_week: bool=False,
    first_n: int=-1,
    pad_id: int=0,
    verbose: bool= False,
) -> Drafts:
    cache = f'drafts_{dl.format}'
    if s2l_week:
        cache += '_s2l'
    if first_n > 0:
        cache += f'_first-{first_n}'
    if pad_id != 0:
        cache += f'_pad-{pad_id}'
    if os.path.exists(f'cache/{cache}.pickle'):
        return pickle.load(open(f'cache/{cache}.pickle', 'rb'))
    assert(len(dl.all_cards.filter(pl.col('id') == pad_id)) == 0)
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
    first_draft = last_draft - ONE_WEEK
    if verbose:
        print(
            f"All drafts:         {CSI}31m{draft_start}{CSI}0m"
            f" to {CSI}32m{last_draft}{CSI}0m"
        )
    if s2l_week:
        first_draft, last_draft = first_draft - ONE_WEEK, first_draft
    if verbose:
        print(
            f"Picking drafts from {CSI}34m{first_draft}{CSI}0m"
            f" to {CSI}32m{last_draft}{CSI}0m"
        )
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

def collate_cards(dt: DL17Lands, nlp_processor) -> Cards:
    return Cards(
        card_id=jnp.array(dt.cards['id'].to_numpy()),
        textual_features=jnp.array(nlp_processor(dt.cards['oracle'])),
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
        nlp_processor: Callable[[str], np.ndarray],
        s2l_week: bool=False,
        pad_id: int=0,
        batch_size: int=32,
        shuffle: bool=True,
        seed: int=42,
        verbose: bool=False
    ):
        self.batch_size = batch_size

        cards = []
        drafts = []
        offset = 1
        for i, dl in enumerate(dataloaders):
            st = time.time()
            cards.append(collate_cards(dl, nlp_processor))
            id_map = make_reverse_dict(cards[-1].card_id, offset)
            cur_drafts = collate_drafts(
                i+1, dl, s2l_week=s2l_week, pad_id=pad_id, verbose=verbose
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
        self.cards = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            *cards
        )
        self.drafts = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            *drafts
        )
        assert(self.drafts.packs.min() == 0 and self.drafts.picks.min() == 0)
        if verbose:
            print(f'Cards use:  {humanfriendly.format_size(sum(
                jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.cards))
            )):>9}')
            print(f'Drafts use: {humanfriendly.format_size(sum(
                jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.drafts))
            )):>9}')

        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    @jaxtyped(typechecker=typechecker)
    def run_batches(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: Any, #optax.OptState,
        step_fn: Callable[
            [
                eqx.Module, eqx.nn.State, Any,
                Cards, Drafts, PRNGKeyArray
            ],
            Tuple[eqx.Module, eqx.nn.State, Any, Float[Array, ""]]
        ],
        key: PRNGKeyArray
    ) -> Tuple[
        eqx.Module, eqx.nn.State, Any, PRNGKeyArray,
        Float[Array, "n_batches"]
    ]:
        n_samples = self.drafts.picks.shape[0]
        if 2 * n_samples < self.batch_size:
            raise ValueError(
                f"Batch size is too large for the dataset (n={n_samples})"
            )
        indices = np.arange(n_samples)
        if n_samples % self.batch_size != 0:
            indices = np.concat([
                indices,
                self.rng.choice(
                    indices,
                    size=self.batch_size - (n_samples % self.batch_size),
                    replace=False
                )
            ])
        if self.shuffle:
            self.rng.shuffle(indices)
        indices = jnp.array(indices).reshape((-1, self.batch_size))

        def foo(
            carry: Tuple[
                eqx.Module, eqx.nn.State, Any, PRNGKeyArray
            ],
            idx: Int[Array, "bs"]
        ) -> Tuple[
            Tuple[eqx.Module, eqx.nn.State, Any, PRNGKeyArray],
            Float[Array, "1"]
        ]:
            model, state, opt_state, key = carry
            key, subkey = jax.random.split(key)
            model, state, opt_state, output = step_fn(
                model, state, opt_state,
                self.cards, self.drafts[idx], subkey
            )
            return (model, state, opt_state, key), output

        (model, state, opt_state, key), outputs = jax.lax.scan(
            foo,
            (model, state, opt_state, key),
            indices
        )
        return model, state, opt_state, key, outputs

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
