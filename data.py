from collections import Counter
from functools import partial
import json
import numpy as np
import os
import polars as pl
import re
import requests
import torch
from torch.utils.data import Dataset
from typing import cast, List
import jax.numpy as jnp

import time

from replay_dtypes import get_dtypes

os.makedirs('data', exist_ok=True)

def full_oracle_polars(df):
    df = df.with_columns(
        faces_oracle=(
            pl.when(pl.col('oracle_text').is_null())
            .then(pl.concat_str([
                pl.col('card_faces').list.get(0).struct.field('oracle_text'),
                pl.col('card_faces').list.get(1).struct.field('oracle_text'),
            ], separator=' // '))
            .otherwise(pl.col('oracle_text'))
        ),
        power_box=(
            pl.when(~pl.col('power').is_null())
            .then(pl.concat_str([
                pl.col('power'),
                pl.lit('/'),
                pl.col('toughness'),
            ]))
            .when(~pl.col('loyalty').is_null())
            .then(pl.col('loyalty'))
            .otherwise(pl.lit(''))
        ),
    )
    df = df.with_columns(oracle=(
          '<name> '
        + pl.col('name')
        + '\n<cost> '
        + pl.col('mana_cost')
        + '\n<type> '
        + pl.col('type_line')
        + '\n<oracle> '
        + pl.col('faces_oracle').str.replace_all('\n', ' <nl> ')
        + '\n<pwl> '
        + pl.col('power_box')
    ))
    return df

all_cards = None
def get_all_card():
    global all_cards
    if all_cards is None:
        filename = 'data/cards_mtga.csv'
        if not os.path.isfile(filename):
            url = (
                'https://17lands-public.s3.amazonaws.com/'
                'analysis_data/cards/cards.csv'
            )
            response = requests.get(url)
            with open('data/cards_mtga.csv', 'wb') as file:
                file.write(response.content)
        dtypes = get_dtypes(filename=filename)
        all_cards = pl.read_csv(filename, schema=dtypes)
    return all_cards

oracle_data = None
def get_oracle():
    global oracle_data
    if oracle_data is None:
        if not os.path.isfile('data/cards_scryfall.parquet'):
            if not os.path.isfile('data/cards_scryfall.json'):
                url = 'https://api.scryfall.com/bulk-data/oracle-cards'
                response = requests.get(url)
                data = response.json()
                response = requests.get(data['download_uri'])
                scryfall_oracle = response.json()
                with open('data/cards_scryfall.json', 'w') as file:
                    json.dump(scryfall_oracle, file)
            oracle_data = pl.read_json(
                'data/cards_scryfall.json',
                infer_schema_length=None
            )
            print(oracle_data.columns)
            # Drop all sets with cards that can have duplicate names
            oracle_data = oracle_data.filter(~pl.col('set').is_in([
                'ugl', 'ust', 'unf', 'cmb2'
            ]))
            # Drop all tokens
            oracle_data = oracle_data.filter(~pl.col('set_type').is_in([
                'token', 'memorabilia'
            ]))
            oracle_data.write_parquet(
                'data/cards_scryfall.parquet',
                use_pyarrow=True,
            )
        else:
            oracle_data = pl.read_parquet('data/cards_scryfall.parquet')
    return oracle_data

def get_draft_data(df_cards, ext):
    filename = f'data/draft_data_public.{ext}.PremierDraft'
    if not os.path.isfile(f"{filename}.compact.parquet"):
        url = (
            'https://17lands-public.s3.amazonaws.com/analysis_data/'
            f"draft_{filename}.csv.gz"
        )
        if not os.path.isfile(f"{filename}.parquet"):
            if not os.path.isfile(f"{filename}.csv.gz"):
                response = requests.get(url)
                with open(f"{filename}.csv.gz", 'wb') as file:
                    file.write(response.content)

            dtypes = get_dtypes(filename=f"{filename}.csv.gz")
            df = pl.read_csv(f"{filename}.csv.gz", schema=dtypes)
            df.write_parquet(f"{filename}.parquet", use_pyarrow=True)
        else:
            df = pl.read_parquet(f"{filename}.parquet")

        cards = {}
        for column in df.columns:
            if column[:5] == 'pool_':
                name = column[5:]
                cards[name] = df_cards.filter(pl.col('name')==name)['id'].item()
        df = df.with_columns(
            pool=pl.concat_list([
                pl.lit(card_id).repeat_by(pl.col(f'pool_{card_name}'))
                for card_name, card_id in cards.items()
            ]),
            pack=pl.concat_list([
                pl.lit(card_id).repeat_by(pl.col(f'pack_card_{card_name}'))
                for card_name, card_id in cards.items()
            ])
        ).drop([
            'event_type',
            'draft_time',
        ] + [
            f'{prefix}_{card_name}'
            for card_name, _ in cards.items()
            for prefix in ['pool', 'pack_card']
        ])
        df.write_parquet(
            f"{filename}.compact.parquet",
            use_pyarrow=True,
        )
    else:
        df = pl.read_parquet(f"{filename}.compact.parquet")
    return df

def card_stats(
    card: str, types: List[str], df_games: pl.DataFrame
) -> float:
    col_sum = cast(pl.Series, sum(df_games[f'{tp}_{card}'] for tp in types))

    return (
        (col_sum * df_games['won']).sum()
        /
        max(1, col_sum.sum())
    )

def get_card_stats(df_cards, ext, include_ext=False):
    filename = f'data/game_data_public.{ext}.PremierDraft'
    stat_cols = {
        'opening_hand': ['opening_hand'],
        'drawn': ['drawn'],
        'tutored': ['tutored'],
        'deck': ['deck'],
        'sideboard': ['sideboard'],
        'GIH': ['opening_hand', 'drawn'],
    }
    if not os.path.isfile(f"{filename}.parquet"):
        if not os.path.isfile(f"{filename}.csv.gz"):
            url = (
                'https://17lands-public.s3.amazonaws.com/analysis_data/'
                f"game_{filename}.csv.gz"
            )
            response = requests.get(url)
            with open(f"{filename}.csv.gz", 'wb') as file:
                file.write(response.content)
        dtypes = get_dtypes(filename=f"{filename}.csv.gz")
        df_games = pl.read_csv(f"{filename}.csv.gz", schema=dtypes)
        df_cards = df_cards.with_columns(*[
            pl.col('name').map_elements(
                partial(card_stats, types=types, df_games=df_games),
                return_dtype=pl.Float32
            ).alias(f"{ext}_{name}")
            for name, types in stat_cols.items()
        ])
        df_cards = df_cards.with_columns(
            weight=pl.col('name').map_elements(
                lambda x: df_games[f'deck_{x}'].sum(),
                return_dtype=pl.Int32
            )
        )
        df_cards.select(
            pl.col('name', *[
                f"{ext}_{name}"
                for name in stat_cols
            ], 'weight')
        ).write_parquet(
            f"{filename}.parquet",
            use_pyarrow=True,
        )
    else:
        df_stats = pl.read_parquet(f"{filename}.parquet")
        c, cs = set(df_cards['name']), set(df_stats['name'])
        if len(c) != len(cs) or len(c - cs) > 0:
            raise Exception(f'Game card stats outdated for {ext}')
        df_cards = df_cards.join(df_stats, on='name', how='left')
    if not include_ext:
        df_cards = df_cards.rename({
            f'{ext}_{name}': name
            for name in stat_cols
        })
    return df_cards

class DL17Lands(Dataset):
    def __init__(self, format='OTJ', include_ext=False, verbose=True):
        st = time.time()
        self.all_cards = get_all_card()
        if verbose:
            print(f"Loading {format}")
            print('-' * 11)
            print(f"Loading card data: {time.time() - st:.6f}s")
            st = time.time()
        self.oracle_data = get_oracle()
        if verbose:
            print(f"Loading oracle data: {time.time() - st:.6f}s")
            st = time.time()
        if format == 'OTJ':
            self.cards, self.drafts = self.OTJ(include_ext)
        elif format == 'MKM':
            self.cards, self.drafts = self.MKM(include_ext)
        else:
            raise NotImplementedError
        if verbose:
            print(f"Making extension dataframe: {time.time() - st:.6f}s")
            print('#' * 50)

    def collect_format(
        self, stlands, expansions, exclude=None, special_guests=None,
        thelist=None, include_ext=False
    ):
        if exclude is None:
            exclude = []
        if special_guests is None:
            special_guests = []
        if thelist is None:
            thelist = []
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
            ~pl.col('name').is_in(split_cards)
        )
        # Drop duplicate basics
        # Stopped the assert as list cards can be duplicates
        # dup_rarities = df_cards.filter(
        #     pl.col('name').is_duplicated()
        # )['rarity'].unique()
        # assert(len(dup_rarities) == 1 and dup_rarities[0] == 'basic')
        df_cards = df_cards.unique(subset='name')

        entries = df_cards.with_columns(pl.col('name').replace({
            'Brazen Borrower': 'Brazen Borrower // Petty Theft',
            'Kellan, Inquisitive Prodigy':
                'Kellan, Inquisitive Prodigy // Tail the Suspect',
        }))
        entries = entries.join(self.oracle_data, on='name', how='left')
        df_cards = df_cards.with_columns(
            oracle=full_oracle_polars(entries)['oracle']
        )
        assert(len(df_cards.filter(pl.col('oracle').is_null())) == 0)

        df_drafts = get_draft_data(df_cards, stlands)

        df_cards = get_card_stats(df_cards, stlands, include_ext)

        return df_cards, df_drafts

    def OTJ(self, include_ext=False):
        return self.collect_format(
            stlands='OTJ',
            expansions=['OTJ', 'OTP', 'BIG'],
            special_guests=[
                'Stoneforge Mystic',
                'Brazen Borrower',
                'Desertion',
                'Morbid Opportunist',
                'Port Razer',
                'Scapeshift',
                'Mystic Snake',
                'Notion Thief',
                'Desert',
                'Prismatic Vista'
            ],
            include_ext=include_ext
        )

    def MKM(self, include_ext=False):
        return self.collect_format(
            stlands='MKM',
            expansions=['MKM'],
            exclude=[
                'Tomik, Wielder of Law',
                'Melek, Reforged Researcher',
                'Voja, Jaws of the Conclave',
                # 'Kellan, Inquisitive Prodigy',
                # 'Tail the Suspect',
            ],
            special_guests=[
                'Crashing Footfalls',
                'Drown in the Loch',
                'Fabricate',
                'Field of the Dead',
                'Gamble',
                'Ghostly Prison',
                'Show and Tell',
                'Tireless Tracker',
                'Tragic Slip',
                'Victimize',
            ],
            thelist=[
                "Smuggler's Copter",
                'Bishop of the Bloodstained',
                'Burden of Guilt',
                'Evolutionary Leap',
                'Combine Chrysalis',
                'Consign // Oblivion',
                'Possibility Storm',
                'Duskmantle, House of Shadow',
                'Enlisted Wurm',
                'Ghost Quarter',
                'Gnaw to the Bone',
                'Goblin Warchief',
                'Hard Evidence',
                'High Alert',
                'Ixidor, Reality Sculptor',
                'Jace, Wielder of Mysteries',
                'Krosan Tusker',
                'Kuldotha Rebirth',
                'Laid to Rest',
                'Leonin Relic-Warder',
                'Magmaw',
                'Mass Hysteria',
                'Maverick Thopterist',
                'Mentor of the Meek',
                "Metalspinner's Puzzleknot",
                'Millstone',
                'Mistveil Plains',
                'Molten Psyche',
                'Monologue Tax',
                'Mystery Key',
                'Nyx Weaver',
                'Putrid Warrior',
                'Quintorius, Field Historian',
                'Ranger-Captain of Eos',
                'Shard of Broken Glass',
                'Spell Snare',
                'Stromkirk Captain',
                'Syr Konrad, the Grim',
                'Treacherous Terrain',
                'Worldspine Wurm',
            ],
            include_ext=include_ext
        )

class SimpleTokenizer:
    def __init__(self, data=None):
        if data:
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
            for idx, (word, _) in enumerate(counter.items(), start=1)
        }
        self.vocab["<unk>"] = 0  # Add unknown token

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, text):
        tokens = self.tokenize(text)
        return torch.tensor([[
            self.vocab.get(token, self.vocab["<unk>"]) for token in tokens
        ]])

from typing import Tuple, Generator
class CardDataset(Dataset):
    def __init__(
        self,
        oracle: List[str],
        target,
        weight,
        tokenizer,
        use_torch=False
    ):
        # tokenizer should return a tensor of shape (?, seq_len)
        # Currently Bert return tokens, pad_mask
        # But simpletokenizer returns tokens without padding
        # TODO:check if jnp works
        self.encodings = (torch if use_torch else jnp).stack(
            list(map(tokenizer, oracle))
        )
        self.target = target
        self.weight = weight
        assert(len(self.encodings) == len(self.target))
        assert(len(self.encodings) == len(self.weight))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return (
            self.encodings[idx],
            self.target[idx],
            self.weight[idx],
        )

    def batch(self, batch_size, shuffle=True, seed=None):
        def generator() -> Generator[Tuple[torch.Tensor, ...], None, None]:
            indices = np.arange(len(self.encodings))
            if shuffle:
                rng = np.random.default_rng(seed)
                rng.shuffle(indices)

            # Create batches
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                yield (
                    self.encodings[batch_indices],
                    self.target[batch_indices],
                    self.weight[batch_indices]
                )

        return generator, (len(self.encodings) + batch_size - 1) // batch_size

def train_test_split(df, test_size=0.2, seed=0):
    return df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=seed)
        .gt(pl.len() * test_size)
        .alias('split')
    ).partition_by('split', include_key=False)
