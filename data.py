import os
import requests
from torch.utils.data import Dataset
import polars as pl
import json

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
            ], separator=" // "))
            .otherwise(pl.col('oracle_text'))
        ),
        power_box=(
            pl.when(~pl.col('power').is_null())
            .then(pl.concat_str([
                pl.lit('\n'),
                pl.col('power'),
                pl.col('toughness'),
            ], separator="/"))
            .when(~pl.col('loyalty').is_null())
            .then(pl.concat_str([
                pl.lit('\n'),
                pl.col('loyalty')
            ]))
            .otherwise(pl.lit('\n'))
        ),
    )
    df = df.with_columns(oracle=(
          pl.col('name')
        + "\n"
        + pl.col('mana_cost')
        + "\n"
        + pl.col('type_line')
        + '\n'
        + pl.col('faces_oracle')
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
                "https://17lands-public.s3.amazonaws.com/"
                "analysis_data/cards/cards.csv"
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
                url = "https://api.scryfall.com/bulk-data/oracle-cards"
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
            "https://17lands-public.s3.amazonaws.com/analysis_data/"
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
            if column[:5] == "pool_":
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

class DL17Lands(Dataset):
    def __init__(self, format="OTJ"):
        st = time.time()
        self.all_cards = get_all_card()
        print(f"Loading card data: {time.time() - st:.6f}s")
        st = time.time()
        self.oracle_data = get_oracle()
        print(f"Loading oracle data: {time.time() - st:.6f}s")
        st = time.time()
        if format == "OTJ":
            self.cards, self.drafts = self.OTJ()
        elif format == "MKM":
            self.cards, self.drafts = self.MKM()
        else:
            raise NotImplementedError
        print(f"Making extension dataframe: {time.time() - st:.6f}s")
        # print(self.cards.shape)
        # print(self.cards.head())

    def collect_format(
        self, stlands, expansions, exclude=None, special_guests=None,
        thelist=None
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

        return df_cards, df_drafts

    def OTJ(self):
        return self.collect_format(
            stlands="OTJ",
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
            ]
        )

    def MKM(self):
        return self.collect_format(
            stlands="MKM",
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
                "Bishop of the Bloodstained",
                "Burden of Guilt",
                "Evolutionary Leap",
                "Combine Chrysalis",
                "Consign // Oblivion",
                "Possibility Storm",
                "Duskmantle, House of Shadow",
                "Enlisted Wurm",
                "Ghost Quarter",
                "Gnaw to the Bone",
                "Goblin Warchief",
                "Hard Evidence",
                "High Alert",
                "Ixidor, Reality Sculptor",
                "Jace, Wielder of Mysteries",
                "Krosan Tusker",
                "Kuldotha Rebirth",
                "Laid to Rest",
                "Leonin Relic-Warder",
                "Magmaw",
                "Mass Hysteria",
                "Maverick Thopterist",
                "Mentor of the Meek",
                "Metalspinner's Puzzleknot",
                "Millstone",
                "Mistveil Plains",
                "Molten Psyche",
                "Monologue Tax",
                "Mystery Key",
                "Nyx Weaver",
                "Putrid Warrior",
                "Quintorius, Field Historian",
                "Ranger-Captain of Eos",
                "Shard of Broken Glass",
                "Spell Snare",
                "Stromkirk Captain",
                "Syr Konrad, the Grim",
                "Treacherous Terrain",
                "Worldspine Wurm",
            ]
        )
