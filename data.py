import os
from torch.utils.data import Dataset
import pandas as pd

all_cards = None
def get_all_card():
    global all_cards
    if all_cards is None:
        all_cards = pd.read_csv('data/cards.csv')
    return all_cards

def load_parquet(file):
    if os.path.isfile(file + ".parquet.gzip"):
        return pd.read_parquet(file + ".parquet.gzip")
    else:
        data = pd.read_csv(file + ".csv")
        data.to_parquet(file + ".parquet.gzip", compression="gzip")
        return data

class DL17Lands(Dataset):
    def __init__(self, format="OTJ"):
        self.all_cards = get_all_card()
        if format == "OTJ":
            self.cards = self.OTJ()
        elif format == "MKM":
            self.cards = self.MKM()
        else:
            raise NotImplementedError
        print(self.cards.shape)
        print(self.cards.head())

    def collect_format(self, expansions, exclude=None, special_guests=None):
        if exclude is None:
            exclude = []
        if special_guests is None:
            special_guests = []
        df = self.all_cards.loc[
            (
                self.all_cards['expansion'].isin(expansions)
                & ~self.all_cards['name'].isin(exclude)
            )
            | (
                (self.all_cards['expansion'] == 'SPG')
                &  self.all_cards['name'].isin(special_guests)
            )
        ]
        # Split cards are listed 3 times (once fully, then once for each half).
        # so we drop the half cards
        df_tmp = pd.DataFrame()
        df_tmp[['part1', 'part2']] = (
            df[df['name'].str.contains('//')]['name']
            .str.split(' // ', expand=True)
        )
        df = df.loc[
            ~df['name'].isin(
                df_tmp['part1'].tolist() + df_tmp['part2'].tolist()
            )
        ]
        df = df.drop_duplicates(subset='name').copy()

        return df

    def OTJ(self):
        return self.collect_format(
            expansions=['OTJ', 'OTP', 'BIG'],
            special_guests = [
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
            expansions=['MKM'],
            exclude = [
                'Tomik, Wielder of Law',
                'Melek, Reforged Researcher',
                'Voja, Jaws of the Conclave',
                'Kellan, Inquisitive Prodigy',
                'Tail the Suspect',
            ]
        )
