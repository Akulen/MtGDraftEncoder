from collections import Counter
import json
import re
import numpy as np
import polars as pl
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

with open('tokens.json', 'r') as f:
    auth_tokens = json.load(f)

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

class Gemma:
    def __init__(self, d_encoder=768, device=None):
        login(token=auth_tokens['huggingface'])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = 'google/embeddinggemma-300M'
        self.model = SentenceTransformer(
            model_id, truncate_dim=d_encoder, device=device
        ).to(device=device)

    def __call__(self, x: str) -> np.ndarray:
        return self.model.encode(x)
