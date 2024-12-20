import math
import torch
import torch.nn as nn
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * (torch.arange(0, d_model, 2) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1]] # (batch, seq_len, d_model)
        return self.dropout(x)

class TorchTransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size, d_model=64, nhead=8, num_encoder_layers=2,
        dim_feedforward=128, dropout=0.1, max_seq_length=100
    ):
        super(TorchTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scaling = torch.tensor(
            d_model, dtype=torch.float32
        ).sqrt()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.max_seq_length = max_seq_length

    def forward(self, x):
        x = self.embedding(x) * self.embedding_scaling
        x = x + self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average pooling over the sequence dimension
        return x

class TorchWinRatePredictor(nn.Module):
    def __init__(
        self, vocab_size, d_model=64, nhead=8, num_encoder_layers=2,
        dim_feedforward=128, dropout=0.1, max_seq_length=100
    ):
        super(TorchWinRatePredictor, self).__init__()
        self.encoder = TorchTransformerEncoder(
            vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward,
            dropout, max_seq_length)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        encoding = self.encoder(x)
        output = torch.sigmoid(self.fc(encoding))
        return output


class BertWinRatePredictor(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(BertWinRatePredictor, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask):
        # Pass input_ids and attention_mask to BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the CLS token representation (batch, hidden_size)
        # cls_embedding = outputs.last_hidden_state[:, 0, :]
        # print(cls_embedding)
        x = outputs.pooler_output

        # Pass through a linear layer and apply sigmoid for
        # binary classification
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))
        return output

if __name__ == '__main__':
    import argparse
    import polars as pl
    import torch.optim as optim
    from tqdm import tqdm
    from transformers import BertTokenizer, get_linear_schedule_with_warmup

    from data import CardDataset, DL17Lands, SimpleTokenizer
    from loss import weighted_MSELoss

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        prog='Transformers',
        description='Simple transformer tests',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--train-set',
        type=str,
        default='MKM',
        help='Extension to use for training'
    )
    parser.add_argument(
        '--test-set',
        type=str,
        default='OTJ',
        help='Extension to use for testing'
    )
    parser.add_argument(
        '-b',
        '--use-bert',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Use pre-trained BERT Transformer'
    )
    parser.add_argument(
        '-w',
        '--weighted_loss',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Weight the loss by the number of occurences of each card'
    )
    parser.add_argument(
        '-p',
        '--plot-training',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Show training loss plot'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Verbose output'
    )

    args = parser.parse_args()

    dataloader_train = DL17Lands(args.train_set, verbose=args.verbose)
    dataloader_test = DL17Lands(args.test_set, verbose=args.verbose)

    df_train = dataloader_train.cards
    df_test = dataloader_test.cards

    vocab_size = None
    max_seq_length = 160 # TODO: Warning
    if args.use_bert:
        _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenizer(text):
            res = _tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=max_seq_length
            )
            assert(res['attention_mask'][0][-1] == 0)
            return torch.concat([
                res['input_ids'],
                res['attention_mask']
            ]).to(device)
    else:
        tokenizer = SimpleTokenizer(df_train['oracle'])
        vocab_size = tokenizer.vocab_size

    data_train = CardDataset(
        df_train['oracle'].to_list(),
        df_train['GIH'].to_torch().to(device),
        df_train['weight'].to_torch().to(device),
        tokenizer,
        use_torch=True
    )
    data_test = CardDataset(
        df_test['oracle'].to_list(),
        df_test['GIH'].to_torch().to(device),
        df_test['weight'].to_torch().to(device),
        tokenizer,
        use_torch=True
    )

    batch_size = 16
    train_batches, n_batches = data_train.batch(
        batch_size=batch_size, shuffle=True, seed=42
    )
    test_batches, _ = data_train.batch(
        batch_size=batch_size, shuffle=False
    )

    # Model setup
    if args.use_bert:
        model = BertWinRatePredictor().to(device)
    else:
        model = TorchWinRatePredictor(
            vocab_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=6,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_length=max_seq_length
        )

    if args.weighted_loss:
        criterion = weighted_MSELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)

    # Training loop
    n_epochs = 2
    total_steps = n_batches * n_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        with tqdm(
            total=n_batches,
            desc=f"Epoch {epoch+1}/{n_epochs}",
            unit="batch"
        ) as pbar:
            running_loss = 0.0
            model.train()
            for inputs, labels, weight in train_batches():
                optimizer.zero_grad()
                outputs = model(*inputs.unbind(1))
                if args.weighted_loss:
                    loss = criterion(outputs.squeeze(), labels, weight ** .5)
                else:
                    loss = criterion(outputs.squeeze(), labels)
                running_loss += loss.item() * len(inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(loss=loss.item()**.5)
                pbar.update(1)
            train_losses.append((running_loss/len(df_train))**.5)

            running_loss = 0.0
            model.eval()
            for inputs, labels, weight in test_batches():
                with torch.no_grad():
                    outputs = model(*inputs.unbind(1))
                if args.weighted_loss:
                    loss = criterion(outputs.squeeze(), labels, weight ** .5)
                else:
                    loss = criterion(outputs.squeeze(), labels)
                running_loss += loss.item() * len(inputs)
            test_losses.append((running_loss/len(df_test))**.5)

            pbar.set_postfix_str(
                f"Train Loss: {train_losses[-1]:.4f}, "
                f"Test Loss: {test_losses[-1]:.4f}"
            )


    print('Training complete.')

    if len(test_losses) > 1:
        print(f'Test Loss: {test_losses[-1]:.4f}')

    if args.plot_training:
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.legend()
        plt.show()

    model.eval()
    def predict(df):
        df = df.with_columns(pred=pl.zeros(len(df)))
        for card in df.iter_rows(named=True):
            res = tokenizer(
                card['oracle'],
            )
            pred = model(*res.unsqueeze(0).to(device).unbind(1)).item()
            df = df.with_columns(
                pred=pl.when(pl.col("name") == card['name'])
                       .then(pred)
                       .otherwise(pl.col("pred"))
            )
        df = df.with_columns(
            error=(pl.col(f'GIH') - pl.col('pred')).abs()
        )
        with pl.Config(set_tbl_rows=len(df)):
            print(df.select(
                pl.col('name', f'GIH', 'pred', 'error')
            ))
            print(df.select(
                pl.col('name', f'GIH', 'pred', 'error')
            ).sort(by='error', descending=True))
            print(df['pred'].describe())

    predict(df_test)
