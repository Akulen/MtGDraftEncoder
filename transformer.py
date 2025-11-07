import math
import torch

from typing import cast, List, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import equinox as eqx

def keysplitter(key: Optional[Array], num: int) -> List | Array:
    if key is None:
        return [None] * num
    return jax.random.split(key, num=num)

rotary_embedding_cache: dict[int, Array] = {}

class JaxPositionalEncoding(eqx.Module):
    d_model: int=eqx.field(static=True)
    max_seq_length: int=eqx.field(static=True)
    dropout: eqx.nn.Dropout

    def __init__(
        self, d_model: int, dropout: float=0.1, max_seq_length: int=5000
    ):
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        enable_dropout: bool=True,
        key: Optional[Array]=None
    ) -> Float[Array, "seq_len d_model"]:
        seq_len, d_model = x.shape
        assert(d_model == self.d_model)
        assert(seq_len <= self.max_seq_length)

        with jax.ensure_compile_time_eval():
            if x.shape[-1] not in rotary_embedding_cache:
                position = jnp.expand_dims(jnp.arange(self.max_seq_length), 1)
                div_term = jnp.exp(
                    -math.log(10000.0) * (jnp.arange(0, d_model, 2) / d_model)
                )
                pe = jnp.zeros((self.max_seq_length, d_model))
                pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
                pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
                rotary_embedding_cache[x.shape[-1]] = pe
            pe = rotary_embedding_cache[x.shape[-1]]

        x = x + pe[:x.shape[0]] # (seq_len, d_model)
        return self.dropout(
            x, inference=not enable_dropout, key=key
        )

class JaxTransformerEncoderLayer(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(
        self, d_model: int, n_head: int, dim_feedforward: int,
        dropout: float=0.1, key: Optional[Array]=None
    ):
        att_key, lin1_key, lin2_key = keysplitter(key, 3)

        self.norm1 = eqx.nn.LayerNorm(shape=d_model)
        self.norm2 = eqx.nn.LayerNorm(shape=d_model)

        self.dropout1 = eqx.nn.Dropout(dropout)
        self.dropout2 = eqx.nn.Dropout(dropout)
        self.dropout3 = eqx.nn.Dropout(dropout)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_head,
            query_size=d_model,
            use_query_bias=True,
            use_output_bias=True,
            dropout_p=dropout,
            key=att_key,
        )

        self.linear1 = eqx.nn.Linear(
            in_features=d_model,
            out_features=dim_feedforward,
            key=lin1_key,
        )
        self.linear2 = eqx.nn.Linear(
            in_features=dim_feedforward,
            out_features=d_model,
            key=lin2_key,
        )

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        mask: Int[Array, "seq_len"],
        enable_dropout: bool=True,
        key: Optional[Array]=None
    ):
        att_key, drop1_key, drop2_key, drop3_key = keysplitter(key, 4)
        x = jax.vmap(self.norm1)(
            x + self.dropout1(
                self.attention(
                    x, x, x,
                    mask=jnp.outer(mask, mask),
                    inference=not enable_dropout,
                    key=att_key
                ),
                inference=not enable_dropout,
                key=drop1_key
            )
        )
        output = jax.vmap(self.norm2)(
            x + self.dropout3(
                jax.vmap(self.linear2)(self.dropout2(
                    jax.nn.relu(jax.vmap(self.linear1)(x)),
                    inference=not enable_dropout,
                    key=drop2_key
                )),
                inference=not enable_dropout,
                key=drop3_key
            )
        )
        return output

class JaxTransformerEncoder(eqx.Module):
    embedding: eqx.nn.Embedding
    embedding_scaling: Float=eqx.field(static=True)
    pos_encoder: JaxPositionalEncoding
    encoder_layers: List[JaxTransformerEncoderLayer]

    def __init__(
        self, vocab_size, d_model=64, nhead=8, num_encoder_layers=2,
        dim_feedforward=128, dropout=0.1, max_seq_length=100,
        key: Optional[Array]=None
    ):
        emb_key, enc_key = keysplitter(key, 2)
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=emb_key)
        self.embedding_scaling = d_model ** .5
        self.pos_encoder = JaxPositionalEncoding(
            d_model, dropout, max_seq_length
        )
        self.encoder_layers = [
            JaxTransformerEncoderLayer(
                d_model=d_model,
                n_head=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                key=enc_key
            )
            for _ in range(num_encoder_layers)
        ]

    def __call__(
        self,
        x: Int[Array, "seq_len"],
        mask: Int[Array, "seq_len"],
        enable_dropout: bool=True,
        key: Optional[Array]=None
    ) -> Float[Array, "d_model"]:
        pos_key, lay_key = keysplitter(key, 2)
        x = jax.vmap(self.embedding)(x) * self.embedding_scaling
        x = x + self.pos_encoder(x, enable_dropout, pos_key)
        keys = jax.random.split(lay_key, len(self.encoder_layers))
        for i, layer in enumerate(self.encoder_layers):
            x = eqx.filter_checkpoint(layer)(x, mask, enable_dropout, keys[i])
        # Average pooling over the sequence dimension
        x = (mask @ x) / jnp.maximum(jnp.array(1), mask.sum())
        return x

class JaxWinRatePredictor(eqx.Module):
    encoder: JaxTransformerEncoder
    fc: eqx.nn.Linear

    def __init__(
        self, vocab_size, d_model=64, nhead=8, num_encoder_layers=2,
        dim_feedforward=128, dropout=0.1, max_seq_length=100,
        key: Optional[Array]=None
    ):
        encoder_key, fc_key = keysplitter(key, 2)
        self.encoder = JaxTransformerEncoder(
            vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward,
            dropout, max_seq_length, encoder_key
        )
        self.fc = eqx.nn.Linear(d_model, 1, key=fc_key)

    def __call__(
        self,
        x: Int[Array, "seq_len"],
        mask: Int[Array, "seq_len"],
        enable_dropout: bool=True,
        key: Optional[Array]=None
    ) -> Float[Array, "1"]:
        encoding = self.encoder(x, mask, enable_dropout, key)
        return jax.nn.sigmoid(self.fc(encoding))

# class BertWinRatePredictor(nn.Module):
#     def __init__(self, pretrained_model_name='bert-base-uncased'):
#         super(BertWinRatePredictor, self).__init__()
#         self.bert = AutoModel.from_pretrained(pretrained_model_name)
#         self.fc1 = nn.Linear(self.bert.config.hidden_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)
# 
#     def forward(self, input_ids, attention_mask):
#         # Pass input_ids and attention_mask to BERT
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
# 
#         # Use the CLS token representation (batch, hidden_size)
#         # cls_embedding = outputs.last_hidden_state[:, 0, :]
#         # print(cls_embedding)
#         x = outputs.pooler_output
# 
#         # Pass through a linear layer and apply sigmoid for
#         # binary classification
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         output = torch.sigmoid(self.fc3(x))
#         return output

def main():
    import time
    import argparse
    import polars as pl
    import torch.optim as optim
    from tqdm import tqdm
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from aim import Run
    import humanhash
    import optax
    from functools import partial

    from data import CardDataset, DL17Lands, SimpleTokenizer

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
        '-bs',
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
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
        '-m',
        '--modern-bert',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Use ModernBERT instead of baseBERT'
    )
    parser.add_argument(
        '-w',
        '--weighted-loss',
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
        '-r',
        '--predict',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Predict card stats'
    )
    parser.add_argument(
        '-a',
        '--aim',
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Log to AIM server'
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

    run = None
    if args.aim:
        run = Run(
            repo='aim://localhost:53800',
            experiment='MTGateauTransformer',
            capture_terminal_logs=False,
            system_tracking_interval=None
        )
        run.name = humanhash.humanize(run.hash, words=3)
        run["hparams"] = {
            "train_set": args.train_set,
            "test_set": args.test_set,
            "batch_size": args.batch_size,
            "use_bert": args.use_bert,
            "weighted_loss": args.weighted_loss,
        }

    if args.modern_bert or args.use_bert:
        raise NotImplementedError
    modern_bert = args.modern_bert
    if modern_bert:
        bert_model = 'answerdotai/ModernBERT-base'
    else:
        bert_model = 'bert-base-uncased'

    dataloader_train = DL17Lands(args.train_set, verbose=args.verbose)

    if args.train_set == args.test_set:
        n = dataloader_train.cards.shape[0]
        df_train = dataloader_train.cards.head(n//2)
        df_test = dataloader_train.cards.tail(n - n//2)
    else:
        dataloader_test = DL17Lands(args.test_set, verbose=args.verbose)

        df_train = dataloader_train.cards
        df_test = dataloader_test.cards

    vocab_size = None
    max_seq_length = 167 # TODO: Warning
    max_emp_length = 0
    if args.use_bert:
        _tokenizer = AutoTokenizer.from_pretrained(bert_model)
        def tokenizer(text):
            nonlocal max_emp_length
            res = _tokenizer(
                text,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=max_seq_length
            )
            max_emp_length = max(max_emp_length, res['attention_mask'][0].sum())
            assert(res['attention_mask'][0][-1] == 0 and len(res['attention_mask']) == 1)
            return torch.concat([
                res['input_ids'],
                res['attention_mask']
            ]).to(device)
    else:
        tokenizer = SimpleTokenizer(
            df_train['oracle'],
            max_seq_length=max_seq_length,
            device=device
        )
        vocab_size = tokenizer.vocab_size

    data_train = CardDataset(
        df_train['oracle'].to_list(),
        df_train['GIH'].to_torch(),
        df_train['weight'].to_torch(),
        tokenizer,
        use_torch=False
    )
    data_test = CardDataset(
        df_test['oracle'].to_list(),
        df_test['GIH'].to_torch(),
        df_test['weight'].to_torch(),
        tokenizer,
        use_torch=False
    )

    devices = jax.local_devices()
    n_devices = len(devices)

    data_train.to_devices(devices)
    data_test.to_devices(devices)
    train_batches, n_batches = data_train.batch(
        batch_size=args.batch_size, shuffle=True, seed=42
    )
    test_batches, _ = data_test.batch(
        batch_size=args.batch_size, shuffle=False
    )

    rng_key = jax.random.PRNGKey(42)

    # Model setup
    if args.use_bert:
        # model = BertWinRatePredictor(bert_model).to(device)
        raise NotImplementedError
    else:
        rng_key, key = jax.random.split(rng_key)
        model = JaxWinRatePredictor(
            vocab_size,
            d_model=768,
            nhead=12,
            num_encoder_layers=12,
            dim_feedforward=768,
            dropout=0.1,
            max_seq_length=max_seq_length,
            key=key
        )

    def MSELoss(input, target, weight=None):
        return ((input - target)**2).mean()
    def weighted_MSELoss(input, target, weight):
        return ((input - target)**2 * weight).sum() / weight.sum()

    tx = optax.adam(learning_rate=5e-6)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = tx.init(params)

    # Training loop
    n_epochs = 50
    total_steps = n_batches * n_epochs

    # # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps = 0,
    #     num_training_steps = total_steps
    # )

    from typing import Callable
    @eqx.filter_value_and_grad
    def compute_loss(
        params, static,
        inputs: Int[Array, "batch_size 2 seq_len"],
        targets: Float[Array, "batch_size"],
        weight: Float[Array, "batch_size"],
        key: Array,
    ) -> Float[Array, "1"]:
        batch_size = inputs.shape[0]
        batched_keys = jax.random.split(key, num=batch_size)
        model = eqx.combine(params, static)
        outputs = jax.vmap(cast(Callable, model), in_axes=(0, 0, None, 0))(
            *jnp.moveaxis(inputs, 1, 0),
            True,
            batched_keys
        )
        if args.weighted_loss:
            loss = weighted_MSELoss(outputs.squeeze(), targets, weight ** .5)
        else:
            loss = MSELoss(outputs.squeeze(), targets).mean()
        return loss

    def training(
        params, static,
        inputs: Int[Array, "batch_size 2 seq_len"],
        targets: Float[Array, "batch_size"],
        weight: Float[Array, "batch_size"],
        opt_state: optax.OptState,
        key: Array,
        tx: optax.GradientTransformation
    ):# -> Tuple[eqx.Module, optax.OptState, Float[Array, "1"], Array]:
        print("compiling", inputs, targets, weight, key)
        key, subkey = jax.random.split(key)
        loss, grads = compute_loss(params, static, inputs, targets, weight, subkey)
        grads = jax.lax.pmean(grads, axis_name="devices")

        updates, opt_state = tx.update(
            grads, opt_state, params
        )
        params = optax.apply_updates(params, updates)
        return params, static, opt_state, loss, key

    def eval(
        model: eqx.Module,
        inputs: Int[Array, "batch_size 2 seq_len"],
        key: Array,
    ) -> Float[Array, "batch_size"]:
        batch_size = inputs.shape[0]
        batched_keys = jax.random.split(key, num=batch_size)
        return jax.vmap(cast(Callable, model), in_axes=(0, 0, None, 0))(
            *jnp.moveaxis(inputs, 1, 0),
            False,
            batched_keys
        )

    p_training = eqx.filter_pmap(partial(training, tx=tx), axis_name="devices")
    p_eval = eqx.filter_pmap(eval)

    train_losses = []
    test_losses = []

    opt_state = jax.device_put_replicated(opt_state, devices)
    params = jax.device_put_replicated(params, devices)
    static = jax.device_put_replicated(static, devices)
    rng_key, subkey = jax.random.split(rng_key)
    training_keys = jax.random.split(subkey, n_devices)

    print(max_emp_length)
    st = time.time()
    for epoch in range(n_epochs):
        with tqdm(
            total=n_batches,
            desc=f"Epoch {epoch+1:3d}/{n_epochs}",
            unit="batch"
        ) as pbar:
            running_loss = 0.0
            for inputs, targets, weight in train_batches():
                params, static, opt_state, loss, training_keys = p_training(
                    params, static,
                    inputs,
                    targets,
                    weight,
                    opt_state,
                    training_keys
                )
                running_loss += loss.mean()
                pbar.set_postfix(loss=loss.mean()**.5)
                pbar.update(1)
            train_losses.append((running_loss/n_batches)**.5)

            running_loss = 0.0
            for inputs, targets, weight in test_batches():
                rng_key, subkey = jax.random.split(rng_key)
                keys = jax.random.split(subkey, n_devices)
                outputs = p_eval(
                    eqx.combine(params, static),
                    inputs,
                    keys
                )
                if args.weighted_loss:
                    loss = weighted_MSELoss(outputs.reshape((n_devices, -1)), targets, weight ** .5)
                else:
                    loss = MSELoss(outputs.reshape((n_devices, -1)), targets).mean()
                running_loss += loss.mean()
            test_losses.append((running_loss/n_batches)**.5)

            pbar.set_postfix_str(
                f"Train Loss: {train_losses[-1]:.4f}, "
                f"Test Loss: {test_losses[-1]:.4f}"
            )

            if run is not None:
                run.track(
                    {
                        "loss/train": train_losses[-1],
                        "loss/test": test_losses[-1],
                    },
                    epoch=epoch,
                )
    if run is not None:
        run.close()


    print(f'Training complete in {time.time() - st:.2f}s.')

    if len(test_losses) > 1:
        print(f'Test Loss: {test_losses[-1]:.4f}')

    if args.plot_training:
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.legend() # TODO: make sure figures exist
        plt.gca().set_ylim(bottom=0)
        plt.savefig(f'figures/jax-transformer_losses_{args.train_set}_{args.test_set}{('_modern' if modern_bert else '_') + 'bert' if args.use_bert else ''}.png')

    def predict(df, rng_key):
        df = df.with_columns(pred=pl.zeros(len(df)))
        for card in df.iter_rows(named=True):
            res = tokenizer(
                card['oracle'],
            )
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, n_devices)
            res = jnp.concat(
                [jnp.array(res.reshape((1,)+res.shape))] * n_devices
            ).reshape((n_devices, 1) + res.shape)
            pred = p_eval(
                model,
                res,
                keys
            )
            pred = pred[0,0,0]
            print(pred)
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

    rng_key, pred_key = jax.random.split(rng_key)
    if args.predict:
        predict(df_test, pred_key)

if __name__ == '__main__':
    main()
