import argparse

parser = argparse.ArgumentParser(
    description="Plotter",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('data', type=str)
group_meta = parser.add_argument_group('Plot Config')
group_meta.add_argument('--width', type=int, default=16)
group_meta.add_argument('--height', type=int, default=9)
group_meta.add_argument('--dpi', type=int, default=300)
group_meta.add_argument('--cmap', type=str, default='tab10')
group_meta.add_argument('--y-max', type=float, default=None)
group_meta.add_argument('--y-min', type=float, default=None)
group_meta.add_argument('--only-test-indicators',
    type=bool, default=False, action=argparse.BooleanOptionalAction
)
group_meta.add_argument('--mark-lines',
    type=bool, default=False, action=argparse.BooleanOptionalAction
)
group_meta.add_argument('--plot-lr',
    type=bool, default=True, action=argparse.BooleanOptionalAction
)
group_meta.add_argument('--skip-batch',
    type=bool, default=False, action=argparse.BooleanOptionalAction
)
args = parser.parse_args()

import os
from gpu_management import set_gpus

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
set_gpus(1, forcing=True)

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
from typing import Tuple
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

config = None
train_epoch_size = None
cnt = 0
cmap = plt.get_cmap('tab10')
MAX_LEN = 6

def plot(data, ax_batch, ax_epoch, label=None):
    if label is None:
        label = []
    if isinstance(data, Tuple):
        field, data = data
        res = []
        for d in data:
            res += plot(d, ax_batch, ax_epoch, label+[field])
        return res
    assert(isinstance(data, dict))
    params = {}
    if 'model' in label:
        for k, v in data['config'].model_params.items():
            params[k] = v
    for hp in label:
        if hp == 'model': continue
        params[hp] = vars(data['config'])[hp]

    global config, train_epoch_size
    config = data['config']
    # if config.model_params['n_set_layers'] + config.model_params['n_layers'] < 15:
    #     return []
    # print(config.model_params['n_set_layers'], config.model_params['n_layers'], config.param_count)

    local_train_epoch_size = data['train_losses'].shape[0] // config.epochs
    assert(train_epoch_size is None or train_epoch_size == local_train_epoch_size)
    train_epoch_size = local_train_epoch_size
    X_train = np.arange(data['train_losses'].shape[0])
    X_val = np.arange(data['val_losses'].shape[0])
    X_test = np.arange(data['test_losses'].shape[0], dtype=np.float32)
    X_test *= X_train[-1] / X_test[-1]
    
    train_epoch_losses = data['train_losses'].reshape((config.epochs, -1)).mean(axis=1)
    val_epoch_losses = data['val_losses'].reshape((config.epochs, -1)).mean(axis=1)
    n_test_epochs = int(np.ceil(config.epochs / config.test_frequency))
    test_epoch_losses = data['test_losses'].reshape((n_test_epochs, -1)).mean(axis=1)

    return [(
        (config.model, params),
        (X_train, data['train_losses']),
        (X_val, data['val_losses']),
        (X_test, data['test_losses']),
        (np.arange(config.epochs), train_epoch_losses, val_epoch_losses),
        (
            (
                  config.epochs
                - np.arange(0, config.epochs, config.test_frequency) - 1
            )[::-1],
            test_epoch_losses,
        ),
        data['lr']
    )]

def str_field(field: str, value):
    if field == 'lr':
        return f'{value:.0e}'
    return str(value)

def main(args: argparse.Namespace):
    global config, train_epoch_size, cmap
    path = pathlib.Path(args.data)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if args.cmap != 'tab10':
        cmap = plt.get_cmap(args.cmap)

    plt.rcParams["font.family"] = "monospace"
    fig_batch, ax_batch = plt.subplots(figsize=(args.width, args.height))
    fig_epoch, ax_epoch = plt.subplots(figsize=(args.width, args.height))
    ax_batch_lr = ax_batch.twinx()
    ax_epoch_lr = ax_epoch.twinx()
    ax_batch_lr.set_yscale('log')
    ax_epoch_lr.set_yscale('log')
    ax_batch.set_xlabel('Batch')
    ax_epoch.set_xlabel('Epoch')

    lines = plot(data, ax_batch, ax_epoch)
    assert(config is not None and train_epoch_size is not None)
    ax_batch.set_ylabel(f'Loss ({config.loss})')
    ax_epoch.set_ylabel(f'Loss ({config.loss})')
    ax_batch.set_title(f'{config.name} Batch Losses')
    ax_epoch.set_title(f'{config.name} Losses')

    if not args.skip_batch:
        ax_batch.plot([], [], color='gray', label='Train Loss')
        ax_batch.plot([], [], color='gray', linewidth='0.75', label='Validation Loss')
        ax_batch.plot([], [], linestyle='dashdot', linewidth='1', color='gray', label='Test Loss')
    ax_epoch.plot([], [], color='gray', label='Train Loss')
    ax_epoch.plot([], [], color='gray', linewidth='0.75', label='Validation Loss')
    ax_epoch.plot([], [], linestyle='dashdot', linewidth='1', color='gray', label='Test Loss')
    if args.plot_lr:
        if not args.skip_batch:
            ax_batch.plot([], [], linestyle=':', linewidth='0.5', color='gray', label='Learning Rate')
        ax_epoch.plot([], [], linestyle=':', linewidth='0.5', color='gray', label='Learning Rate')

    longest_model_name = 0
    field_lengths = {}
    field_values = {}
    for i, ((model_name, params), _, _, _, _, _, _) in enumerate(lines):
        longest_model_name = max(longest_model_name, len(model_name))
        for k, v in params.items():
            if k not in field_lengths:
                field_lengths[k] = 0
                field_values[k] = set()
            field_lengths[k] = max(field_lengths[k], len(str_field(k, v)))
            field_values[k].add(v)

    batch_y_min = np.inf
    batch_y_max = -np.inf
    epoch_y_min = np.inf
    epoch_y_max = -np.inf
    for i, (
        (model_name, params),
        (batch_x_train, batch_y_train),
        (batch_x_val, batch_y_val),
        (batch_x_test, batch_y_test),
        (epoch_x_train, epoch_y_train, epoch_y_val),
        (epoch_x_test, epoch_y_test),
        lr
    ) in enumerate(lines):
        batch_y_min = min(batch_y_min,
            np.min(batch_y_train),
            np.min(batch_y_val),
            np.min(batch_y_test)
        )
        batch_y_max = max(batch_y_max,
            np.max(batch_y_train),
            np.max(batch_y_val),
            np.max(batch_y_test)
        )
        epoch_y_min = min(epoch_y_min,
            np.min(epoch_y_train),
            np.min(epoch_y_val),
            np.min(epoch_y_test)
        )
        epoch_y_max = max(epoch_y_max,
            np.max(epoch_y_train),
            np.max(epoch_y_val),
            np.max(epoch_y_test)
        )
        label_params = ''
        param_cnt = 0
        if len(field_values) > 0:
            start = True
            for field in field_values:
                if len(field_values[field]) == 1:
                    continue
                if not start:
                    label_params += ','
                start = False
                if field in params:
                    param_cnt += 1
                    label_params += \
                        f'{field}={str_field(field, params[field]):<{field_lengths[field]}}'
                else:
                    label_params += \
                        ' ' * (len(field) + 1 + field_lengths[field])
        label = f'{model_name:<{longest_model_name}}'
        if param_cnt > 0:
            label += f'[{label_params}]'
        if not args.skip_batch:
            ax_batch.plot(1+batch_x_train, batch_y_train, label=label, color=cmap(i))
            ax_batch.plot(1+batch_x_val, batch_y_val, linewidth='0.75', color=cmap(i))
            ax_batch.plot(1+batch_x_test, batch_y_test, linestyle='dashdot', linewidth='1', color=cmap(i))
        ax_epoch.plot(1+epoch_x_train, epoch_y_train, label=label, color=cmap(i))
        ax_epoch.plot(1+epoch_x_train, epoch_y_val, linewidth='0.75', color=cmap(i))
        ax_epoch.plot(1+epoch_x_test, epoch_y_test, linestyle='dashdot', linewidth='1', color=cmap(i))
        if args.plot_lr:
            ax_batch_lr.set_ylabel('Learning Rate')
            ax_epoch_lr.set_ylabel('Learning Rate')
            X_val = np.concat([
                [0],
                1+epoch_x_train
            ])
            if len(lr) < len(X_val):
                lr = np.concat([lr, [lr[-1]]*(len(X_val)-len(lr))])
            if not args.skip_batch:
                    ax_batch_lr.plot(X_val*train_epoch_size, lr, linestyle=':', linewidth=0.5, color=cmap(i))
            ax_epoch_lr.plot(X_val, lr, linestyle=':', linewidth=0.5, color=cmap(i))

    ax_batch.set_xlim(1, config.epochs*train_epoch_size)
    ax_epoch.set_xlim(1, config.epochs)

    # TODO: Move below lines???
    for epoch in range(config.epochs+1):
        is_test_epoch = (config.epochs-epoch) % config.test_frequency == 0
        if not is_test_epoch and args.only_test_indicators:
            continue
        ls = "dashed" if is_test_epoch else "dotted"
        if args.only_test_indicators:
            ls = "dotted"
        ax_batch.axvline(
            epoch * train_epoch_size,
            linestyle=ls, color="0.7", lw=1
        )
    for epoch in range(config.epochs+1):
        is_test_epoch = (config.epochs-epoch) % config.test_frequency == 0
        if is_test_epoch:
            ax_epoch.axvline(
                epoch,
                linestyle="dotted" if args.only_test_indicators else "dashed",
                color="0.7", lw=1
            )
    if args.mark_lines:
        for ax in [ax_batch, ax_epoch]:
            for y_pos in ax.get_yticks():
                ax.axhline(y_pos, color='0.85', linestyle='--', linewidth=0.5)

    if args.y_max is not None:
        batch_y_max = args.y_max
        epoch_y_max = args.y_max
    if args.y_min is not None:
        batch_y_min = args.y_min
        epoch_y_min = args.y_min
    ax_batch.set_ylim(top=batch_y_max)
    ax_batch.set_ylim(bottom=batch_y_min)
    ax_epoch.set_ylim(top=epoch_y_max)
    ax_epoch.set_ylim(bottom=epoch_y_min)

    ax_batch.legend()
    ax_epoch.legend()
    if not args.skip_batch:
        fig_batch.savefig(path.parent / 'losses_batch.png', dpi=args.dpi)
    fig_epoch.savefig(path.parent / 'losses_epoch.png', dpi=args.dpi)


if __name__ == '__main__':
    main(args)
