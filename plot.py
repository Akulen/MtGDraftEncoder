import os

from config import parse_plot

plot_args = None
if __name__ == '__main__':
    plot_args = parse_plot()

    from gpu_management import set_gpus
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    set_gpus(1, forcing=False)

CSI = '\x1b[' #]

import argparse
import copy
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import pickle
from typing import List, Optional, Tuple
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

def str_field(field: str, value):
    if field == 'lr':
        return f'{value:.0e}'
    return str(value)

def fields_to_label(params, fields, prefix=''):
    label = ''
    cnt = 0
    start = True
    for field, (value, length) in fields.items():
        if len(value) == 1:
            continue
        if not start:
            label += ','
        start = False
        if prefix+field in params:
            cnt += 1
            label += \
                f'{field}={str_field(field, params[prefix+field]):<{length}}'
        else:
            label += \
                ' ' * (len(field) + 1 + length)
    return label, cnt

def make_labels(
    configs: List[argparse.Namespace],
    show_param_count=True,
    average_runs=False
):
    longest_model_name = 0
    field_lengths = {}
    field_values = {}
    params_list = []
    for config in configs:
        params = {}
        for k, v in vars(config).items():
            if k in ['model', 'loss', 'test_frequency']: continue
            if average_runs and k == 'seed': continue
            if not show_param_count and k == 'param_count': continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    params[f'{k}.{k2}'] = v2
            else:
                params[k] = v
        params_list.append(params)
        longest_model_name = max(longest_model_name, len(config.model))
        for k, v in params.items():
            if k not in field_lengths:
                field_lengths[k] = 0
                field_values[k] = set()
            field_lengths[k] = max(field_lengths[k], len(str_field(k, v)))
            if isinstance(v, list):
                v = tuple(v)
            field_values[k].add(v)

    labels = []
    for config, params in zip(configs, params_list):
        label_model_params, cnt = fields_to_label(
            params,
            {
                field[13:]: (field_values[field], field_lengths[field])
                for field in field_values
                if field.startswith('model_params.')
            },
            'model_params.'
        )
        label_params, _ = fields_to_label(
            params,
            {
                field: (field_values[field], field_lengths[field])
                for field in field_values
                if not field.startswith('model_params.')
            }
        )
        label = f'{config.model:<{longest_model_name}}'
        if cnt > 0:
            label += f'[{label_model_params}]'
        label += ',' + label_params
        labels.append(label)
    return labels

def smooth(Y, smooth_window):
    extra = (smooth_window + 1) // 2
    Y = np.repeat(Y, [extra] + [1] * (len(Y)-2) + [extra])
    Y = np.convolve(Y, np.ones(smooth_window) / smooth_window, mode='valid')
    return Y

def plot_curve(ax, X, Y, color, args, offset=0):
    # Y = Y[:offset]
    # X = X[:offset]
    # offset = 0
    min_index = offset + np.argmin(Y[offset:])
    max_index = offset + np.argmax(Y[offset:])

    min_x = X[min_index]
    min_y = Y[min_index]
    max_x = X[max_index]
    max_y = Y[max_index]

    ax.plot(
        X,
        Y,
        color=cmap(color),
        **args
    )

    if min_x > offset+1:
        ax.plot(min_x, min_y, marker='x', color=cmap(color), markersize=5)
    if max_x > offset+1:
        ax.plot(max_x, max_y, marker='x', color=cmap(color), markersize=5)
    return ((min_index, min_x, min_y), (max_index, max_x, max_y))

def plot_single(
    ax: Axes,
    args: dict,
    color: int,
    loss: int,
    results: List[Tuple[float, List[List[float]]]],
    max_epoch_size: int,
    group_batches: bool=True,
    smooth_window: int = 1,
    x_max: Optional[int] = None,
    offset: int=0,
    **kwargs
):
    X, Y = list(zip(*results))
    Y = [y[loss] for y in Y]
    if not group_batches:
        X = np.concatenate([
            x - 1 + np.arange(len(y)) / max_epoch_size
            for x, y in zip(X, Y)
        ])
    Y = np.concatenate([
        [y.mean()] if group_batches else y
        for y in Y
    ])
    if smooth_window > 1:
        Y = smooth(Y, smooth_window)
    X = np.array(X)
    if x_max is not None:
        X = X[X <= x_max]
        Y = Y[:len(X)]
    ((min_i, min_x, min_y), (max_i, max_x, max_y)) = plot_curve(
        ax, X, Y, color, args, offset
    )
    return Y, np.zeros_like(Y), ((min_i, min_x, min_y), (max_i, max_x, max_y))

def plot_average(
    ax: Axes,
    args: dict,
    color: int,
    loss: int,
    results: List[List[Tuple[float, List[List[float]]]]],
    max_epoch_size: int,
    group_batches: bool = True,
    smooth_window: int = 1,
    ci_alpha: float = 0.1,
    x_max: Optional[int] = None,
    offset: int=0
):
    if len(results) == 0:
        return np.array([])
    Ys = []
    X_ref = None

    for run in results:
        X, Y = list(zip(*run))
        Y = [y[:, loss] for y in Y]

        if not group_batches:
            X_run = np.concatenate([
                x - 1 + np.arange(len(y)) / max_epoch_size
                for x, y in zip(X, Y)
            ])
        else:
            X_run = np.asarray(X)

        Y_run = np.concatenate([
            [y.mean()] if group_batches else y
            for y in Y
        ])
        if smooth_window > 1:
            Y_run = smooth(Y_run, smooth_window)
        X = np.array(X)

        if X_ref is None:
            X_ref = X_run
        else:
            assert np.allclose(X_ref, X_run)

        Ys.append(Y_run)
    assert X_ref is not None

    Y = np.stack(Ys, axis=0)
    if x_max is not None:
        X_ref = X_ref[X_ref <= x_max]
        Y = Y[:,:len(X_ref)]
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0, ddof=1)

    ((min_i, min_x, min_y), (max_i, max_x, max_y)) = plot_curve(
        ax, X_ref, Y_mean, color, args, offset
    )
    ax.fill_between(
        X_ref,
        Y_mean - Y_std / 2,
        Y_mean + Y_std / 2,
        color=cmap(color),
        alpha=ci_alpha,
        linewidth=0
    )

    return Y_mean, Y_std, ((min_i, min_x, min_y), (max_i, max_x, max_y))

def plot(
    exp_args, labels, results, plot_args, path, group_batches=True, loss=None,
    fine_tune=None
):
    if loss is None:
        loss = exp_args.loss
    if isinstance(loss, List):
        for i, l in enumerate(loss):
            print(l)
            plot(
                exp_args, labels, results, plot_args, path,
                group_batches=group_batches, loss=(i, l), fine_tune=fine_tune
            )
        return
    if isinstance(loss, str):
        loss = (0, loss)
    fts = set()
    for run in exp_args.runs:
        fts.add(run.fine_tune)
    if fine_tune is None and len(fts) > 1:
        for ft in fts:
            print(f'Fine Tune {ft} days')
            plot(
                exp_args, labels, results, plot_args, path,
                group_batches=group_batches, loss=loss, fine_tune=ft
            )
        return
    assert(fine_tune in fts)
    ft_filter = [
        i for i, run in enumerate(exp_args.runs) if run.fine_tune == fine_tune
    ]
    runs = [exp_args.runs[i] for i in ft_filter]
    results = [results[i] for i in ft_filter]
    labels = [labels[i] for i in ft_filter]
    i_loss, loss = loss
    fig, ax = plt.subplots(figsize=(plot_args.width, plot_args.height))
    ax_lr = None
    if plot_args.plot_lr:
        ax_lr = ax.twinx()
        # ax_lr.set_yscale('log')
    # ax.set_xlabel('Epoch' if group_batches else 'Batch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({loss})')
    # ax.set_title(f'{exp_args.name}{'' if group_batches else ' Batch'} Losses')

    set_config = {
        'train_losses': {
            'name': 'Train',
            'line': {
                'linewidth': 3,
            }
        },
        'val_losses': {
            'name': 'Validation',
            'line': {
                'linewidth': 1.5,
            }
        },
        'test_losses': {
            'name': 'Test',
            'line': {
                'linestyle': 'dashdot',
                'linewidth': 2,
            }
        }
    }
    set_ci_alpha = {
        'train_losses': 
            0.3 if (
                    not plot_args.focus_valid
                and not plot_args.focus_test
            ) else 0.1,
        'val_losses': 0.3 if plot_args.focus_valid else 0.1,
        'test_losses': 0.3 if plot_args.focus_test else 0.1
    }
    for dataset in set_config:
        ax.plot(
            [], [], **set_config[dataset]['line'], color='grey',
            label=f'{set_config[dataset]["name"]} Loss'
        )
    if ax_lr is not None:
        ax.plot(
            [], [], linestyle=':', linewidth='0.5', color='gray',
            label='Learning Rate'
        )

    max_epochs = max(
        run.epochs + (
            run.fine_tune_epochs if run.fine_tune is not None else 0
        )
        for run in runs
    )
    test_epochs = runs[0].test_frequency
    n_epochs = runs[0].epochs
    for run in runs[1:]:
        if run.test_frequency != test_epochs:
            test_epochs = None
            break
        if run.epochs != n_epochs:
            test_epochs = None
            break
    for epoch in range(max_epochs+1):
        is_test_epoch = False
        if test_epochs is not None:
            is_test_epoch = (n_epochs - epoch) % test_epochs == 0
        if is_test_epoch and group_batches:
            ls = "dotted" if plot_args.only_test_indicators else "dashed"
            ax.axvline(
                epoch,
                linestyle=ls,
                color="0.7", lw=1
            )
        if not is_test_epoch and plot_args.only_test_indicators:
            continue
        ls = "dashed" if is_test_epoch else "dotted"
        if plot_args.only_test_indicators:
            ls = "dotted"
        if not group_batches:
            ax.axvline(
                epoch,
                linestyle=ls, color="0.7", lw=1
            )

    ####################
    if ax_lr is not None:
        ax_lr.set_ylabel('Learning Rate')
        for i, result in enumerate(results):
            ax_lr.plot(
                *list(zip(*result['lr'])),
                linestyle=':', linewidth=1, color=cmap(i)
            )
    ##########
    all_values = []
    all_stats = {}
    for dataset in set_config:
        max_epoch_size = 1
        if not group_batches:
            max_epoch_size = max(
                len(ys)
                for result in results
                for _, ys in result[dataset]
            )
        if plot_args.average_runs:
            _grouped_runs = {}
            for i, run in enumerate(results):
                run['config'].seed = None
                hs = hashlib.sha256(json.dumps(
                    vars(run['config']),
                    sort_keys=True
                ).encode('utf-8')).hexdigest()
                if hs not in _grouped_runs:
                    _grouped_runs[hs] = []
                _grouped_runs[hs].append((i, run))
            grouped_runs = []
            for hs, runs in _grouped_runs.items():
                label = labels[runs[0][0]]
                for i, _ in runs[1:]:
                    assert labels[i] == label
                offset = 0
                if runs[0][1]['config'].fine_tune is not None:
                    offset = runs[0][1]['config'].fine_tune_epochs
                tmp_args = copy.deepcopy(set_config[dataset]['line'])
                if dataset == 'train_losses':
                    tmp_args['label'] = label
                grouped_runs.append((
                    tmp_args,
                    [run[dataset] for _, run in runs],
                    offset,
                    label
                ))
            runs = grouped_runs
        else:
            runs = []
            for label, result in zip(labels, results):
                tmp_args = copy.deepcopy(set_config[dataset]['line'])
                if dataset == 'train_losses':
                    tmp_args['label'] = label
                offset = 0
                if result['config'].fine_tune is not None:
                    offset = result['config'].fine_tune_epochs
                runs.append((tmp_args, result[dataset], offset, label))
        for i_run, (tmp_args, result, offset, label) in enumerate(runs):
            plot_fn = plot_average if plot_args.average_runs else plot_single
            values, values_std, stats = plot_fn(
                ax, tmp_args, i_run, i_loss, result, max_epoch_size,
                group_batches, smooth_window=plot_args.smooth_window,
                ci_alpha=set_ci_alpha[dataset], x_max=plot_args.x_max,
                offset=offset
            )
            if (i_run, label) not in all_stats:
                all_stats[(i_run, label)] = {}
            all_stats[(i_run, label)][dataset] = (stats, values, values_std)
            all_values.append(values)
    for i_run, label in all_stats:
        print(f'Run #{i_run+1:02}[{CSI}33m{label}{CSI}0m]:')
        (
            ((min_index, min_x, _), (max_index, max_x, _)), _, _
        ) = all_stats[(i_run, label)]['val_losses']
        # min_index = -1
        for dataset in all_stats[(i_run, label)]:
            _, Y, Y_std = all_stats[(i_run, label)][dataset]
            color, color_hl = 0, 0
            if dataset == 'test_losses':
                color = 34
                color_hl = 32
            # if dataset == 'test_losses':
            #     min_index = (min_index + 1) // 5
            print(
                f'{CSI}{color}m{dataset:>12}: '
                f'Best Epoch: {min_x:03}/loss: '
                    f'{CSI}{color_hl}m{Y[min_index]:.4f} \\pm {Y_std[min_index]:.4f}{CSI}{color}m '
                f'(Worst Epoch: {max_x:03}/loss: '
                   f'{Y[max_index]:.4f}_{{\\pm{Y_std[max_index]:.4f}}}){CSI}0m'
            )
    all_values = np.concatenate(all_values)
    if plot_args.exclude_outliers:
        p = 0.01
        y_min, y_max = np.quantile(all_values, [p, 1-p])
        y_mean = (y_min + y_max) / 2
        y_delta = y_mean - y_min
        margin = 1.1
        y_min = y_mean - margin * y_delta
        y_max = y_mean + margin * y_delta
    else:
        y_min = all_values.min()
        y_max = all_values.max()
    ####################

    ax.set_xlim(
        1 if group_batches else 0,
        plot_args.x_max or max_epochs
    )
    if plot_args.y_max is not None:
        y_max = plot_args.y_max
    if plot_args.y_min is not None:
        y_min = plot_args.y_min
    ax.set_ylim(top=y_max)
    ax.set_ylim(bottom=y_min)

    if plot_args.mark_lines:
        for y_pos in ax.get_yticks():
            ax.axhline(y_pos, color='0.85', linestyle='--', linewidth=0.5)

    if plot_args.legend_fontsize > 0:
        ax.legend(fontsize=plot_args.legend_fontsize)
    else:
        ax.legend()
    filename = f'{'epoch' if group_batches else 'batch'}_{loss}'
    if fine_tune is not None:
        filename += f'_ft{fine_tune}'
    fig.savefig(
        path.parent / f'{filename}.png',
        dpi=plot_args.dpi,
        bbox_inches='tight'
    )

if plot_args is not None:
    path = pathlib.Path(plot_args.data)
    with open(path, 'rb') as f:
        exp_args, results = pickle.load(f)

    cmap = plt.get_cmap(plot_args.cmap)

    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 12 # 18

    labels = make_labels(
        [run['config'] for run in results],
        plot_args.show_param_count,
        plot_args.average_runs
    )
    # del exp_args.runs
    # print(exp_args)

    plot(exp_args, labels, results, plot_args, path)
    if not plot_args.skip_batch:
        plot(exp_args, labels, results, plot_args, path, group_batches=False)
