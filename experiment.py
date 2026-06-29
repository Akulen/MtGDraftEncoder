import os

from config import parse_exp

exp_args = None
if __name__ == '__main__':
    exp_args = parse_exp()

    from gpu_management import set_gpus
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    set_gpus(exp_args.n_gpus, forcing=True)

import argparse
import copy
import time
from collections import namedtuple
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pickle
import optax
from tqdm import tqdm
from typing import Any, List, Optional, Tuple
from jaxtyping import PRNGKeyArray

from data import JaxDraftDataset, make_dataset_from_args, fine_tuning_dataset
from models import get_model, get_model_params, DraftWRPredictor
from training import Trainer

CSI = "\x1b[" #]

# os.makedirs('cache/jax', exist_ok=True)
# jax.config.update("jax_compilation_cache_dir", 'cache/jax')
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

def base_args(args):
    new_args = copy.deepcopy(vars(args))
    # del new_args['name']
    # del new_args['n_gpus']
    # del new_args['conf']
    # del new_args['verbose']
    return new_args

def get_hash(args: argparse.Namespace):
    return hashlib.sha256(
        json.dumps(base_args(args), sort_keys=True).encode('utf-8')
    ).hexdigest()

def get_symlink(exp_name, i_run, model_name) -> str:
    return  (
        f'experiments/results/{exp_name}/model-{i_run:02}-{model_name}.eqx'
    )

def check_cached_model(
    exp_name: str, i_run: int, args: argparse.Namespace
) -> Optional[dict]:
    hs = get_hash(args)
    if not os.path.exists(f'experiments/cache/{hs}'):
        return None
    with open(f'experiments/cache/{hs}/results.pkl', 'rb') as f:
        results = pickle.load(f)
        if base_args(args) != base_args(results['config']):
            raise ValueError(
                "Cached results do not match args.\n"
                f"{CSI}31m{base_args(args)}{CSI}0m\n"
                f"{CSI}34m{base_args(results['config'])}{CSI}0m"
            )
    print(f'{CSI}34mUsing cached results for{CSI}0m {hs}')
    symlink = get_symlink(exp_name, i_run, args.model)
    if os.path.lexists(symlink):
        os.unlink(symlink)
    os.symlink(
        f'../../cache/{hs}/model-{args.model}.eqx',
        symlink
    )
    return results

def prepare_trainer(
    args: argparse.Namespace, model: DraftWRPredictor, n_steps: int
) -> Tuple[Trainer, Any, Any, Any]:
    lr = args.lr
    if args.scheduler == 'warmup_cosine':
        n_restarts = args.scheduler_params.get('n_restarts', 0)
        total_steps = n_steps * args.epochs
        assert total_steps % (n_restarts + 1) == 0
        total_steps //= (n_restarts + 1)
        init_lr = lr * float(args.scheduler_params.get('init_mult', 1e-2))
        end_lr  = lr * float(args.scheduler_params.get('end_mult',  1e-4))
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=lr,
            end_value=end_lr,
            warmup_steps=int(args.scheduler_params.get(
                'warmup_steps', args.epochs / 10
            ) * n_steps),
            decay_steps=total_steps,
        )
        if n_restarts > 0:
            restart_factor = float(args.scheduler_params.get(
                'restart_factor', 1
            ))
            scheduler = optax.join_schedules(
                schedules=[scheduler] + [
                    optax.cosine_decay_schedule(
                        init_value=lr * restart_factor ** i,
                        alpha=end_lr * restart_factor ** i,
                        decay_steps=total_steps,
                    )
                    for i in range(1, n_restarts + 1)
                ],
                boundaries=[(i+1) * total_steps for i in range(n_restarts)]
            )
        lr_transform = namedtuple('lr_transform', ['update'])(
            lambda updates, state, value: (None, state)
        )
        lr_transform_state = namedtuple('lr_transform_state', ['scale'])(1.)
    elif args.scheduler == 'adaptive':
        lr_transform = optax.contrib.reduce_on_plateau(
            patience=args.scheduler_params.get('patience', 5),
            cooldown=args.scheduler_params.get('cooldown', 0),
            factor=args.scheduler_params.get('factor', 0.5),
            rtol=args.scheduler_params.get('rtol', 1e-4),
            accumulation_size=args.scheduler_params.get('accumulation_size', 1),
        )
        lr_transform_state = lr_transform.init(
            eqx.filter(model, eqx.is_inexact_array)
        )
        scheduler = lr
    else:
        raise NotImplementedError(f'Unknown scheduler {args.scheduler}')

    if args.optimizer == 'lion':
        tx = optax.inject_hyperparams(optax.lion)(learning_rate=scheduler)
    elif args.optimizer == 'adam':
        tx = optax.inject_hyperparams(optax.adam)(learning_rate=scheduler)
    elif args.optimizer == 'adamw':
        tx = optax.inject_hyperparams(optax.adamw)(learning_rate=scheduler)
    else:
        raise NotImplementedError(f'Unknown optimizer {args.optimizer}')

    trainer = Trainer(tx, loss=args.loss)
    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

    return trainer, opt_state, lr_transform, lr_transform_state

def epoch_loop(
    args: argparse.Namespace,
    n_epochs: int,
    data_train: JaxDraftDataset,
    data_val: JaxDraftDataset,
    data_test: JaxDraftDataset,
    params: DraftWRPredictor,
    static: DraftWRPredictor,
    state: eqx.nn.State,
    opt_state: Any,
    lr_transform: Any,
    lr_transform_state: Any,
    key: PRNGKeyArray,
    first_epoch: int=1
) -> Tuple[
    List, List, List, List, PRNGKeyArray, int, Tuple[
        DraftWRPredictor, eqx.nn.State, Any
    ], Tuple[int, np.ndarray], Tuple[int, np.ndarray], Tuple[int, np.ndarray]
]:
    best_epoch = -1
    best_state = (params, state, opt_state)
    best_val_loss = jnp.inf
    best_test_loss = None

    train_losses = []
    val_losses = []
    test_losses = []
    test_preds = (-1, np.zeros(0))
    test_trues = (-1, np.zeros(0))
    test_npicks = (-1, np.zeros(0))
    last_val_loss = 'N/A'
    last_test_loss = 'N/A'
    lr_history = []
    for i_epoch in (pbar:=tqdm(range(first_epoch, first_epoch+n_epochs))):
        (
            params, state, opt_state, lr_transform_state, key, losses, _, _, _
        ) = data_train.run_batches(
            params, state, opt_state, lr_transform_state,
            args.batch_size, True, key
        )
        train_losses.append((i_epoch, np.array(losses)))
        last_train_loss = losses[:, 0].mean()
        pbar.set_description(
            f"Loss: {last_train_loss:.6f} "
            f"[Val: {last_val_loss}, Test: {last_test_loss}]"
        )

        (
            params, state, opt_state, lr_transform_state, key, losses, _, _, _
        ) = data_val.run_batches(
            params, state, opt_state, lr_transform_state,
            args.batch_size, False, key
        )
        val_losses.append((i_epoch, np.array(losses)))
        loss_mean = losses[:, 0].mean()
        if loss_mean <= best_val_loss:
            best_val_loss = loss_mean
            best_state = (params, state, opt_state)
            best_epoch = i_epoch
            best_test_loss = last_test_loss
        last_val_loss = f'{loss_mean:.6f}'
        pbar.set_description(
            f"Loss: {last_train_loss:.6f} "
            f"[Val: {last_val_loss}, Test: {last_test_loss}]"
        )
        _, lr_transform_state = lr_transform.update(
            updates=eqx.filter(
                eqx.combine(params, static),
                eqx.is_inexact_array
            ),
            state=lr_transform_state,
            value=losses[:, 0].mean()
        )
        lr_history.append((
            i_epoch,
            (
                  opt_state.hyperparams['learning_rate']
                * lr_transform_state.scale #type: ignore
            ).item()
        ))
        if (
                args.scheduler == 'adaptive'
            and len(lr_history) > 1
            and lr_history[-1] != lr_history[-2]
        ):
            params, state, opt_state = best_state

        if i_epoch != 1 and (args.epochs-i_epoch) % args.test_frequency != 0:
            continue
        (
            params, state, opt_state, lr_transform_state, key, losses, pred,
            true, mask
        ) = data_test.run_batches(
            params, state, opt_state, lr_transform_state,
            args.batch_size, False, key
        )
        test_losses.append((i_epoch, np.array(losses)))
        if i_epoch == best_epoch:
            test_preds = (i_epoch, np.array(pred).astype(np.float16))
            test_trues = (i_epoch, np.array(true))
            mask = np.array(mask)
            assert(len(mask.shape) == 3)
            n, m = mask.shape[:2]
            mask_pad = np.concatenate([
                mask, np.zeros((n, m, 1), dtype=np.bool)
            ], axis=2)
            n_picks = np.argmax(~mask_pad, axis=2)
            tail_falses = np.arange(45)[None, None, :] >= n_picks[:, :, None]
            assert((mask ^ tail_falses).all())
            test_npicks = (i_epoch, n_picks)
            best_test_loss = f'{loss_mean:.6f}'
        loss_mean = losses[:, 0].mean()
        last_test_loss = f'{loss_mean:.6f}'
        pbar.set_description(
            f"Loss: {last_train_loss:.6f} "
            f"[Val: {last_val_loss}, Test: {last_test_loss}]"
        )
        if jnp.isnan(last_train_loss):
            break

    with np.printoptions(precision=3, suppress=True):
        print(f'Epoch {best_epoch:03}:')
        print(f'Train Losses: {train_losses[best_epoch-first_epoch][1][:, 0].mean()}')
        print(f'Valid Losses: {val_losses[best_epoch-first_epoch][1][:, 0].mean()}')
        if best_test_loss is not None:
            print(f'Test Losses: {best_test_loss}')

    return (
        train_losses, val_losses, test_losses, lr_history, key, best_epoch,
        best_state, test_preds, test_trues, test_npicks
    )

def train_model(
    args: argparse.Namespace,
    data_train: JaxDraftDataset,
    data_val: JaxDraftDataset,
    data_test: JaxDraftDataset,
    params: DraftWRPredictor,
    static: DraftWRPredictor,
    state: eqx.nn.State,
    trainer: Trainer,
    opt_state: Any,
    lr_transform: Any,
    lr_transform_state: Any,
    key: PRNGKeyArray,
    verbose: int=0
) -> Tuple[dict, DraftWRPredictor, eqx.nn.State, Any]:
    data_train.shard_data(trainer)
    data_val.shard_data(trainer)
    data_test.shard_data(trainer)
    data_train.set_step_function(static, trainer.train_step)
    data_val.set_step_function(static, trainer.eval_step)
    data_test.set_step_function(static, trainer.eval_step)

    data_train.precompile(
        params, state, opt_state, lr_transform_state, args.batch_size, key,
        verbose=verbose, verbose_name='train'
    )
    data_val.precompile(
        params, state, opt_state, lr_transform_state, args.batch_size, key,
        verbose=verbose, verbose_name='validation'
    )
    data_test.precompile(
        params, state, opt_state, lr_transform_state, args.batch_size, key,
        verbose=verbose, verbose_name='test'
    )


    (
        train_losses, val_losses, test_losses, lr_history, key, best_epoch,
        best_state, test_preds, test_trues, test_npicks
    ) = epoch_loop(
        args, args.epochs, data_train, data_val, data_test, params, static,
        state, opt_state, lr_transform, lr_transform_state, key
    )
    lr_history = (
          [(0, opt_state.hyperparams['learning_rate'].item())]
        + lr_history
    )

    hs = get_hash(args)
    os.makedirs(f'experiments/cache/{hs}', exist_ok=True)
    eqx.tree_serialise_leaves(
        f'experiments/cache/{hs}/model-{args.model}.eqx',
        best_state
    )

    results = {
        'config': args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'lr': lr_history,
        'key': np.array(key),
        'best_epoch': best_epoch,
        'test_preds': test_preds,
        'test_trues': test_trues,
        'test_npicks': test_npicks,
    }
    with open(f'experiments/cache/{hs}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results, *best_state

def run(
    exp_name: str,
    i_run: int,
    args: argparse.Namespace,
    data_train: JaxDraftDataset,
    data_val: JaxDraftDataset,
    data_test: JaxDraftDataset,
    verbose: int=0
):
    constant_dataset = None
    if args.model == 'Constant':
        assert 'dataset' in args.model_params
        match args.model_params['dataset']:
            case 'train':
                pass
            case 'val':
                data_val.reset()
                data_train = copy.deepcopy(data_val)
            case 'test':
                data_test.reset()
                data_train = copy.deepcopy(data_test)
            case _:
                raise ValueError(f'Unknown dataset {args.model_params["dataset"]}')
        constant_dataset = args.model_params['dataset']
        del args.model_params['dataset']

    match args.encoder:
        case 'gemma':
            from nlp import Gemma
            encoder = Gemma()
        case _:
            raise NotImplementedError(f'Unknown encoder {args.encoder}')
    data_train.seed = args.seed
    data_val.seed = args.seed
    data_test.seed = args.seed
    data_train.process_data(encoder, args.graph_density, args.graph_type)
    data_val.process_data(encoder, args.graph_density, args.graph_type)
    data_test.process_data(encoder, args.graph_density, args.graph_type)

    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    model, state = eqx.nn.make_with_state(get_model(args.model))( #type: ignore
        key=subkey,
        cards=data_train.cards,
        use_meta=args.use_meta,
        **args.model_params
    )
    if constant_dataset is not None:
        args.model_params['dataset'] = constant_dataset
    params, static = eqx.partition(model, eqx.is_array)
    args.param_count = sum(
        x.size
        for x in jax.tree_util.tree_leaves(params)
    )
    for param, sign in get_model_params(args.model).items():
        if param in ['self', 'key', 'cards', 'use_meta']: continue
        if param not in args.model_params:
            args.model_params[param] = sign.default
    if verbose > 0:
        print()
        print('#' * 80)
        print()
        print(f'Run #{i_run:02}', args)

    if args.fine_tune is None:
        args.fine_tune_epochs = None

    results = check_cached_model(exp_name, i_run, args)
    if results is not None:
        return results

    fine_tune = args.fine_tune
    fine_tune_epochs = args.fine_tune_epochs
    args.fine_tune = None
    args.fine_tune_epochs = None

    trainer, opt_state, lr_transform, lr_transform_state = prepare_trainer(
        args, model, data_train.n_steps(args.batch_size)
    )
    (
        params, static, state, opt_state, lr_transform_state, key
    ) = trainer.shard_model(
        params, static, state, opt_state, lr_transform_state, key
    )

    results = check_cached_model(exp_name, i_run, args)
    if results is not None:
        params, state, opt_state = eqx.tree_deserialise_leaves(
            f'experiments/cache/{get_hash(args)}/model-{args.model}.eqx',
            (params, state, opt_state)
        )
    else:
        results, params, state, opt_state = train_model(
            args, data_train, data_val, data_test, params, static, state,
            trainer, opt_state, lr_transform, lr_transform_state, key, verbose
        )

    args.fine_tune = fine_tune
    args.fine_tune_epochs = fine_tune_epochs
    hs = get_hash(args)

    if fine_tune is not None:
        assert fine_tune_epochs is not None
        sets = args.dataset['test_set']
        if args.dataset['temporal_split']: # Warning: dataset leak
            sets = args.dataset['train_set']
        data, data_val2 = fine_tuning_dataset(
            sets=tuple(sets),
            n_days=fine_tune,
            verbose=verbose,
        )
        data.process_data(encoder, args.graph_density, args.graph_type)
        data_val2.process_data(encoder, args.graph_density, args.graph_type)
        data.shard_data(trainer)
        data_val2.shard_data(trainer)
        data_test.shard_data(trainer)
        data.set_step_function(static, trainer.train_step)
        data_val2.set_step_function(static, trainer.eval_step)
        data_test.set_step_function(static, trainer.eval_step)

        data.precompile(
            params, state, opt_state, lr_transform_state, args.batch_size, key,
            verbose=verbose, verbose_name='fine-tuning'
        )
        data_val2.precompile(
            params, state, opt_state, lr_transform_state, args.batch_size, key,
            verbose=verbose, verbose_name='validation'
        )
        data_test.precompile(
            params, state, opt_state, lr_transform_state, args.batch_size, key,
            verbose=verbose, verbose_name='test'
        )

        key = jnp.array(results['key'])

        (
            train_losses, val_losses, test_losses, lr_history, key, best_epoch,
            best_state, test_preds, test_trues, test_npicks
        ) = epoch_loop(
            args, fine_tune_epochs, data, data_val2, data_test, params,
            static, state, opt_state, lr_transform, lr_transform_state, key,
            1+args.epochs
        )
        train_losses = results['train_losses'] + train_losses
        val_losses = results['val_losses'] + val_losses
        test_losses = results['test_losses'] + test_losses
        lr_history = results['lr'] + lr_history

        os.makedirs(f'experiments/cache/{hs}', exist_ok=True)
        eqx.tree_serialise_leaves(
            f'experiments/cache/{hs}/model-{args.model}.eqx',
            best_state
        )

        results = {
            'config': args,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'lr': lr_history,
            'key': np.array(key),
            'best_epoch': best_epoch,
            'test_preds': test_preds,
            'test_trues': test_trues,
            'test_npicks': test_npicks,
        }
        with open(f'experiments/cache/{hs}/results.pkl', 'wb') as f:
            pickle.dump(results, f)

    symlink = get_symlink(exp_name, i_run, args.model)
    if os.path.lexists(symlink):
        os.unlink(symlink)
    os.symlink(
        f'../../cache/{hs}/model-{args.model}.eqx',
        symlink
    )

    return results

def main(args: argparse.Namespace):
    data_train, data_val, data_test = make_dataset_from_args(
        args.dataset, verbose=args.verbose
    )
    args.dataset['data_size'] = {
        'train_n_drafts': data_train.drafts.set_id.shape[0],
        'val_n_drafts': data_val.drafts.set_id.shape[0],
        'test_n_drafts': data_test.drafts.set_id.shape[0],
    }
    if args.verbose > 0:
        tmp_args = copy.deepcopy(args)
        del tmp_args.runs #type: ignore
        print('Experiment:', tmp_args)

    os.makedirs('experiments/results', exist_ok=True)
    os.makedirs('experiments/cache', exist_ok=True)
    os.makedirs(f'experiments/results/{args.name}', exist_ok=True)

    results = []
    for i_run, run_args in enumerate(args.runs):
        results.append(run(
            args.name, i_run+1, run_args, data_train, data_val, data_test,
            verbose=args.verbose
        ))

    with open(f'experiments/results/{args.name}/results.pkl', 'wb') as f:
        pickle.dump((args, results), f)

if exp_args is not None:
    main(exp_args)
