import argparse
import os
import sys
import time

args = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MTGateauExperimenter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment Metadata
    group_meta = parser.add_argument_group('Experiment Metadata')
    group_meta.add_argument('--name',
        type=str, default='<current datetime>', help='Experiment Name'
    )
    group_meta.add_argument('--n-gpus','-d', 
        type=int, default=2, help='Number of GPUs'
    )
    group_meta.add_argument('--test-frequency', type=int, default=5)
    group_meta.add_argument('--seed', type=int, default=42, help='Random seed')

    # Experiment Config
    group_data = parser.add_argument_group('Data Config')
    group_data.add_argument('--train-set',
        action='extend', nargs='+', type=str, help='Set(s) to use for training'
    )
    group_data.add_argument('--val-set',
        action='extend', nargs='+', type=str, help='Set(s) to use for validation'
    )
    group_data.add_argument('--test-set',
        action='extend', nargs='+', type=str, help='Set(s) to use for testing'
    )
    group_data.add_argument('--temporal-split',
        type=bool, default=False, action=argparse.BooleanOptionalAction,
        help='Split data temporally. Overrides --test-set, using second-to-last '
             'week for training, and last week for testing'
    )
    group_data.add_argument('--time-window',
        type=int, default=7, help='Time window extracted as datasets, in days'
    )
    group_data.add_argument('--data-head',
        type=int, default=-1, help='Use only N samples per dataset, -1 to disable'
    )
    group_data.add_argument('--use-meta',
        type=bool, default=False, action=argparse.BooleanOptionalAction,
        help='Use meta features'
    )
    group_data.add_argument('--graph_density',
        type=float, default=0.1, help='Similarity graph density'
    )
    group_data.add_argument('--graph_type',
        type=str, default='knn', choices=['knn', 'global'],
        help='Similarity graph type'
    )

    group_archi = parser.add_argument_group('Model Architecture')
    group_archi.add_argument('--encoder',
        type=str, default='gemma', choices=['gemma', 'bert'], help='Encoder')
    group_archi.add_argument('--tokenizer',
        type=str, default='bert', choices=['bert'],
        help='Tokenizer, Only used if the encoder is BERT'
    )
    group_archi.add_argument('--model', '-m',
        type=str, default='DraftGraph', choices=[
            'DraftAttention', 'DraftTransformer', 'DraftGraph',
            'LinearRegression'
        ],
        help='Model'
    )
    group_archi.add_argument('--model-params', '-mp', type=str, default='{}',
        help='Model parameters as a JSON string, leave empty to use defaults, use '
             '\'h\' to list available parameters'
    )

    group_train = parser.add_argument_group('Training Config')
    group_train.add_argument('--epochs',
        type=int, default=10, help='Number of epochs'
    )
    group_train.add_argument('--batch-size', '-bs',
        type=int, default=32, help='Batch size')
    group_train.add_argument('--optimizer',
        type=str, default='lion', choices=['adam', 'adamw', 'lion'],
        help='Optimizer'
    )
    group_train.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    group_train.add_argument('--scheduler-params', type=str, default='{}')
    group_train.add_argument('--warmup',
        type=float, default=0.1, help='Percentage of warmup.'
    )
    group_train.add_argument('--loss',
        type=str, default='NLL', choices=['MSE', 'NLL'], help='Loss function'
    )

    # Other
    group_other = parser.add_argument_group('Other')
    group_other.add_argument('--conf',
        action='append', help='Use configuration file'
    )
    group_other.add_argument('--verbose', '-v',
        type=bool, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    CSI = "\x1b[" #]

    if args.conf is not None:
        import yaml
        for conf_fname in args.conf:
            with open(conf_fname, 'r') as f:
                conf = yaml.safe_load(f)
                parser.set_defaults(**conf)
        # Reload arguments to override config file values with command line values
        args = parser.parse_args()

    if args.data_head != -1:
        raise NotImplementedError(
            f"Data head {args.data_head} is not yet implemented."
        )
    if args.encoder == 'bert':
        raise NotImplementedError(
            f"BERT is not yet implemented."
        )

    if args.name == '<current datetime>':
        args.name = time.strftime("%Y%m%d-%H%M%S")
    if args.train_set is None:
        raise ValueError(
            "At least one training set must be provided."
        )
    if not args.temporal_split:
        if args.test_set is None:
            raise ValueError(
                "At least one test set must be provided if temporal split isn't enabled."
            )
        if args.val_set is None:
            raise ValueError(
                "At least one validation set must be provided if temporal split isn't enabled."
            )

    def print_model_params(model_name):
        from models import get_model_params
        print(f'{CSI}34mAvailable parameters for {CSI}32m{model_name}{CSI}34m:{CSI}0m')
        for param, sign in get_model_params(model_name).items():
            if param in ['self', 'key', 'cards']: continue
            print(f'{CSI}32m{param}{CSI}34m: {CSI}33m{str(sign)[len(str(param))+2:]}{CSI}0m')
    if args.model_params == 'h':
        print_model_params(args.model)
        sys.exit()
    if isinstance(args.model_params, str):
        import json
        args.model_params = json.loads(args.model_params)
    if isinstance(args.scheduler_params, str):
        import json
        args.scheduler_params = json.loads(args.scheduler_params)
    if args.temporal_split:
        args.test_set = args.train_set
        args.val_set = args.train_set

    hyperparameters = {
        'float': ['lr'],
        'categorical': [], #['model', 'encoder'],
    }
    all_hp = hyperparameters['float'] + hyperparameters['categorical']
    for hp in hyperparameters['float']:
        if isinstance(vars(args)[hp], list):
            vars(args)[hp] = list(map(float, vars(args)[hp]))
    if isinstance(args.model_params, list):
        if isinstance(args.model, str):
            args.model = [args.model] * len(args.model_params)
        if (
            not isinstance(args.model, list)
            or len(args.model) != len(args.model_params)
        ):
            raise ValueError(
                "Shape of --model-params must match --model."
            )
        help_exit = False
        for i in range(len(args.model_params)):
            if args.model_params[i] is None:
                args.model_params[i] = {}
            if args.model_params[i] == 'h':
                help_exit = True
                print_model_params(args.model[i])
        if help_exit:
            sys.exit()

    from gpu_management import set_gpus

    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform' 
    set_gpus(args.n_gpus, forcing=True)

import copy
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pickle
import optax
from tqdm import tqdm
from typing import Optional

from data import JaxDraftDataset, make_dataset_from_args
from models import get_model, get_model_params
from training import Trainer

# os.makedirs('cache/jax', exist_ok=True)
# jax.config.update("jax_compilation_cache_dir", 'cache/jax')
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

def base_args(args):
    new_args = copy.deepcopy(vars(args))
    del new_args['name']
    del new_args['n_gpus']
    # del new_args.epochs # TODO
    del new_args['conf']
    del new_args['verbose']
    return new_args

__cnt = 0

def run(
    args: argparse.Namespace,
    data_train: Optional[JaxDraftDataset]=None,
    data_val: Optional[JaxDraftDataset]=None,
    data_test: Optional[JaxDraftDataset]=None
):
    global __cnt
    if isinstance(args.encoder, list):
        results = []
        for i, encoder in enumerate(args.encoder):
            new_args = copy.deepcopy(args)
            new_args.encoder = encoder
            results.append(run(new_args))
        return 'encoder', results
    
    if isinstance(args.use_meta, list):
        results = []
        for i, use_meta in enumerate(args.use_meta):
            new_args = copy.deepcopy(args)
            new_args.use_meta = use_meta
            results.append(run(new_args))
        return 'use_meta', results

    if data_train is None or data_val is None or data_test is None:
        data_train, data_val, data_test = make_dataset_from_args(args)
        args.data_size = {
            'train_n_drafts': data_train.drafts.set_id.shape[0],
            'val_n_drafts': data_val.drafts.set_id.shape[0],
            'test_n_drafts': data_test.drafts.set_id.shape[0],
        }

    if isinstance(args.model, list):
        results = []
        for i, model in enumerate(args.model):
            new_args = copy.deepcopy(args)
            new_args.model = model
            if isinstance(args.model_params, list):
                new_args.model_params = args.model_params[i]
            results.append(run(new_args, data_train, data_val, data_test))
        return 'model', results
    for hp in all_hp:
        if isinstance(vars(args)[hp], list):
            results = []
            for val in vars(args)[hp]:
                new_args = copy.deepcopy(args)
                vars(new_args)[hp] = val
                results.append(run(new_args, data_train, data_val, data_test))
            return hp, results

    lr = args.lr
    if 'lr' in args.model_params: # Allow for per-model lr
        lr = float(args.model_params['lr'])
        del args.model_params['lr']
        args.lr = lr
    if 'scheduler_params' in args.model_params:
        for k, v in args.model_params['scheduler_params'].items():
            args.scheduler_params[k] = v
        del args.model_params['scheduler_params']

    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    model, state = eqx.nn.make_with_state(get_model(args.model))(
        key=subkey,
        cards=data_train.cards,
        **args.model_params
    )
    params, static = eqx.partition(model, eqx.is_array)
    args.model_params['lr'] = lr
    args.param_count = sum(
        x.size
        for x in jax.tree_util.tree_leaves(params)
    )
    for  param, sign in get_model_params(args.model).items():
        if param in ['self', 'key', 'cards']: continue
        if param not in args.model_params:
            args.model_params[param] = sign.default
    print(args)
    __cnt += 1
    hs = hashlib.sha256(
        json.dumps(base_args(args), sort_keys=True).encode('utf-8')
    ).hexdigest()
    symlink = f'experiments/results/{args.name}/model-{__cnt:02}-{args.model}.eqx'
    if os.path.exists(f'experiments/cache/{hs}'):
        with open(f'experiments/cache/{hs}/results.pkl', 'rb') as f:
            results = pickle.load(f)
            if base_args(args) != base_args(results['config']):
                raise ValueError(
                    "Cached results do not match args."
                )
        print(f'{CSI}34mUsing cached results for{CSI}0m {hs}')
        if os.path.lexists(symlink):
            os.unlink(symlink)
        os.symlink(
            f'experiments/cache/{hs}/model-{args.model}.eqx',
            symlink
        )
        return results


    # total_steps = data_train.n_steps() * args.epochs
    # scheduler = optax.warmup_cosine_decay_schedule(
    #     init_value=lr / 10, peak_value=lr,
    #     warmup_steps=int(total_steps*args.warmup),
    #     decay_steps=total_steps, end_value=lr / 1e4
    # )
    if args.optimizer == 'lion':
        tx = optax.lion(learning_rate=lr)
    elif args.optimizer == 'adam':
        tx = optax.adam(learning_rate=lr)
    elif args.optimizer == 'adamw':
        tx = optax.adamw(learning_rate=lr)
    else:
        raise NotImplementedError(f'Unknown optimizer {args.optimizer}')
    trainer = Trainer(tx, loss=args.loss)
    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

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

    (
        params, static, state, opt_state, lr_transform_state, key
    ) = trainer.shard_model(
        params, static, state, opt_state, lr_transform_state, key
    )
    data_train.shard_data(trainer)
    data_val.shard_data(trainer)
    data_test.shard_data(trainer)
    data_train.set_step_function(static, trainer.train_step)
    data_val.set_step_function(static, trainer.eval_step)
    data_test.set_step_function(static, trainer.eval_step)

    st = time.time()
    data_train.precompile(params, state, opt_state, lr_transform_state, key)
    if args.verbose:
        print(f'Precompiled training loop in {time.time() - st:.6f}s')
    st = time.time()
    data_val.precompile(params, state, opt_state, lr_transform_state, key)
    if args.verbose:
        print(f'Precompiled validation loop in {time.time() - st:.6f}s')
    st = time.time()
    data_test.precompile(params, state, opt_state, lr_transform_state, key)
    if args.verbose:
        print(f'Precompiled test loop in {time.time() - st:.6f}s')

    best_state = (params, state, opt_state)
    best_val_loss = jnp.inf

    train_losses = []
    val_losses = []
    test_losses = []
    last_val_loss = 'N/A'
    last_test_loss = 'N/A'
    lr_history = [lr]
    for i_epoch in (pbar:=tqdm(range(args.epochs))):
        if i_epoch > 0:
            import sys
            sys.exit()
        params, state, opt_state, lr_transform_state, key, losses = data_train.run_batches(
            params, state, opt_state, lr_transform_state,
            key
        )
        train_losses.append(losses)
        pbar.set_description(
            f"Loss: {losses.mean():.6f} [Val: {last_val_loss}, Test: {last_test_loss}]"
        )

        params, state, opt_state, lr_transform_state, key, losses = data_val.run_batches(
            params, state, opt_state, lr_transform_state,
            key
        )
        val_losses.append(losses)
        if losses.mean() <= best_val_loss:
            best_val_loss = losses.mean()
            best_state = (params, state, opt_state)
        last_val_loss = f'{losses.mean():.6f}'
        pbar.set_description(
            f"Loss: {train_losses[-1].mean():.6f} [Val: {last_val_loss}, Test: {last_test_loss}]"
        )
        _, lr_transform_state = lr_transform.update(
            updates=eqx.filter(eqx.combine(params,static),eqx.is_inexact_array),
            state=lr_transform_state, value=losses.mean()
        )
        lr_history.append(lr * lr_transform_state.scale) #type: ignore
        if len(lr_history) > 1 and lr_history[-1] != lr_history[-2]:
            params, state, opt_state = best_state

        if (args.epochs-i_epoch-1) % args.test_frequency != 0: continue
        params, state, opt_state, lr_transform_state, key, losses = data_test.run_batches(
            params, state, opt_state, lr_transform_state,
            key
        )
        test_losses.append(losses)
        last_test_loss = f'{losses.mean():.6f}'
        pbar.set_description(
            f"Loss: {train_losses[-1].mean():.6f} [Val: {last_val_loss}, Test: {last_test_loss}]"
        )
        if jnp.isnan(train_losses[-1].mean()):
            while len(train_losses) < args.epochs:
                train_losses.append(train_losses[-1])
            while len(val_losses) < args.epochs:
                val_losses.append(val_losses[-1])
            while len(test_losses) < (args.epochs + args.test_frequency - 1) // args.test_frequency:
                test_losses.append(test_losses[-1])
            break

    os.makedirs(f'experiments/cache/{hs}', exist_ok=True)
    eqx.tree_serialise_leaves(
        f'experiments/cache/{hs}/model-{args.model}.eqx',
        best_state[:2]
    )

    if os.path.lexists(symlink):
        os.unlink(symlink)
    os.symlink(
        f'experiments/cache/{hs}/model-{args.model}.eqx',
        symlink
    )

    train_losses = np.array(jnp.concat(train_losses))
    val_losses = np.array(jnp.concat(val_losses))
    test_losses = np.array(jnp.concat(test_losses))
    result = {
        'config': args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'lr': np.array(lr_history),
    }
    with open(f'experiments/cache/{hs}/results.pkl', 'wb') as f:
        pickle.dump(result, f)
    return result

def main(args: argparse.Namespace):
    global __cnt

    os.makedirs('experiments/results', exist_ok=True)
    os.makedirs('experiments/cache', exist_ok=True)
    os.makedirs(f'experiments/results/{args.name}', exist_ok=True)

    __cnt = 0
    results = run(args)

    with open(f'experiments/results/{args.name}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    assert args is not None
    main(args)
