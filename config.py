import argparse
import copy
import sys
import time
import json

def override_args(
    args: dict, override: dict | None, path: list | None=None
) -> None:
    if override is None:
        return
    if path is None:
        path = []
    for k, v in override.items():
        if len(path) == 0 and k not in args:
            raise ValueError(f"Unknown argument: {k}")
        if isinstance(v, dict):
            if isinstance(args[k], str):
                args[k] = json.loads(args[k])
            override_args(args[k], v, path+[k])
        else:
            args[k] = v

def parse_exp():
    parser = argparse.ArgumentParser(
        description="MTGateauExperimenter "
        "(Starred arguments can be lists in config file)",
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

    # Experiment Config
    group_data = parser.add_argument_group('Data Config')
    group_data.add_argument('--train-set',
        action='extend', nargs='+', type=str, help='Set(s) to use for training'
    )
    group_data.add_argument('--val-set',
        action='extend', nargs='+', type=str,
        help='Set(s) to use for validation'
    )
    group_data.add_argument('--test-set',
        action='extend', nargs='+', type=str, help='Set(s) to use for testing'
    )
    group_data.add_argument('--temporal-split',
        type=bool, default=False, action=argparse.BooleanOptionalAction,
        help='Split data temporally. Overrides --test-set, using second-to-last'
             ' week for training, and last week for testing'
    )
    group_data.add_argument('--time-window',
        type=int, default=7, help='Time window extracted as datasets, in days'
    )
    group_data.add_argument('--data-head',
        type=int, default=-1,
        help='Use only N samples per dataset, -1 to disable'
    )

    group_archi = parser.add_argument_group('Model Architecture')
    group_archi.add_argument('--model', '-m',
        type=str, default='DraftGraph', choices=[
            'DraftAttention', 'DraftTransformer', 'DraftGraph',
            'LinearRegression'
        ],
        help='*Model'
    )
    group_archi.add_argument('--model-params', '-mp', type=str, default='{}',
        help='Model parameters as a JSON dictionary string, leave empty to use '
             'defaults, use \'h\' instead to list available parameters'
    )
    group_archi.add_argument('--models', type=str, default='[]',
        help='List of model configs. Overrides other settings when defined, '
             'except lists.'
    )
    group_archi.add_argument('--encoder',
        type=str, default='gemma', choices=['gemma', 'bert'], help='Encoder')
    group_archi.add_argument('--tokenizer',
        type=str, default='bert', choices=['bert'],
        help='Tokenizer, Only used if the encoder is BERT'
    )
    group_archi.add_argument('--use-meta',
        type=bool, default=False, action=argparse.BooleanOptionalAction,
        help='Use meta features'
    )
    group_archi.add_argument('--graph-density',
        type=float, default=0.1, help='Similarity graph density'
    )
    group_archi.add_argument('--graph-type',
        type=str, default='knn', choices=['knn', 'global'],
        help='Similarity graph type'
    )

    group_train = parser.add_argument_group('Training Config')
    group_train.add_argument('--epochs',
        type=int, default=10, help='*Number of epochs'
    )
    group_train.add_argument('--batch-size', '-bs',
        type=int, default=32, help='*Batch size')
    group_train.add_argument('--optimizer',
        type=str, default='lion', choices=['adam', 'adamw', 'lion'],
        help='Optimizer'
    )
    group_train.add_argument('--lr',
        type=float, default=1e-6, help='*Learning rate'
    )
    group_train.add_argument('--scheduler',
        type=str, default='adaptive', choices=['warmup_cosine', 'adaptive'],
        help='Scheduler'
    )
    group_train.add_argument('--scheduler-params', type=str, default='{}')
    group_train.add_argument('--loss',
        type=str, default='NLL', choices=['MSE', 'NLL'], help='Loss function'
    )
    group_train.add_argument('--fine-tune', type=int, default=None,
        help='*Fine tune the iteration with best validation loss on the first k'
             ' days of the test set'
    )
    group_train.add_argument('--fine-tune-epochs', type=int, default=10,
        help='Number of fine-tuning epochs'
    )

    group_meta.add_argument('--test-frequency', type=int, default=5)
    group_meta.add_argument('--seed', type=int, default=42, help='*Random seed')

    # Other
    group_other = parser.add_argument_group('Other')
    group_other.add_argument('--conf',
        action='append', help='Use configuration file'
    )
    group_other.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()
    CSI = "\x1b[" #]

    if args.conf is not None:
        import yaml
        for conf_fname in args.conf:
            with open(conf_fname, 'r') as f:
                conf = yaml.safe_load(f)
                for k in conf:
                    if k in ['train_set', 'val_set', 'test_set', 'fine_tune']:
                        continue
                    if parser.get_default(k) is None:
                        raise ValueError(f"Unknown parameter {k}")
                parser.set_defaults(**conf)
        # Reload arguments to override config file values with command line
        # values
        args = parser.parse_args()

    if args.name == '<current datetime>':
        args.name = time.strftime("%Y%m%d-%H%M%S")
    if isinstance(args.models, str):
        args.models = json.loads(args.models)

    if args.data_head != -1:
        raise NotImplementedError(
            f"Data head {args.data_head} is not yet implemented."
        )
    if args.encoder == 'bert':
        raise NotImplementedError(
            f"BERT is not yet implemented."
        )

    if args.train_set is None:
        raise ValueError(
            "At least one training set must be provided."
        )
    if not args.temporal_split:
        if args.test_set is None:
            raise ValueError(
                "At least one test set must be provided if temporal split isn't"
                " enabled."
            )
        if args.val_set is None:
            raise ValueError(
                "At least one validation set must be provided if temporal split"
                " isn't enabled."
            )
    if args.temporal_split:
        args.test_set = args.train_set
        args.val_set = args.train_set

    dataset_hp = [
        'train_set', 'val_set', 'test_set', 'temporal_split', 'time_window',
        'data_head'
    ]

    args.dataset = {
        hp: vars(args)[hp]
        for hp in dataset_hp
    }
    for hp in dataset_hp:
        del vars(args)[hp]

    exp_args_hp = ['name', 'n_gpus', 'dataset', 'loss', 'verbose', 'conf']
    run_args_hp = ['dataset', 'loss']

    exp_args = copy.deepcopy(args)
    for hp in vars(args):
        if not hp in exp_args_hp:
            del vars(exp_args)[hp]

    for hp in exp_args_hp:
        if hp not in run_args_hp:
            del vars(args)[hp]

    models = args.models
    del args.models #type: ignore
    if len(models) > 0:
        runs = []
        for overrides in models:
            run = copy.deepcopy(args)
            if isinstance(run, str):
                run = json.loads(run)
            override_args(vars(run), overrides)
            runs.append(run)
    else: runs = [copy.deepcopy(args)]

    arg_lists = ['seed', 'model', 'epochs', 'batch_size', 'lr', 'fine_tune']
    for hp in arg_lists:
        if isinstance(vars(args)[hp], list):
            base_runs = runs
            runs = []
            for run in base_runs:
                for val in vars(run)[hp]:
                    new_run = copy.deepcopy(run)
                    vars(new_run)[hp] = val
                    runs.append(new_run)

    hyperparameters = {
        'float': ['lr'],
        'json': ['scheduler_params', 'model_params']
    }
    for run in runs:
        for hp in hyperparameters['float']:
            if isinstance(vars(run)[hp], str):
                vars(run)[hp] = float(vars(run)[hp])
        if run.model_params == 'h':
            from models import get_model_params
            print(
                f'{CSI}34mAvailable parameters for '
                f'{CSI}32m{run.model}{CSI}34m:{CSI}0m'
            )
            for param, sign in get_model_params(run.model).items():
                if param in ['self', 'key', 'cards']: continue
                print(
                    f'{CSI}32m{param}{CSI}34m: '
                    f'{CSI}33m{str(sign)[len(str(param))+2:]}{CSI}0m'
                )
            sys.exit()
        for hp in hyperparameters['json']:
            if isinstance(vars(run)[hp], str):
                vars(run)[hp] = json.loads(vars(run)[hp])
    exp_args.runs = runs

    return exp_args

def parse_plot():
    parser = argparse.ArgumentParser(
        description="Plotter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str)
    group_meta = parser.add_argument_group('Plot Config')
    group_meta.add_argument('--width', type=int, default=16)
    group_meta.add_argument('--height', type=int, default=9)
    group_meta.add_argument('--dpi', type=int, default=300)
    group_meta.add_argument('--legend-fontsize', type=int, default=-1)
    group_meta.add_argument('--cmap', type=str, default='tab10')
    group_meta.add_argument('--y-max', type=float, default=None)
    group_meta.add_argument('--y-min', type=float, default=None)
    group_meta.add_argument('--x-max', type=int, default=None)
    group_meta.add_argument('--exclude-outliers',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--only-test-indicators',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--mark-lines',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--plot-lr',
        type=bool, default=True, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--average-runs',
        type=bool, default=False, action=argparse.BooleanOptionalAction,
        help='Average over runs with the same config but different seeds'
    )
    group_meta.add_argument('--smooth-window', type=int, default=1)
    group_meta.add_argument('--focus-valid',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--focus-test',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--skip-batch',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--show-param-count',
        type=bool, default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()
    assert args.smooth_window > 0 and args.smooth_window % 2 == 1
    return args
