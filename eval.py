import os

if __name__ == "__main__":
    from gpu_management import set_gpus

    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    set_gpus(2, forcing=True)

import time
import pickle
import jax
import jax.numpy as jnp
import copy
import numpy as np
import equinox as eqx
from typing import cast, Any, Tuple
from jaxtyping import Array, Float, Bool, Int, PRNGKeyArray
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import top_k_accuracy_score

from nlp import get_encoder
from models import get_model, DraftWRPredictor
from training import Trainer
from experiment import prepare_trainer
from data_types import Cards, Sets, Drafts
from data import JaxDraftDataset, make_dataset_from_args

def make_query(drafts: Drafts) -> Drafts:
    packs = drafts.packs
    drafts = jax.tree.map(
        lambda x: jnp.tile(
            x, (45, 15) + tuple(1 for _ in x.shape)
        ).reshape((45, 15) + x.shape),
        drafts
    )
    X, Y = jnp.meshgrid(jnp.arange(45), jnp.arange(15), indexing='ij')
    drafts = eqx.tree_at(
        lambda d: d.picks,
        drafts,
        drafts.picks.at[X, Y, X].set(packs)
    )
    return drafts

def eval_fn(
    model: DraftWRPredictor,
    state: eqx.nn.State,
    _: Any,
    __: Any,
    cards: Cards,
    sets: Sets,
    drafts: Drafts,
    key: PRNGKeyArray
) -> Tuple[
    DraftWRPredictor,
    eqx.nn.State,
    Any,
    Float[Array, "n_drafts 45 15"],
    Float[Array, "bs 45"],
    Int[Array, "bs 2"],
    Bool[Array, "bs 45"]
]:
    batch_size = drafts.set_id.shape[0]
    query = jax.vmap(make_query)(drafts)
    query = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), query)
    keys = jax.random.split(key, query.set_id.shape[0])
    outputs, _ = jax.vmap(
        model, in_axes=(0, None, None, 0, None, None)
    )(keys, cards, sets, query, state, True)
    outputs = jnp.diagonal(
        outputs.reshape(-1, 45, 15, 45),
        axis1=1,
        axis2=3
    ).transpose(0, 2, 1)
    outputs = jnp.where(drafts.packs > 0, outputs, jnp.nan)
    outputs = cast(Float[Array, "n_drafts 45 15"], outputs)
    return (
        model, state, None, outputs, jnp.zeros((0, 45)),
        jnp.zeros((0, 2), dtype=jnp.int32), jnp.zeros((0, 45), dtype=jnp.bool)
    )
eval_jit = eqx.filter_jit(eval_fn)

def prepare_rankings(
    params: DraftWRPredictor,
    static: DraftWRPredictor,
    state: eqx.nn.State,
    batch_size: int,
    data: JaxDraftDataset,
    trainer: Trainer,
    key: PRNGKeyArray
):
    st = time.time()
    data.shard_data(trainer)
    data.set_step_function(static, eval_fn)
    data.precompile(params, state, None, None, batch_size, key)
    print(f'Precompiled ranking in {time.time() - st:.6f}s')

def card_rankings(
    params: DraftWRPredictor,
    state: eqx.nn.State,
    batch_size: int,
    data: JaxDraftDataset,
    key: PRNGKeyArray
) -> Float[Array, "n_drafts 45 15"]:
    st = time.time()
    params, state, _, _, key, outputs, _, _, _ = data.run_batches(
        params, state, None, None, batch_size, False, key
    )
    print(f'Ranked all drafts in {time.time() - st:.6f}s')
    return outputs.reshape(-1, 45, 15)[:data.drafts.set_id.shape[0]]

def rank_draft(
    params: DraftWRPredictor,
    static: DraftWRPredictor,
    state: eqx.nn.State,
    data: JaxDraftDataset,
    draft_id: int,
    trainer: Trainer,
    key: PRNGKeyArray
) -> Float[Array, "n_drafts 45 15"]:
    drafts = data.drafts
    data.drafts = data.drafts[draft_id:draft_id+1]
    prepare_rankings(params, static, state, 1, data, trainer, key)
    result = card_rankings(params, state, 1, data, key)
    data.drafts = drafts
    return result

def compute_all_rankings(path, i_model):
    if os.path.exists(f'{path}/rankings_test.pkl'):
        with open(f'{path}/rankings_test.pkl', 'rb') as f:
            return pickle.load(f)
    with open(f'{path}/results.pkl', 'rb') as f:
        exp_args, results = pickle.load(f)
        results = results[i_model]
        args = results['config']
        print(args)
    data_train, data_val, data_test = make_dataset_from_args(args.dataset)

    encoder = get_encoder(args.encoder)

    data_train.process_data(encoder, args.graph_density, args.graph_type)
    data_val.process_data(encoder, args.graph_density, args.graph_type)
    data_test.process_data(encoder, args.graph_density, args.graph_type)

    model_params = args.model_params
    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    model, state = eqx.nn.make_with_state(get_model(args.model))( #type: ignore
        key=subkey, cards=data_train.cards, use_meta=args.use_meta,
        **model_params
    )
    params, static = eqx.partition(model, eqx.is_array)
    trainer, opt_state, lr_transform, lr_transform_state = prepare_trainer(
        args, model, data_train.n_steps(args.batch_size)
    )
    static, key = trainer.shard_model(static, key)
    params, state, opt_state = eqx.tree_deserialise_leaves(
        path+f'/model-{i_model+1:02}-{args.model}.eqx',
        (params, state, opt_state)
    )
    key, subkey = jax.random.split(key)
    print(data_test.drafts.set_id.shape)
    # data_test.drafts = data_test.drafts[:1024]
    n_devices = len(jax.devices())
    batch_size = max(1, (args.batch_size * n_devices) // 4)
    prepare_rankings(
        params, static, state, batch_size, data_test,
        trainer, key
    )
    rankings = card_rankings(
        params, state, batch_size, data_test, key
    )
    with open(f'{path}/rankings_test.pkl', 'wb') as f:
        pickle.dump((data_test.drafts, rankings), f)
    return data_test.drafts, rankings

def top_eps_accuracy_score(y_true, y_scores, eps=0.01):
    assert y_true.shape == y_scores.shape[:-1]
    y_true = y_true.reshape(-1)
    y_scores = y_scores.reshape(-1, y_scores.shape[-1])
    maxs = y_scores.max(axis=1)
    is_top = y_scores[np.arange(y_scores.shape[0]), y_true] > maxs - eps
    return is_top.mean()

def top_k_accuracy_stats(y_true, y_scores, k, n_picks=15, weight=None):
    y_true_scores = y_scores[np.arange(y_scores.shape[0]), y_true]
    y_scores = copy.deepcopy(y_scores)
    y_scores.sort(axis=1)
    y_scores = y_scores[:,::-1]
    is_top = np.argmax(y_scores == y_true_scores[:,None], axis=1) < k
    acc = is_top.reshape(-1, n_picks).mean(axis=1)
    mean = np.average(acc, weights=weight)
    var = np.average((acc - mean)**2, weights=weight)
    return mean, var

def main():
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 18

    path = "experiments/results/bertram"
    drafts, rankings = compute_all_rankings(path, 0)
    pick_pos = np.argmax(
        drafts.packs == drafts.picks[:, :, None],
        axis=2
    )
    rankings = np.array(rankings)
    rankings = np.where(np.isnan(rankings), -1, rankings)
    n_picks = (rankings[0,0] >= 0).sum() * 3

    rank_dict = {
        0: 'null',
        1: 'bronze',
        2: 'silver',
        3: 'gold',
        4: 'platinum',
        5: 'diamond',
        6: 'mythic',
    }

    X = np.logspace(-3, -1.5, 101)
    for rank in range(1, 7):
        picks = pick_pos[drafts.rank == rank,:n_picks]
        ranks = rankings[drafts.rank == rank,:n_picks]
        if picks.shape[0] == 0:
            continue
        Y = np.array([
            top_eps_accuracy_score(
                picks,
                ranks,
                x
            )
            for x in X
        ])
        plt.plot(X, Y, label=rank_dict[rank])
    plt.xscale('log')
    plt.legend()
    plt.savefig(f'{path}/accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    wrs = np.unique(drafts.player_wr)
    wrs.sort()
    wr_mask = (0.4 <= drafts.player_wr) & (drafts.player_wr <= 0.7)
    wrs = wrs[(0.4 <= wrs) & (wrs <= 0.7)]
    for k in [1, 3]:
        print(k, top_k_accuracy_score(
            pick_pos[wr_mask,:n_picks].reshape(-1),
            rankings[wr_mask,:n_picks].reshape(-1, 15),
            k=k,
            labels=np.arange(15)
        ))
        means, vars = [], []
        for wr in wrs:
            mask = drafts.player_wr == wr
            m = top_k_accuracy_score(
                pick_pos[mask,:n_picks].reshape(-1),
                rankings[mask,:n_picks].reshape(-1, 15),
                k=k,
                labels=np.arange(15)
            )
            means.append(m)
            # vars.append(v)
        plt.plot(
            wrs,
            means,
            label=f'k={k}'
        )
        # plt.fill_between(
        #     wrs,
        #     np.array(means) - np.sqrt(np.array(vars)),
        #     np.array(means) + np.sqrt(np.array(vars)),
        #     alpha=0.2
        # )
    plt.xlabel('Player Win-Rate')
    plt.ylabel('Top-k Accuracy')
    plt.legend()
    plt.savefig(f'{path}/accuracy_wr.png', dpi=300, bbox_inches='tight')
    plt.close()

    sns.kdeplot(rankings[rankings >= 0].reshape(-1))
    plt.savefig(f'{path}/wr_density.png', dpi=300, bbox_inches='tight')
    plt.close()

    sns.kdeplot(drafts.player_wr.reshape(-1))
    plt.savefig(f'{path}/wr_density_wr.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
