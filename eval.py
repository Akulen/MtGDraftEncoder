import os
from gpu_management import set_gpus

os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
set_gpus(1, forcing=True)

import time
import pickle
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Tuple
from jaxtyping import Array, Float, Int, PRNGKeyArray

from models import DraftWRPredictor, DraftGraph
from data_types import Cards, Sets, Drafts
from data import JaxDraftDataset, make_dataset_from_args

def make_query(drafts: Drafts) -> Tuple[Drafts, Int[Array, "45 15"]]:
    packs = drafts.packs
    drafts = jax.tree.map(lambda x: jnp.repeat(x, 45*15).reshape((45, 15) + x.shape), drafts)
    X, Y = jnp.meshgrid(jnp.arange(45), jnp.arange(15), indexing='ij')
    drafts = eqx.tree_at(
        lambda d: d.picks,
        drafts,
        drafts.picks.at[X, Y, X].set(packs)
    )
    return drafts, jnp.zeros((45, 15), dtype=jnp.int32)

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
    DraftWRPredictor, eqx.nn.State, Any, Float[Array, "n_drafts 45 15"]
]:
    query, step_ind = jax.vmap(make_query)(drafts)
    query = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), query)
    keys = jax.random.split(key, query.set_id.shape[0])
    outputs, _ = jax.vmap(
        model, in_axes=(0, None, None, 0, None, None)
    )(keys, cards, sets, query, state, True)
    return model, state, None, outputs[step_ind]

def card_rankings(
    params: DraftWRPredictor,
    static: DraftWRPredictor,
    state: eqx.nn.State,
    data: JaxDraftDataset,
    key: PRNGKeyArray
) -> Float[Array, "n_drafts 45 15"]:
    data.batch_size //= 32
    data.set_step_function(static, eval_fn)
    data.precompile(params, state, None, None, key)
    st = time.time()
    params, state, _, _, key, outputs = data.run_batches(
        params, state, None, None, key
    )
    print(f'Ranked all drafts in {time.time() - st:.6f}s')
    return outputs

def main():
    path = "experiments/results/SameSet"
    with open(f'{path}/results.pkl', 'rb') as f:
        results = pickle.load(f)
        results = results[1][0]
        args = results['config']
        print(args)
    data_train, data_val, data_test = make_dataset_from_args(args)
    model_params = args.model_params
    del model_params['lr']
    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    model, state = eqx.nn.make_with_state(DraftGraph)(
        key=subkey, cards=data_train.cards, **model_params
    )
    params, static = eqx.partition(model, eqx.is_array)
    params, state = eqx.tree_deserialise_leaves(
        path+'/model-00-DraftGraph.eqx',
        (params, state)
    )
    key, subkey = jax.random.split(key)
    rankings = card_rankings(params, static, state, data_test, key)

if __name__ == '__main__':
    main()
