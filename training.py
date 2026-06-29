import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from typing import assert_never, Any, Callable, List, Literal, Optional, Tuple
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from data_types import Cards, Sets, Drafts
from models import DraftWRPredictor

Loss = Callable[
    [Float[Array, "bs"], Float[Array, "bs"], Optional[Bool[Array, "bs"]]],
    Float[Array, ""]
]
LossWR = Callable[
    [Float[Array, "bs"], Int[Array, "bs 2"], Optional[Bool[Array, "bs"]]],
    Float[Array, ""]
]

# PICK_WEIGHT = (jnp.arange(1, 46) / 45) ** 2
PICK_WEIGHT = (jnp.arange(1, 46) / 45)

def MSE_loss(
    y_pred: Float[Array, "bs 45"],
    y_true: Float[Array, "bs"],
    mask: Optional[Bool[Array, "bs 45"]]=None
) -> Float[Array, ""]:
    if mask is None:
        mask = jnp.ones_like(y_pred)
    mask *= PICK_WEIGHT
    return ((y_pred - y_true.reshape((-1, 1))) ** 2 * mask).sum() / mask.sum()

def MSE_wr_loss(
    y_pred: Float[Array, "bs 45"],
    y_true: Int[Array, "bs 2"],
    mask: Optional[Bool[Array, "bs 45"]]=None
) -> Float[Array, ""]:
    return MSE_loss(y_pred, y_true[:,0] / y_true.sum(axis=-1), mask)

pascal = np.zeros((9, 7), dtype=np.int32)
pascal[0, 0] = 1
for i in range(1, 9):
    pascal[i, 0] = pascal[i-1, 0]
    for j in range(1, 7):
        pascal[i, j] = pascal[i-1, j] + pascal[i-1, j-1]
pascal = jnp.array(pascal)

def NLL_wr(
    y_pred: Float[Array, "bs n_picks"],
    y_true: Int[Array, "bs 2"],
    eps=1e-4
) -> Float[Array, "bs n_picks"]:
    W = y_true[:,0,None]
    L = y_true[:,1,None]
    p = jnp.clip(y_pred, eps, 1-eps)
    log_p_win_cap = (
        jnp.log(pascal[6 + L, 6]) +
        7 * jnp.log(p) +
        L * jnp.log1p(-p)
    )
    log_p_loss_cap = (
        jnp.log(pascal[W + 2, 2]) +
        W * jnp.log(p) +
        3 * jnp.log1p(-p)
    )
    biased_log_p = jnp.where(
        (y_true[:,0] == 7).reshape((-1, 1)),
        log_p_win_cap,
        log_p_loss_cap
        #   pascal[6+y_true[:,1], 6].reshape((-1, 1))
        # * y_pred**7 * (1-y_pred)**y_true[:,1].reshape((-1, 1)), # 7 Wins
        #   pascal[2+y_true[:,0], 2].reshape((-1, 1))
        # * (1-y_pred)**3 * y_pred**y_true[:,0].reshape((-1, 1)) # 3 Losses
    )
    return -biased_log_p

def masked_mean(
    x: Float[Array, "a b"],
    mask: Optional[Bool[Array, "a b"]]=None,
    weight: Optional[Float[Array, "b"]]=None
) -> Float[Array, ""]:
    if mask is None:
        mask = jnp.ones_like(x)
    if weight is not None:
        mask *= PICK_WEIGHT
    return (x * mask).sum() / mask.sum()


def NLL_wr_loss(
    y_pred: Float[Array, "bs 45"],
    y_true: Int[Array, "bs 2"],
    mask: Optional[Bool[Array, "bs 45"]]=None,
    eps=1e-4
) -> Float[Array, ""]:
    return masked_mean(NLL_wr(y_pred, y_true, eps), mask, PICK_WEIGHT)

def KL_wr_loss(
    y_pred: Float[Array, "bs 45"],
    y_true: Int[Array, "bs 2"],
    mask: Optional[Bool[Array, "bs 45"]]=None,
    eps=1e-4
) -> Float[Array, ""]:
    nll_pred = NLL_wr(y_pred, y_true, eps)
    nll_true = NLL_wr((y_true[:,0] / y_true.sum(axis=1))[:,None], y_true, eps)
    return masked_mean(nll_pred - nll_true, mask, PICK_WEIGHT)

LOSSES = {
    'MSE': MSE_loss,
}
LOSSES_WR = {
    'MSE': MSE_wr_loss,
    'NLL': NLL_wr_loss,
    'KL': KL_wr_loss,
}

class Trainer:
    def __init__(
        self,
        tx: optax.GradientTransformation,
        loss: Loss | LossWR | Literal['MSE'] | List[Loss | LossWR | Literal['MSE']]=MSE_loss,
        target: Literal['wr', 'pwr']='wr',
        n_devices: Optional[int]=None
    ):
        if n_devices is None:
            n_devices = jax.device_count()
        self.mesh = jax.make_mesh((n_devices,), ("batch",))
        self.model_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )
        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, "batch")
        )
        self.compute_loss_grad = eqx.filter_grad(
            self.compute_loss, has_aux=True
        )
        self.tx = tx
        if not isinstance(loss, list):
            loss = [loss]
        self.loss = []
        def wrapper(f):
            if target == 'wr':
                def foo(y, drafts, mask=None):
                    return f(y, drafts.game_outcome, mask)
            elif target == 'pwr':
                assert False # fix compute_loss output
                def foo(y, drafts, mask=None):
                    return f(y, drafts.pwr, mask)
            else:
                assert_never(target)
            return foo

        for l in loss:
            l_fn = None
            if isinstance(l, str):
                L = LOSSES_WR if target == 'wr' else LOSSES
                if l in L:
                    l_fn = L[l]
                else:
                    raise NotImplementedError(f'Loss {l} not implemented for target {target}')
            else:
                l_fn = l
            assert l_fn is not None
            self.loss.append(wrapper(l_fn))

    def shard_model(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.model_sharding)

    def shard_data(self, *data):
        if len(data) == 1:
            data = data[0]
        return eqx.filter_shard(data, self.data_sharding)

    def compute_loss(
        self,
        model: DraftWRPredictor,
        state: eqx.nn.State,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        key: PRNGKeyArray,
        inference: bool=False
    ) -> Tuple[
        Float[Array, ""],
        Tuple[
            Float[Array, "L"], Float[Array, "bs 45"], Float[Array, "bs 2"],
            Bool[Array, "bs 45"], eqx.nn.State
        ]
    ]:
        batch_size = drafts.picks.shape[0]
        keys = jax.random.split(key, batch_size)
        outputs, state = jax.vmap(
            model,
            in_axes=(0, None, None, 0, None, None)
        )(keys, cards, sets, drafts, state, inference)
        mask = drafts.picks != 0
        loss = []
        for l in self.loss:
            loss.append(l(outputs, drafts, mask))
        return loss[0], (
            jnp.stack(loss),
            outputs,
            drafts.game_outcome, # Fix this for pwr
            mask,
            state
        )

    @eqx.filter_jit(donate="all")
    def train_step(
        self,
        model: DraftWRPredictor,
        state: eqx.nn.State,
        opt_state: Any,
        lr_transform_state: Any,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        key: PRNGKeyArray
    ) -> Tuple[
        DraftWRPredictor,
        eqx.nn.State,
        Any,
        Float[Array, "L"],
        Float[Array, "bs 45"],
        Float[Array, "bs 2"],
        Bool[Array, "bs 45"]
    ]:
        model, state, opt_state, cards = self.shard_model(
            model, state, opt_state, cards
        )
        grads, (loss, pred, true, mask, state) = self.compute_loss_grad(
            model, state, cards, sets, drafts, key
        )

        updates, opt_state = self.tx.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        updates = optax.tree.scale(lr_transform_state.scale, updates)

        model = eqx.apply_updates(model, updates)
        model, state, opt_state = self.shard_model(
            model, state, opt_state
        )
        return model, state, opt_state, loss, pred[:0], true[:0], mask[:0]

    @eqx.filter_jit(donate="all")
    def eval_step(
        self,
        model: DraftWRPredictor,
        state: eqx.nn.State,
        opt_state: Any,
        _: Any,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        key: PRNGKeyArray
    ) -> Tuple[
        DraftWRPredictor,
        eqx.nn.State,
        Any,
        Float[Array, "L"],
        Float[Array, "bs 45"],
        Float[Array, "bs 2"],
        Bool[Array, "bs 45"]
    ]:
        model, state, opt_state, cards = self.shard_model(
            model, state, opt_state, cards
        )
        _, (loss, pred, true, mask, _) = self.compute_loss(
            model, state, cards, sets, drafts, key, inference=True
        )
        return model, state, opt_state, loss, pred, true, mask
