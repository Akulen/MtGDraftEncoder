import jax
import equinox as eqx
import optax
from typing import assert_never, Any, Callable, Literal, Optional, Tuple
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Real

from data_types import Cards, Sets, Drafts
from models import DraftWRPredictor

Loss = Callable[
    [Float[Array, "bs"], Real[Array, "bs"], Optional[Bool[Array, "bs"]]],
    Float[Array, ""]
]

def MSE_loss(
    y_pred: Float[Array, "bs 45"],
    y_true: Float[Array, "bs"],
    mask: Optional[Bool[Array, "bs 45"]]=None
) -> Float[Array, ""]:
    if mask is None:
        return ((y_pred - y_true.reshape((-1, 1))) ** 2).mean()
    return ((y_pred - y_true.reshape((-1, 1))) ** 2 * mask).sum() / mask.sum()

class Trainer:
    def __init__(
        self,
        tx: optax.GradientTransformation,
        loss: Loss | Literal['MSE']=MSE_loss,
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
            self.mesh, jax.sharding.PartitionSpec("batch")
        )
        self.compute_loss_grad = eqx.filter_value_and_grad(
            self.compute_loss, has_aux=True
        )
        self.tx = tx
        if isinstance(loss, str):
            if loss == 'MSE':
                loss = MSE_loss
            else:
                raise NotImplementedError
        if target == 'wr':
            self.loss = lambda y, drafts, mask=None: loss(
                y, drafts.win_rate, mask
            )
        elif target == 'pwr':
            self.loss = lambda y, drafts, mask=None: loss(
                y, drafts.pwr, mask
            )
        else:
            assert_never(target)

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
    ) -> Tuple[Float[Array, ""], eqx.nn.State]:
        batch_size = drafts.picks.shape[0]
        keys = jax.random.split(key, batch_size)
        outputs, state = jax.vmap(
            model,
            in_axes=(0, None, None, 0, None, None)
        )(keys, cards, sets, drafts, state, inference)
        mask = drafts.picks != 0
        loss = self.loss(outputs, drafts, mask)
        return loss, state

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
    ) -> Tuple[DraftWRPredictor, eqx.nn.State, Any, Float[Array, ""]]:
        model, state, opt_state, cards = self.shard_model(
            model, state, opt_state, cards
        )
        drafts = self.shard_data(drafts)
        (loss, state), grads = self.compute_loss_grad(
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
        return model, state, opt_state, loss

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
    ) -> Tuple[DraftWRPredictor, eqx.nn.State, Any, Float[Array, ""]]:
        model, state, opt_state, cards = self.shard_model(
            model, state, opt_state, cards
        )
        drafts = self.shard_data(drafts)
        loss, state = self.compute_loss(
            model, state, cards, sets, drafts, key, inference=True
        )
        return model, state, opt_state, loss
