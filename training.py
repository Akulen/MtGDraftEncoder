import jax
import equinox as eqx
import optax
from typing import assert_never, cast, Callable, Literal, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray, Real

from data_types import Cards, Drafts

Loss = Callable[[Float[Array, "bs"], Real[Array, "bs"]], Float[Array, ""]]

def MSE_loss(
    y_pred: Float[Array, "bs d_out"],
    y_true: Float[Array, "bs d_out"],
) -> Float[Array, ""]:
    return ((y_pred - y_true) ** 2).mean()

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
            self.loss = lambda y, drafts: loss(y, drafts.win_rate)
        elif target == 'pwr':
            self.loss = lambda y, drafts: loss(y, drafts.pwr)
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
        model: eqx.Module,
        state: eqx.nn.State,
        cards: Cards,
        drafts: Drafts,
        key: PRNGKeyArray
    ) -> Tuple[Float[Array, ""], eqx.nn.State]:
        batch_size = drafts.picks.shape[0]
        keys = jax.random.split(key, batch_size)
        outputs, state = jax.vmap(
            cast(Callable, model),
            in_axes=(None, 0, None, 0)
        )(cards, drafts, state, keys)
        loss = self.loss(outputs, drafts)
        return loss, state

    @eqx.filter_jit(donate="all")
    def train_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        cards: Cards,
        drafts: Drafts,
        key: PRNGKeyArray
    ) -> Tuple[eqx.Module, eqx.nn.State, optax.OptState, Float[Array, ""]]:
        model, state, opt_state, cards = self.shard_model(
            model, state, opt_state, cards
        )
        drafts = self.shard_data(drafts)
        (loss, state), grads = self.compute_loss_grad(
            model, state, cards, drafts, key
        )

        updates, opt_state = self.tx.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        model, state, opt_state = self.shard_model(
            model, state, opt_state
        )
        return model, state, opt_state, loss

    @eqx.filter_jit(donate="all")
    def eval_step(
        self,
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        cards: Cards,
        drafts: Drafts,
        key: PRNGKeyArray
    ) -> Tuple[eqx.Module, eqx.nn.State, optax.OptState, Float[Array, ""]]:
        model, state, opt_state, cards = self.shard_model(
            model, state, opt_state, cards
        )
        drafts = self.shard_data(drafts)
        loss, state = self.compute_loss(model, state, cards, drafts, key)
        return model, state, opt_state, loss
