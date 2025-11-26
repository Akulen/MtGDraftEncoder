import equinox as eqx
from abc import abstractmethod, ABC
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from data_types import Cards, Sets, Drafts

class DraftWRPredictor(eqx.Module, ABC):
    @abstractmethod
    def __call__(self,
        key: PRNGKeyArray,
        cards: Cards,
        sets: Sets,
        drafts: Drafts,
        state: eqx.nn.State,
        inference: bool=False
    ) -> Tuple[Float[Array, "45"], eqx.nn.State]:
        pass
