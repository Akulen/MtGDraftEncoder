import jax
import jax.numpy as jnp
from typing import Optional
from jaxtyping import Array, Bool, Float

def masked_mean(
    x: Float[Array, "n"],
    mask: Optional[Bool[Array, "n"]]=None,
) -> Float[Array, "n"]:
    if mask is None:
        return jnp.mean(x)
    return jnp.sum(x * mask) / jnp.sum(mask)

def masked_var(
    x: Float[Array, "n"],
    mask: Optional[Bool[Array, "n"]]=None,
    mean: Optional[Float[Array, ""]]=None,
    ddof: int=0
) -> Float[Array, "n"]:
    if mask is None:
        return jnp.var(x, ddof=ddof)
    if mean is None:
        mean = masked_mean(x, mask)
    return jnp.sum((x - mean) ** 2 * mask) / (jnp.sum(mask) - ddof)

def layer_norm(
    x: Float[Array, "n"],
    mask: Optional[Bool[Array, "n"]]=None,
    eps: float=1e-5,
    ddof: int=0,
) -> Float[Array, "n"]:
    mean = masked_mean(x, mask)
    variance = masked_var(x, mask, mean, ddof=ddof)
    variance = jnp.maximum(0.0, variance)
    inv = jax.lax.rsqrt(variance + eps)
    if mask is None:
        out = (x - mean) * inv
    else:
        out = jnp.where(
            mask,
            (x - mean) * inv,
            x
        )
    return out
