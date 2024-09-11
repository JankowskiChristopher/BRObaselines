from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import InfoDict, Model, Params


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float,
           target_entropy: float) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')

    return new_temp, info

class Pessimism(nn.Module):
    initial_pessimism: float = 0.5

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        pessimism = self.param('pessimism',
                              init_fn=lambda key: jnp.full(
                                  (), 0.0))
        return pessimism + self.initial_pessimism
    
def update_pessimism(pessimism: Model, 
                     errors: float) -> Tuple[Model, InfoDict]:
    def pessimism_loss_fn(pessimism_params: Params) -> Tuple[Model, InfoDict]:
        pessimism_value = pessimism.apply({'params': pessimism_params})
        pessimism_loss = pessimism_value * errors
        return pessimism_loss, {
            'pessimism_loss': pessimism_loss,
            'pessimism_model': pessimism_value,
        }
    new_pessimism, info = pessimism.apply_gradient(pessimism_loss_fn)
    info.pop('grad_norm')
    return new_pessimism, info
