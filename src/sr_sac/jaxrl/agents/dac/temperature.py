from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import InfoDict, Model, Params

class TemperatureOffset(nn.Module):
    init_value: float = 1.0
    offset: float = 0.0
    log_temp_min: float = -10.0
    log_temp_max: float = 7.5
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.init_value)))
        
        log_temp = self.log_temp_min + (self.log_temp_max - self.log_temp_min) * 0.5 * (1 + nn.tanh(log_temp))
        return jnp.exp(log_temp) - self.offset 

def update_optimism(
        optimism: Model, empirical_kl: float, target_kl: float, beta_lb: float = 1.0
        ) -> Tuple[Model, InfoDict]:
    def optimism_loss_fn(optimism_params):
        beta_ub = optimism.apply({'params': optimism_params})
        optimism_loss = (beta_ub - beta_lb) * (empirical_kl - target_kl).mean()
        return optimism_loss, {'beta_ub': beta_ub, 'optimism_loss': optimism_loss}
    new_beta, info = optimism.apply_gradient(optimism_loss_fn)
    info.pop('grad_norm')
    return new_beta, info

def update_regularizer(
        regularizer: Model, empirical_kl: float, target_kl: float
        ) -> Tuple[Model, InfoDict]:
    def regularizer_loss_fn(regularizer_params):
        kl_weight = regularizer.apply({'params': regularizer_params})
        regularizer_loss = -kl_weight * (empirical_kl - target_kl).mean()
        return regularizer_loss, {'kl_weight': kl_weight, 'regularizer_loss': regularizer_loss}
    new_regularizer, info = regularizer.apply_gradient(regularizer_loss_fn)
    info.pop('grad_norm')
    return new_regularizer, info

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
