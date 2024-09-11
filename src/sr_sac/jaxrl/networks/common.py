import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax._src.typing import Array, ArrayLike

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def l2_norm(tree):
    return jnp.sqrt(sum((x**2).mean() for x in jax.tree_util.tree_leaves(tree)))

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
    """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

    Args:
        y: A pytree of numpy ndarray, vector y in the equation above.
    """
    gradient_norm = jnp.sqrt(sum(
        [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
    normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
    return normalized_gradient

def normalize_second_grad(second: jnp.ndarray, first_norm:float, second_norm: float) -> jnp.ndarray:
    """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

    Args:
        y: A pytree of numpy ndarray, vector y in the equation above.
    """
    normalized_gradient = jax.tree_map(lambda x: x * (first_norm / second_norm), second)
    return normalized_gradient

@jax.jit
def crelu(x: ArrayLike) -> Array:
    return jnp.concatenate((jnp.maximum(x, 0), jnp.maximum(-x, 0)), axis=-1)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Batch_stats = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


def record_activations(x, layer):
    return IdentityLayer(name=f'{layer.name}_act')(x)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    layer_names = []

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = '/'.join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f'{layer.name}_act')(x)
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            layer = nn.Dense(size, kernel_init=default_init())
            x = layer(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = self._record_activations(x, layer)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x
    
class MLPClassic(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    dropout_rate: Optional[float] = None
    layer_names = []
    output_nodes: int = 1
    categorical: bool = False

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = '/'.join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f'{layer.name}_act')(x)
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = self.activations(x)
            x = self._record_activations(x, layer2)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = self.activations(x)
            x = self._record_activations(x, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer3(x)
            x = self.activations(x)
            x = self._record_activations(x, layer3)
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer4(x)
            x = self.activations(x)
            x = self._record_activations(x, layer4)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x
        
class MLPClassic2(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    dropout_rate: Optional[float] = None
    layer_names = []
    output_nodes: int = 1
    categorical: bool = False

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = '/'.join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f'{layer.name}_act')(x)
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer2)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer2(x)
            x = self.activations(x)
            x = nn.LayerNorm()(x)
            x = self._record_activations(x, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer3(x)
            x = self.activations(x)
            x = nn.LayerNorm()(x)
            x = self._record_activations(x, layer3)
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer4(x)
            x = self.activations(x)
            x = nn.LayerNorm()(x)
            x = self._record_activations(x, layer4)
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x

class MLP_LN(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    dropout_rate: Optional[float] = None
    layer_names = []
    output_nodes: int = 1
    categorical: bool = False

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = '/'.join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f'{layer.name}_act')(x)
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            res = self._record_activations(res, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            res = self._record_activations(res, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer4(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            res = self._record_activations(res, layer4)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer5(res)
            res = nn.LayerNorm()(res)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x

class MLP_SN(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    dropout_rate: Optional[float] = None
    layer_names = []
    output_nodes: int = 1
    categorical: bool = False

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = '/'.join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f'{layer.name}_act')(x)
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            res = nn.LayerNorm()(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer2)(res, update_stats=training)
            res = self.activations(res)
            res = self._record_activations(res, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer3)(res, update_stats=training)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            res = nn.LayerNorm()(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer2)(res, update_stats=training)
            res = self.activations(res)
            res = self._record_activations(res, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer3)(res, update_stats=training)
            res = self._record_activations(res, layer3)
            x = res + x
            res = nn.LayerNorm()(x)
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer4)(res, update_stats=training)
            res = self.activations(res)
            res = self._record_activations(res, layer4)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer5)(res, update_stats=training)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x

class MLP_SN2(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    dropout_rate: Optional[float] = None
    layer_names = []
    output_nodes: int = 1
    categorical: bool = False

    def _record_activations(self, x, layer):
        if self.is_initializing():
            name = '/'.join(layer.scope.path)
            self.layer_names.append(name)
        return IdentityLayer(name=f'{layer.name}_act')(x)
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = self.activations(res)
            res = self._record_activations(res, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer3)(res, update_stats=training)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = self._record_activations(x, layer1)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = self.activations(res)
            res = self._record_activations(res, layer2)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer3)(res, update_stats=training)
            x = res + x
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer4(x)
            res = self.activations(res)
            res = self._record_activations(res, layer4)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = nn.SpectralNorm(layer5)(res, update_stats=training)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            if self.categorical:
                x = nn.softmax(x, axis=-1)
            return x

@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    batch_stats: Batch_stats
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)


        params = variables.pop('params')
        if 'batch_stats' in variables:
            batch_stats = variables.pop('batch_stats')
        else:
            batch_stats = None

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None


        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   batch_stats=batch_stats,
                   tx=tx,
                   opt_state=opt_state)


    def __call__(self, *args, **kwargs):
        if self.batch_stats is not None:
            return self.apply_fn.apply({'params': self.params, 'batch_stats':self.batch_stats}, mutable=['batch_stats'], *args, **kwargs)
        else: 
            return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, use_sam:bool=False, rho:float=0.01) -> Tuple[Any, 'Model']:
        # TODO refactor this function to make it more readable
        def get_sam_gradient(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            if self.batch_stats is not None:
                first_grads, info = grad_fn(model_params, self.batch_stats)
                self.replace(batch_stats=info.pop('batch_stats_updates')["batch_stats"])
            else:
                first_grads, info = grad_fn(model_params)
            dual_grads = dual_vector(first_grads)
            def update_fn(p, n):
                return p + rho * n
            noised_model = jax.tree_map(update_fn,
                                        model_params,
                                        dual_grads)
            if self.batch_stats is not None:
                second_grads, info = grad_fn(noised_model, batch_stats=self.batch_stats)
                self.replace(batch_stats=info.pop('batch_stats_updates')["batch_stats"])
            else:
                second_grads, info = grad_fn(noised_model)

            return first_grads, second_grads, info

        if use_sam:
            first_grads, grads, info = get_sam_gradient(self.params)
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            if self.batch_stats is not None:
                grads, info = grad_fn(self.params, batch_stats=self.batch_stats)
                self.replace(batch_stats=info.pop('batch_stats_updates')["batch_stats"])
            else:
                grads, info = grad_fn(self.params)

        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(SaveState(params=self.params, opt_state=self.opt_state)))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(params=self.params, opt_state=self.opt_state), contents
            )
        return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)


def split_tree(tree, key):
    tree_head = tree.unfreeze()
    tree_enc = tree_head.pop(key)
    tree_head = flax.core.FrozenDict(tree_head)
    tree_enc = flax.core.FrozenDict(tree_enc)
    return tree_enc, tree_head
