from jax import random, value_and_grad, jit, vmap
import jax.numpy as jnp
import logging

logger = logging.getLogger()


@jit
def predict(params, X):
    # per-example predictions
    activations = X
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    output = jnp.dot(final_w, activations) + final_b
    return output


def batch_func(predict_func):
    f = vmap(predict_func, in_axes=(None, 0))
    return f


def random_layer_params(m, n, key):
    w_key, b_key = random.split(key)
    return kaiming(w_key, m, n), jnp.zeros((n,))


def kaiming(key, m, n):
    return random.normal(key, (n, m)) * jnp.sqrt(2.0 / m)


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    logger.info(f"Randomly initializing a network with layers {sizes}")
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def relu(x):
    return jnp.maximum(0, x)


def mse_loss(func, params, X, Y):
    preds = func(params, X)
    lo = jnp.mean(jnp.square(preds - Y))
    if jnp.isnan(lo):
        raise ValueError(
            f"Loss went to nan, the predictions were {preds} and the target was {Y}"
        )
    return lo


def update(func, params, X, Y, step_size=0.001, grad_clip=1):
    l, grads = value_and_grad(mse_loss, argnums=1)(func, params, X, Y)
    if jnp.isnan(grads[0][0]).any():
        raise ValueError(
            f"gradient went to nan, the inputs were {X} and the target was {Y}"
        )

    grads = [
        (
            jnp.clip(dw, a_max=grad_clip, a_min=-grad_clip),
            jnp.clip(db, a_max=grad_clip, a_min=-grad_clip),
        )
        for (dw, db) in grads
    ]
    return l, jit(grad_descent, static_argnums=2)(params, grads, step_size)


def grad_descent(params, grads, step_size):
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]
