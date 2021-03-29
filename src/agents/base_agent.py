from itertools import chain
from abc import abstractmethod
from collections import deque
from jax import random
import jax.numpy as jnp
import model as m
import logging
import os

logger = logging.getLogger()


class BaseAgent:
    def __init__(
        self,
        layer_spec,
        lr=0.001,
        epsilon_hlife=500,
        epsilon=1,
        epsilon_min=0.2,
        buffer_size=1000000,
        discount_factor=0.90,
        seed=0,
        **kwargs,
    ):
        # Options
        self.lr = lr
        self.epsilon_init = epsilon
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 2 ** (-1 / epsilon_hlife)

        # Setup key for initilisation
        self.key = random.PRNGKey(seed)

        self.buffer = ReplayBuffer(self.buffer_size)
        self.params = m.init_network_params(layer_spec, self.key)
        self.layer_spec = layer_spec
        self.steps_trained = 0

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def _pack_params(self, k):
        p = []
        for w, b in getattr(self, k):
            p.append(w)
            p.append(b)
        return p

    @classmethod
    def _unpack_params(cls, p):
        layer_spec = [list(p)[0].shape[1]]
        params = []
        for w, b in zip(*[iter(p)] * 2):
            params.append((jnp.array(w), jnp.array(b)))
            layer_spec.append(len(b))
        return params, layer_spec

    def save(self, fp_dir):
        # Each attribute that we need to save goes in this dictionary
        for k, v in self.params_to_save.items():
            jnp.savez(os.path.join(fp_dir, k), *self._pack_params(k))

    @classmethod
    def load(cls, fp_dir):
        fp = os.path.join(fp, "params.npz")
        z = jnp.load(fp).values()
        params, layer_spec = cls._unpack_params(z)
        instance = cls(layer_spec, random.PRNGKey(0))
        instance.params = params
        logger.info(f"Successfully loaded model from {fp}")
        return instance


class ParamRegister:
    def __init__(self):
        self.register = []

    def register_param(self, name, value):
        self.register.append(name)
        setattr(self, name, value)

    def _flatten_params(self):
        p = {}
        for k in self.register:
            v = getattr(self, k)
            for i, e in enumerate(chain(*v)):
                p[k + f".{i}"] = e
        return p

    @classmethod
    def _unflatten_weights(cls, p):
        params = list(zip(p[::2], p[1::2]))
        return params

    @classmethod
    def _read_npz(cls, it):
        tmp = {}
        for k, v in it:
            k1, k2 = k.split(".")
            if k1 in tmp:
                tmp[k1].append(jnp.array(v))
            else:
                tmp[k1] = [jnp.array(v)]
        return tmp

    def save(self, fp_dir):
        # Each attribute that we need to save goes in this dictionary
        p = self._flatten_params()
        jnp.savez(os.path.join(fp_dir, "params.npz"), **p)

    @classmethod
    def load(cls, fp_dir):
        fp = os.path.join(fp_dir, "params.npz")
        z = jnp.load(fp).items()
        var = cls._read_npz(z)
        instance = cls()
        for k, v in var.items():
            params = cls._unflatten_weights(v)
            setattr(instance, k, params)
        logger.info(f"Successfully loaded parameters from {fp}")
        return instance


class ReplayBuffer:
    def __init__(self, maxlen, seed=0):
        self.max_len = maxlen
        self.buf = deque(maxlen=maxlen)
        self.key = random.PRNGKey(seed)

    def sample_batch(self, batch_size):
        self.key = random.split(self.key)[0]
        idxs = random.randint(self.key, (batch_size,), 0, len(self))
        batch = [self[idx] for idx in idxs]

        # Each item to be its own tensor of len batch_size
        b = list(zip(*batch))
        buf_mean = 0
        buf_std = 1
        return [(jnp.asarray(t) - buf_mean) / buf_std for t in b]

    def append(self, x):
        self.buf.append(x)

    def __getitem__(self, idx):
        return self.buf[idx]

    def __len__(self):
        return len(self.buf)
