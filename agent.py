from collections import deque
from jax import random
import jax.numpy as jnp
from model import init_network_params, update, predict, batch_func
import logging

logger = logging.getLogger()


class Agent:
    def __init__(
        self,
        layer_spec,
        key,
        lr=0.001,
        epsilon_hlife=500,
        epsilon=1,
        epsilon_min=0.2,
        buffer_size=1000000,
        discount_factor=0.90,
        **kwargs,
    ):
        # Options
        self.lr = lr
        self.epsilon_init = epsilon
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 2 ** (-1 / epsilon_hlife)

        self.buffer = ReplayBuffer(self.buffer_size)
        self.params = init_network_params(layer_spec, key)
        self.layer_spec = layer_spec
        self.predict = lambda observations: predict(self.params, observations)
        self.batched_predict = lambda observations: batch_func(predict)(self.params, observations)
        self.steps_trained = 0

    def act(self, key, observation, explore=True):
        self.epsilon = (self.epsilon_decay ** self.steps_trained) * (
            self.epsilon_init - self.epsilon_min
        ) + self.epsilon_min
        if explore and random.uniform(key) < self.epsilon:
            action = random.randint(key, (), 0, self.layer_spec[-1])
        else:
            Q = predict(self.params, observation)
            a = jnp.argmax(Q)
            action = a

        return int(action)

    def update(self, key, batch_size):
        self.steps_trained += 1
        (observations, actions, rewards, next_observations, dones) = self.buffer.sample_batch(
            key, batch_size
        )

        # Target Q values
        #  print('observations', observations)
        next_Q_values = self.batched_predict(next_observations)
        #  print('next Q values', next_Q_values)
        max_next_Q_values = jnp.max(next_Q_values, axis=-1)
        target_Q_values = rewards + (1 - dones) * self.discount_factor * max_next_Q_values
        #  print('target_Q_values', target_Q_values)

        # Function to calcucalte Q value predictions
        def get_Q_for_actions(params, observations):
            pred_Q_values = batch_func(predict)(params, observations)
            idx = jnp.expand_dims(actions, -1)
            pred_Q_values = jnp.take_along_axis(pred_Q_values, idx, -1).squeeze()
            return pred_Q_values

        # Caclulate loss and perform gradient descent
        loss, self.params = update(
            get_Q_for_actions, self.params, observations, target_Q_values, self.lr
        )
        return loss

    def _pack_params(self):
        p = []
        for w, b in self.params:
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

    def save(self, fp):
        jnp.savez(fp, *self._pack_params())

    @classmethod
    def load(cls, fp):
        z = jnp.load(fp).values()
        params, layer_spec = cls._unpack_params(z)
        instance = cls(layer_spec, random.PRNGKey(0))
        instance.params = params
        logger.info(f"Successfully loaded model from {fp}")
        return instance


class ReplayBuffer:
    def __init__(self, maxlen):
        self.max_len = maxlen
        self.buf = deque(maxlen=maxlen)

    def sample_batch(self, key, batch_size):
        idxs = random.randint(key, (batch_size,), 0, len(self))
        batch = [self[idx] for idx in idxs]
        # Each item to be its own tensor of len batch_size
        b = list(zip(*batch))
        #  buf = jnp.array(self.buf)
        buf_mean = 0
        buf_std = 1
        return [(jnp.asarray(t) - buf_mean) / buf_std for t in b]

    def append(self, x):
        self.buf.append(x)

    def __getitem__(self, idx):
        return self.buf[idx]

    def __len__(self):
        return len(self.buf)
