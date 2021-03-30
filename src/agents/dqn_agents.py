from jax import random
import jax.numpy as jnp
from agents.base_agent import BaseAgent
import model as m


class DQNAgent(BaseAgent):
    def __init__(self, layer_spec=None, **kwargs):
        super().__init__(**kwargs)
        self.predict = lambda observations: m.predict(self.params, observations)
        self.batched_predict = lambda observations: m.batch_func(m.predict)(
            self.params, observations
        )

        if layer_spec is not None:
            self.register_param(
                "params", m.init_network_params(layer_spec, self.key)
            )
            self.layer_spec = layer_spec

    def act(self, observation, explore=True):
        self.key, subkey = random.split(self.key)

        self.epsilon = (self.epsilon_decay ** self.steps_trained) * (
            self.epsilon_init - self.epsilon_min
        ) + self.epsilon_min
        if explore and random.uniform(self.key) < self.epsilon:
            action = random.randint(subkey, (), 0, self.layer_spec[-1])
        else:
            Q = m.predict(self.params, observation)
            action = jnp.argmax(Q)

        return int(action)

    def update(self, batch_size):
        def get_Q_for_actions(params, observations):
            """Calculate Q values for action that was taken"""
            pred_Q_values = m.batch_func(m.predict)(params, observations)
            pred_Q_values = index_Q_at_action(pred_Q_values, actions)
            return pred_Q_values

        (
            obs,
            actions,
            r,
            next_obs,
            dones,
        ) = self.buffer.sample_batch(batch_size)

        max_next_Q_values = self.get_max_Q_values(next_obs)
        target_Q_values = self.get_target_Q_values(r, dones, max_next_Q_values)

        #  Caclulate loss and perform gradient descent
        loss, self.params = m.update(
            get_Q_for_actions, self.params, obs, target_Q_values, self.lr
        )

        self.steps_trained += 1
        return loss

    def get_max_Q_values(self, next_obs):
        """Calculate max Q values for next state"""
        next_Q_values = self.batched_predict(next_obs)
        max_next_Q_values = jnp.max(next_Q_values, axis=-1)
        return max_next_Q_values

    def get_target_Q_values(self, rewards, dones, max_next_Q_values):
        """Calculate target Q values based on discounted max next_Q_values"""
        target_Q_values = (
            rewards + (1 - dones) * self.discount_factor * max_next_Q_values
        )
        return target_Q_values


class DQNFixedTarget(DQNAgent):
    def __init__(self, layer_spec=None, update_every=100, **kwargs):
        super().__init__(layer_spec=layer_spec, **kwargs)
        self.update_every = update_every
        # Need to update key so target_params != params
        self.key = random.split(self.key)[0]
        if layer_spec is not None:
            self.register_param(
                "target_params", m.init_network_params(layer_spec, self.key)
            )

        # Target functions
        self.batched_predict_target = lambda observations: m.batch_func(m.predict)(
            self.target_params, observations
        )

    def get_max_Q_values(self, next_obs):
        if self.steps_trained % self.update_every == 0:
            # Jax arrays are immutable
            self.target_params = self.params
        next_Q_values = self.batched_predict_target(next_obs)
        max_next_Q_values = jnp.max(next_Q_values, axis=-1)
        return max_next_Q_values


class DDQN(DQNFixedTarget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_max_Q_values(self, next_obs):
        if self.steps_trained % self.update_every == 0:
            # Jax arrays are immutable
            self.target_params = self.params

        next_Q_values_target = self.batched_predict_target(next_obs)
        next_Q_values_online = self.batched_predict(next_obs)
        actions = jnp.argmax(next_Q_values_online, axis=-1)
        return index_Q_at_action(next_Q_values_target, actions)


def index_Q_at_action(Q_values, actions):
    # Q_values [bsz, n_actions]
    # Actions [bsz,]
    idx = jnp.expand_dims(actions, -1)
    # pred_Q_values [bsz,]
    pred_Q_values = jnp.take_along_axis(Q_values, idx, -1).squeeze()
    return pred_Q_values
