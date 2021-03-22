import gym
from agent import Agent
from jax import random
from argparse import ArgumentParser
import logging
import jax.numpy as jnp

#  logging.basicConfig(format='%(message)s', level=logging.INFO)

logger = logging.getLogger()
formatter = logging.Formatter("%(message)s")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


def run(env, key, agent, training=True, ep_steps=20, render=False, warm_up_steps=0, **kwargs):
    ep_rewards = []
    ep_losses = []

    total_steps = 0
    for i_episode in range(int(ep_steps)):
        observation = env.reset()
        ep_reward = 0
        ep_loss = []
        done = False
        t = 0

        while not done:
            if render:
                env.render()

            # Generate new key
            key = random.split(key)[1]

            # Step environment and add to buffer
            observation, reward, done, info = play_one_step(key, env, agent, observation, training)

            # Update model if training
            if training and total_steps > warm_up_steps:
                loss = agent.update(key, kwargs["batch_size"])
                ep_loss.append(loss)

            # Update counters:
            ep_reward += reward
            t += 1
            total_steps += 1

        # End of episode logging
        msg = f"Episode {i_episode}, Reward {ep_reward}"
        tmsg = msg + f", Epsilon {agent.epsilon:.4f}"
        if training and total_steps <= warm_up_steps:
            logger.info("Warmup: " + tmsg)

        elif training and total_steps > warm_up_steps:
            ep_mean_loss = jnp.array(ep_loss).mean()
            logger.info("Training: " + tmsg + f", Loss {ep_mean_loss:4f}")
            ep_rewards.append(ep_reward)
            ep_losses.append(ep_mean_loss)

        else:
            logger.info("Testing: " + msg)

    env.close()
    return ep_rewards, ep_losses, agent


def play_one_step(key, env, agent, observation, explore, training=True):
    action = agent.act(key, observation, explore)
    next_observation, reward, done, info = env.step(action)
    if training:
        agent.buffer.append((observation, action, reward / 100, next_observation, done))
    #  if done and env._elapsed_steps < env._max_episode_steps:
    #  reward = -5

    return next_observation, reward, done, info


def train(env, seed=1, n_layers=2, batch_size=32, train_eps=200, warm_up_steps=100, **kwargs):
    key = random.PRNGKey(seed)
    observation_size = sum(env.observation_space.shape)
    layer_spec = [observation_size] + n_layers * [32] + [env.action_space.n]
    agent = Agent(layer_spec, key, **kwargs)
    return run(
        env, key, agent, batch_size=batch_size, ep_steps=train_eps, warm_up_steps=100, **kwargs
    )


def test(env, agent, test_eps=100, **kwargs):
    # Key doesn't matter but provide one anyway
    key = random.PRNGKey(0)
    return run(env, key, agent, training=False, ep_steps=test_eps, **kwargs)[0]


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", help="name of environment", default="CartPole-v1")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--train_eps", type=int, default=200)
    parser.add_argument("--test_eps", type=int, default=100)
    parser.add_argument("--epsilon_hlife", type=int, default=1000)
    parser.add_argument("--lr", type=float, help='learning rate', default=1e-3)
    args = parser.parse_args()

    env = gym.make(vars(args).pop("env"))
    env.seed(args.seed)
    if args.test:
        test(env, **vars(args))
    else:
        rewards, losses, agent = train(env, **vars(args))
        test(env, agent, **vars(args))


if __name__ == "__main__":
    main()
