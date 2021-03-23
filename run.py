import gym
from agent import Agent
from jax import random
from argparse import ArgumentParser
import logging
import jax.numpy as jnp
import numpy as np
import os
from shutil import rmtree
from pathlib import Path
from utils import parse_logs

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
        msg = f"Episode {i_episode}, Total Steps {total_steps}, Reward {ep_reward}"
        ep_rewards.append(ep_reward)
        tmsg = msg + f", Epsilon {agent.epsilon:.4f}"
        if training and total_steps <= warm_up_steps:
            logger.info("Warmup: " + tmsg)

        elif training and total_steps > warm_up_steps:
            ep_mean_loss = jnp.array(ep_loss).mean()
            logger.info("Training: " + tmsg + f", Loss {ep_mean_loss:4f}")
            ep_losses.append(ep_mean_loss)

        else:
            logger.info("Testing: " + msg)

    env.close()
    if not training:
        logger.info(
            f"Testing: Average reward over {i_episode + 1} episodes {jnp.array(ep_rewards).mean():0.3f}"
        )
    return ep_rewards, ep_losses, agent


def play_one_step(key, env, agent, observation, training=False):
    action = agent.act(key, observation, training)
    next_observation, reward, done, info = env.step(action)
    if training:
        agent.buffer.append((observation, action, reward, next_observation, done))

    return next_observation, reward, done, info


def train(
    env,
    seed=1,
    n_layers=2,
    batch_size=32,
    train_eps=200,
    warm_up_steps=100,
    save_dir=None,
    **kwargs,
):
    key = random.PRNGKey(seed)
    observation_size = sum(env.observation_space.shape)
    layer_spec = [observation_size] + n_layers * [32] + [env.action_space.n]
    agent = Agent(layer_spec, key, **kwargs)
    rewards, losses, agent = run(
        env,
        key,
        agent,
        batch_size=batch_size,
        ep_steps=train_eps,
        warm_up_steps=warm_up_steps,
        **kwargs,
    )
    if save_dir is not None:
        agent.save(os.path.join(save_dir, "params.npz"))
    return rewards, losses, agent


def test(env, agent, test_eps=100, **kwargs):
    # Key doesn't matter but provide one anyway
    key = random.PRNGKey(0)
    return run(env, key, agent, training=False, ep_steps=test_eps, **kwargs)[0]


def demo(env, test_eps=20, save_dir=None, **kwargs):
    env_name = env.unwrapped.spec.id
    if save_dir is None:
        raise ValueError("Must specify save_dir so model can be found")

    fp = get_best_model(env_name, save_dir)
    agent = Agent.load(fp)
    key = random.PRNGKey(0)

    # Always render demo
    kwargs.pop("render")
    run(env, key, agent, training=False, ep_steps=test_eps, render=True, **kwargs)


def get_best_model(env_name, out_dir):
    scores = []
    model_paths = []
    for path in Path(os.path.join(out_dir, env_name)).glob("*/*"):
        try:
            _, rt = parse_logs(os.path.join(path, "log"))
            if rt is not None:
                scores.append(rt)
                model_paths.append(os.path.join(path, "params.npz"))
        except (AttributeError, FileNotFoundError):
            continue

    if len(model_paths) == 0:
        raise FileNotFoundError("Could not find any trained models")

    best_score = np.argmax(np.array(scores))
    return model_paths[int(best_score)]


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", help="name of environment", default="CartPole-v1")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--train_eps", type=int, default=200)
    parser.add_argument("--test_eps", type=int, default=10)
    parser.add_argument("--epsilon_hlife", type=int, default=1000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--warm_up_steps", type=int, default=500)
    parser.add_argument("--save_dir", default=None, help="directory to save model and logs")
    args = parser.parse_args()

    env = gym.make(vars(args).pop("env"))
    env.seed(args.seed)

    if args.demo:
        demo(env, **vars(args))

    else:
        #  Set up save directory
        if args.save_dir is not None:
            if os.path.exists(args.save_dir):
                # Clear save directory if not empty
                rmtree(args.save_dir)
        Path(args.save_dir).mkdir(parents=True)

        # Set up file based logging
        fh = logging.FileHandler(os.path.join(args.save_dir, "log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        rewards, losses, agent = train(env, **vars(args))
        test(env, agent, **vars(args))


if __name__ == "__main__":
    main()
