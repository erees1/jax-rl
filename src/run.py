import json
import gym
from agents import DQNAgent, DQNFixedTarget, DDQN
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


WEIGHTS_NAME = "params.npz"


def log_msg(t, i, total_steps, ep_reward, epsilon=None, loss=None):
    msg = f"{t}: Episode {i}, Total Steps {total_steps}, Reward {ep_reward}"
    if loss is not None:
        msg += f", Loss {loss:4f}"
    if epsilon is not None:
        msg += f", Epsilon {epsilon:.4f}"
    logging.info(msg)


def run(
    env,
    agent,
    training=True,
    ep_steps=20,
    render=False,
    warm_up_eps=0,
    seed=0,
    val_every=10,
    log=True,
    logging_callback=log_msg,
    **kwargs,
):
    ep_rewards, ep_losses = [], []
    total_steps = 0
    if not training:
        desc = "Testing"
    else:
        desc = "Warmup"

    for ep in range(int(ep_steps + warm_up_eps)):
        obs = env.reset()
        ep_reward = 0
        ep_loss = []
        done = False
        t = 0

        # Episode loop
        while not done:
            if render:
                env.render()

            # Step environment and add to buffer
            obs, reward, done, info = play_one_step(env, agent, obs, training)

            # Update model if training
            if training and ep > warm_up_eps:
                desc = "Training"
                loss = agent.update(kwargs["batch_size"])
                ep_loss.append(loss)

            # Update counters:
            ep_reward += reward
            t += 1
            total_steps += 1

        ep_rewards.append(ep_reward)

        # Log appropriatley
        ep_mean_loss = jnp.array(ep_loss).mean() if ep_loss else None
        epsilon = agent.epsilon
        logging_callback(desc, ep, total_steps, ep_reward, epsilon, ep_mean_loss)

        # Validate performance periodically
        if val_every is not None and ep % val_every == 0 and training:
            val_eps = 10
            val_rewards, _, _ = run(
                env,
                agent,
                test_eps=val_eps,
                training=False,
                render=False,
                logging_callback=(lambda *x: x),
            )
            mean_val_reward = np.array(val_rewards).mean()
            log_msg(
                f"Validating over {val_eps} episodes",
                ep,
                total_steps,
                mean_val_reward,
            )

    return ep_rewards, ep_losses, agent


def play_one_step(env, agent, observation, training=False):
    action = agent.act(observation, training)
    next_observation, reward, done, info = env.step(action)
    if training:
        agent.buffer.append((observation, action, reward, next_observation, done))

    return next_observation, reward, done, info


def train(env, agent, train_eps=200, save_dir=None, **kwargs):
    rewards, losses, agent = run(
        env, agent, ep_steps=train_eps, logging_callback=log_msg, **kwargs
    )
    if save_dir is not None:
        agent.save(os.path.join(save_dir, WEIGHTS_NAME))
    return rewards, losses, agent


def test(env, agent, test_eps=100, warm_up_eps=0, **kwargs):
    # agent could specify path to weights
    if isinstance(agent, str):
        agent.load(agent)
    ep_rewards = run(
        env,
        agent,
        training=False,
        warm_up_eps=0,
        ep_steps=test_eps,
        logging_callback=log_msg,
        **kwargs,
    )[0]
    logger.info(
        f"Testing: Average reward over {test_eps} episodes {jnp.array(ep_rewards).mean():0.3f}"
    )
    return ep_rewards


def demo(env, agent, agent_spec=None, test_eps=5, save_dir=None, **kwargs):
    env_name = env.unwrapped.spec.id
    if save_dir is None:
        raise ValueError("Must specify save_dir so model can be found")

    dir_to_check = os.path.join(save_dir, env_name, agent_spec)
    fp = get_best_model(dir_to_check)
    agent = agent.load(fp)

    # Always render demo
    kwargs.pop("render")
    return run(env, agent, training=False, ep_steps=test_eps, render=True, **kwargs)


def get_best_model(out_dir):
    scores = []
    model_paths = []
    for path in Path(out_dir).glob("*/*"):
        try:
            _, rt = parse_logs(os.path.join(path, "log"))
            if rt is not None:
                scores.append(rt)
                model_paths.append(os.path.join(path, WEIGHTS_NAME))
        except (AttributeError, FileNotFoundError, NotADirectoryError):
            continue

    if len(model_paths) == 0:
        raise FileNotFoundError(f"Could not find any trained models in {out_dir}")

    best_score = np.argmax(np.array(scores))
    return model_paths[int(best_score)]


def setup_save_dir(save_dir):
    if os.path.exists(save_dir):
        # Clear save directory if not empty
        rmtree(save_dir)
    Path(save_dir).mkdir(parents=True)


def add_file_logging(save_dir):
    """ Set up file based logging """
    fh = logging.FileHandler(os.path.join(save_dir, "log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def save_args(args):
    """ Save parameters as json """
    dict_args = vars(args)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fh:
        json.dump(dict_args, fh)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--agent",
        default="dqn",
        choices=["dqn", "dqnft", "ddqn"],
        help="What algorithm to use",
    )
    parser.add_argument("--env", help="name of environment", default="CartPole-v1")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument(
        "--render", default="False", type=str, help="whether to render environment"
    )
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--train_eps", type=int, default=200)
    parser.add_argument("--test_eps", type=int, default=10)
    parser.add_argument("--epsilon_hlife", type=int, default=1000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--warm_up_eps", type=int, default=20)
    parser.add_argument(
        "--save_dir", default=None, help="directory to save model and logs"
    )

    # DQN arguments
    parser.add_argument(
        "--dqnft_update_every",
        default=100,
        help="how often to copy online parameters to target network in DQN with fixed target",
    )

    args = parser.parse_args()
    args.render = False if args.render in ["False", "false", "f"] else True

    # Create environment specified
    env = gym.make(vars(args).pop("env"))
    env.seed(args.seed)

    # Neural network spec
    observation_size = sum(env.observation_space.shape)
    if args.demo:
        layer_spec = None
    else:
        layer_spec = [observation_size] + args.n_layers * [32] + [env.action_space.n]

    # Load the agent
    agent_spec = vars(args).pop("agent")
    if agent_spec == "dqn":
        Agent = DQNAgent
    elif agent_spec == "dqnft":
        Agent = DQNFixedTarget
    elif agent_spec == "ddqn":
        Agent = DDQN
    agent = Agent(layer_spec=layer_spec, **vars(args))

    if args.demo:
        demo(env, agent, agent_spec=agent_spec, **vars(args))

    else:
        # Setup logging etc
        if args.save_dir is not None:
            setup_save_dir(args.save_dir)
            save_args(args)
            add_file_logging(args.save_dir)

        # Train and test
        rewards, losses, agent = train(env, agent, **vars(args))
        test(env, agent, **vars(args))

    env.close()


if __name__ == "__main__":
    main()
