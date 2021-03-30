# Deep reinforcment learning algorithms implemented with Jax

<img src="README.assets/logo.png" alt="image-20210329144135162"  />

Implementation of Deep Reinforcement Learning using Jax. Testing on the OpenAI gym CartPole environment.

## Algorithms

1. DQN - [Mnih V, Kavukcuoglu K, Silver D, et al. Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
2. DQN with target network - [Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
3. DDQN - [Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double Q-Learning](https://arxiv.org/abs/1509.06461)

## Usage

```bash
# Install deps
pip install -r requirements.text
```

### Training models
```bash
# Using launch script, by default set up to run multiple seeds
./launch.sh

# Using python
python3 run.py --agent dqn --train_eps 100 --n_layers 3 --seed 1 --test_eps 30 --lr 0.03 --batch_size 256 --warm_up_steps 500 --epsilon_hlife 1500 --save_dir out/CartPole-v1/dqn/example_run/1
```

### Demo
```bash
# Running demo with trained model, will search the <out> directory to find best performing model
python3 run.py --demo --save_dir out --agent dqn
```

### Results
Results are logged while running, the notebook [`notebooks/results.ipynb`](notebooks/results.ipynb) plots reward curves from training and testing. The notebook uses [`utils.py`](src/utils.py) to parse the logs. 


## Structure
* [`notebooks/results.ipynb`](notebooks/results.ipynb) - Visualization of training and test curves
* [`src/agents`](./agents) - Directory containing algorithms
* [`src/model.py`](src/model.py) - Jax code with neural network implementation, loss function and SGD
* [`src/run.py`](src/run.py) - Top level interface to train, test and demo models.
* [`src/utils.py`](src/utils.py) - Code to parse logs

