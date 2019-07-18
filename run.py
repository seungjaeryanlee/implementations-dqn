#!/usr/bin/env python
"""Implement Deep Q-Network.

Glossary
--------
_b  : Batch
env : Environment
obs : Observation
rew : Reward
"""
from collections import deque
from collections import namedtuple
import copy
import logging
import random
from typing import Callable
from typing import Tuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Setup logger
logger = logging.getLogger('main_logger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('run.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(filename)12s | %(levelname)8s | %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# Hyperparameters
ENV_STEPS = 10000
REPLAY_BUFFER_SIZE = 1000
MIN_REPLAY_BUFFER_SIZE = 100
BATCH_SIZE = 32
DISCOUNT = 1
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DURATION = 5000
RANDOM_SEED = 0xC0FFEE
TARGET_NETWORK_UPDATE_RATE = 32

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Transition = namedtuple("Transition", ["obs", "action", "rew", "next_obs", "done"])


class QNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize Q-Network.

        Parameters
        ----------
        in_dim : int
            Dimension of the input layer.
        out_dim : int
            Dimension of the output layer.

        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the network. Should be the observation.

        Returns
        -------
        q_values : torch.Tensor
            Output tensor of the network. Q-values of all actions.
        """
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, maxlen: int):
        """Initialize simple replay buffer.

        Parameters
        ----------
        maxlen : int
            The capacity of the replay buffer.

        """
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition: Transition):
        """Add new transition to the buffer.

        Parameters
        ----------
        transition: Transition
            The transition to add to the buffer.
        """
        assert type(transition) == Transition
        self.buffer.append(transition)

    def get_torch_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a randomly selected batch in torch.Tensor format.

        Parameters
        ----------
        batch_size : int
            The size of the output batches.

        Returns
        -------
        obs_b : torch.FloatTensor
            Batched observations.
        action_b : torch.LongTensor
            Batched actions.
        rew_b : torch.FloatTensor
            Batched rewards.
        next_obs_b : torch.FloatTensor
            Batched observations of the next step.
        done_b : torch.FloatTensor
            Batched terminal booleans.
        """
        transition_b = random.sample(self.buffer, batch_size)
        obs_b, action_b, rew_b, next_obs_b, done_b = zip(*transition_b)
        obs_b = torch.FloatTensor(obs_b)
        action_b = torch.LongTensor(action_b)
        rew_b = torch.FloatTensor(rew_b)
        next_obs_b = torch.FloatTensor(next_obs_b)

        return obs_b, action_b, rew_b, next_obs_b, done_b


def get_linear_anneal_func(
    start_value: float, end_value: float, end_steps: int
) -> Callable:
    """Create a linear annealing function.

    Parameters
    ----------
    start_value : float
        Initial value for linear annealing.
    end_value : float
        Terminal value for linear annealing.
    end_steps : int
        Number of steps to anneal value.

    Returns
    -------
    linear_anneal_func : Callable
        A function that returns annealed value given a step index.
    """
    return lambda x: (end_value - start_value) * x / end_steps + start_value


def select_action(
    env: gym.Env, obs: torch.Tensor, q_net: nn.Module, epsilon: float = 0
) -> int:
    """Select action based on epsilon-greedy policy.

    Parameters
    ----------
    env : gym.Env
        Environment to train the agent in.
    obs : torch.Tensor
        Observation from the current timestep.
    q_net : nn.Module
        An action-value network to find the greedy action.
    epsilon : float
        Probability of choosing a random action.

    Returns
    -------
    action : int
        The chosen action.
    """
    assert 0 <= epsilon <= 1

    # Random action
    if random.random() < epsilon:
        return env.action_space.sample()

    # Greedy action
    q_values = q_net(torch.FloatTensor(obs))
    return q_values.argmax().item()


def main():
    env = gym.make("CartPole-v0")
    env.seed(RANDOM_SEED)
    obs = env.reset()

    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    target_q_net = copy.deepcopy(q_net)
    replay_buffer = ReplayBuffer(maxlen=REPLAY_BUFFER_SIZE)
    optimizer = optim.SGD(q_net.parameters(), lr=0.1)
    get_epsilon = get_linear_anneal_func(EPSILON_START, EPSILON_END, EPSILON_DURATION)

    episode_return = 0
    episode_i = 0
    for step_i in range(ENV_STEPS):
        # Select and make action
        epsilon = get_epsilon(step_i)
        action = select_action(env, obs, q_net, epsilon)
        next_obs, rew, done, info = env.step(action)

        # Update replay buffer and train QNetwork
        replay_buffer.append(
            Transition(obs=obs, action=action, rew=rew, next_obs=next_obs, done=done)
        )
        if len(replay_buffer) >= MIN_REPLAY_BUFFER_SIZE:
            obs_b, action_b, rew_b, next_obs_b, done_b = replay_buffer.get_torch_batch(
                BATCH_SIZE
            )
            assert obs_b.shape == (BATCH_SIZE, env.observation_space.shape[0])
            assert action_b.shape == (BATCH_SIZE,)
            assert rew_b.shape == (BATCH_SIZE,)
            assert next_obs_b.shape == (BATCH_SIZE, env.observation_space.shape[0])

            target = rew_b + DISCOUNT * target_q_net(next_obs_b).max(dim=-1)[0]
            prediction = q_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
            assert target.shape == (BATCH_SIZE,)
            assert prediction.shape == (BATCH_SIZE,)

            td_loss = F.smooth_l1_loss(prediction, target)
            assert td_loss.shape == ()

            optimizer.zero_grad()
            td_loss.backward()
            optimizer.step()

        if step_i % TARGET_NETWORK_UPDATE_RATE == 0:
            target_q_net = copy.deepcopy(q_net)

        # Logging
        episode_return += rew
        if done:
            # TODO(seungjaeryanlee): Use logging and wandb
            logger.info("Episode {:4d} Return: {}".format(episode_i, episode_return))
            env.reset()
            episode_return = 0
            episode_i += 1

        obs = next_obs


if __name__ == "__main__":
    main()
