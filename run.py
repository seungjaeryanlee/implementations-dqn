#!/usr/bin/env python
"""Implement Deep Q-Network.

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

Logging
-------
1. You can view runs online via Weights & Biases (wandb):
https://app.wandb.ai/seungjaeryanlee/implementations-dqn/runs

2. You can use TensorBoard to view runs offline:
```
tensorboard --logdir=tensorboard_logs --port=2223
```

Glossary
--------
_b  : Batch
env : Environment
obs : Observation
rew : Reward
"""
import copy
import random
from collections import deque, namedtuple
from typing import Callable, Tuple

import configargparse
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import get_logger, make_reproducible

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

    def get_torch_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        done_b = torch.FloatTensor(done_b)

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

    def linear_anneal_func(x):
        assert x >= 0
        return (end_value - start_value) * min(x, end_steps) / end_steps + start_value

    return linear_anneal_func


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
    """Run only when this file is called directly."""
    # Setup hyperparameters
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    parser.add("--ENV_STEPS", dest="ENV_STEPS", type=int)
    parser.add("--REPLAY_BUFFER_SIZE", dest="REPLAY_BUFFER_SIZE", type=int)
    parser.add("--MIN_REPLAY_BUFFER_SIZE", dest="MIN_REPLAY_BUFFER_SIZE", type=int)
    parser.add("--BATCH_SIZE", dest="BATCH_SIZE", type=int)
    parser.add("--DISCOUNT", dest="DISCOUNT", type=float)
    parser.add("--EPSILON_START", dest="EPSILON_START", type=float)
    parser.add("--EPSILON_END", dest="EPSILON_END", type=float)
    parser.add("--EPSILON_DURATION", dest="EPSILON_DURATION", type=int)
    parser.add("--RANDOM_SEED", dest="RANDOM_SEED", type=int)
    parser.add("--TARGET_NET_UPDATE_RATE", dest="TARGET_NET_UPDATE_RATE", type=int)
    parser.add("--USE_TENSORBOARD", dest="USE_TENSORBOARD", action="store_true")
    parser.add("--USE_WANDB", dest="USE_WANDB", action="store_true")
    ARGS = parser.parse_args()
    if not hasattr(ARGS, "USE_TENSORBOARD"):
        ARGS.USE_TENSORBOARD = False
    if not hasattr(ARGS, "USE_WANDB"):
        ARGS.USE_WANDB = False

    print()
    print("+--------------------------------+--------------------------------+")
    print("| Hyperparameters                | Value                          |")
    print("+--------------------------------+--------------------------------+")
    for arg in vars(ARGS):
        print("| {:30} | {:<30} |".format(arg, getattr(ARGS, arg)))
    print("+--------------------------------+--------------------------------+")
    print()

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    if ARGS.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="tensorboard_logs")
    if ARGS.USE_WANDB:
        import wandb

        wandb.init(project="implementations-dqn", config=ARGS)

    # Fix random seeds
    # TODO(seungjaeryanlee): This is not working!
    make_reproducible(seed=ARGS.RANDOM_SEED, use_random=True, use_torch=True)

    # Setup environment
    env = gym.make("CartPole-v0")
    env.seed(ARGS.RANDOM_SEED)
    obs = env.reset()

    # Setup agent
    # TODO(seungjaeryanlee): Implement Save & Load
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    target_q_net = copy.deepcopy(q_net)
    replay_buffer = ReplayBuffer(maxlen=ARGS.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(q_net.parameters())
    get_epsilon = get_linear_anneal_func(
        ARGS.EPSILON_START, ARGS.EPSILON_END, ARGS.EPSILON_DURATION
    )

    if ARGS.USE_WANDB:
        wandb.watch(q_net)

    episode_return = 0
    episode_i = 0
    for step_i in range(ARGS.ENV_STEPS):
        # Select and make action
        epsilon = get_epsilon(step_i)
        action = select_action(env, obs, q_net, epsilon)
        next_obs, rew, done, info = env.step(action)

        # Update replay buffer and train QNetwork
        replay_buffer.append(Transition(obs, action, rew, next_obs, done))
        if len(replay_buffer) >= ARGS.MIN_REPLAY_BUFFER_SIZE:
            obs_b, action_b, rew_b, next_obs_b, done_b = replay_buffer.get_torch_batch(
                ARGS.BATCH_SIZE
            )
            assert obs_b.shape == (ARGS.BATCH_SIZE, env.observation_space.shape[0])
            assert action_b.shape == (ARGS.BATCH_SIZE,)
            assert rew_b.shape == (ARGS.BATCH_SIZE,)
            assert next_obs_b.shape == (ARGS.BATCH_SIZE, env.observation_space.shape[0])
            assert done_b.shape == (ARGS.BATCH_SIZE,)

            target = (
                rew_b
                + (1 - done_b) * ARGS.DISCOUNT * target_q_net(next_obs_b).max(dim=-1)[0]
            )
            prediction = q_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
            assert target.shape == (ARGS.BATCH_SIZE,)
            assert prediction.shape == (ARGS.BATCH_SIZE,)

            td_loss = F.smooth_l1_loss(prediction, target)
            assert td_loss.shape == ()

            optimizer.zero_grad()
            td_loss.backward()
            optimizer.step()

            # Log td_loss
            logger.debug(
                "Episode {:4d}  Steps {:5d}  Loss {:6.6f}".format(
                    episode_i, step_i, td_loss.item()
                )
            )
            if ARGS.USE_TENSORBOARD:
                writer.add_scalar("td_loss", td_loss.item(), step_i)
            if ARGS.USE_WANDB:
                wandb.log(
                    {"TD Loss": td_loss.item()},
                    step=step_i,
                )

        if step_i % ARGS.TARGET_NET_UPDATE_RATE == 0:
            target_q_net = copy.deepcopy(q_net)

        episode_return += rew

        # If episode is finished
        if done:
            logger.info(
                "Episode {:4d}  Steps {:5d}  Return {:4d}".format(
                    episode_i, step_i, int(episode_return)
                )
            )
            if ARGS.USE_TENSORBOARD:
                writer.add_scalar("episode_return", episode_return, episode_i)
            if ARGS.USE_WANDB:
                wandb.log(
                    {"Episode Return": episode_return, "Episode Count": episode_i},
                    step=step_i,
                )
            env.reset()
            episode_return = 0
            episode_i += 1

        # Prepare for next step
        obs = next_obs


if __name__ == "__main__":
    main()
