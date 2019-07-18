#!/usr/bin/env python
"""Implement Deep Q-Network.

Glossary
--------
env : Environment
obs : Observation
rew : Reward
"""
from collections import deque
from collections import namedtuple
import gym
import torch
import torch.nn as nn


# Hyperparameters
ENV_STEPS = 10000
REPLAY_BUFFER_SIZE = 1000


Transition = namedtuple('Transition', ['obs', 'action', 'rew', 'next_obs', 'done'])

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
            nn.Linear(128, out_dim)
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


def main():
    env = gym.make('CartPole-v0')
    obs = env.reset()

    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    episode_return = 0
    episode_count = 0
    for i in range(ENV_STEPS):
        # TODO(seungjaeryanlee): Use epsilon-greedy policy
        q_values = q_net(torch.FloatTensor(obs))
        action = q_values.argmax().item()
        next_obs, rew, done, info = env.step(action)
        replay_buffer.append(Transition(obs=obs, action=action, rew=rew, next_obs=next_obs, done=done))
        episode_return += rew
        if done:
            env.reset()
            episode_return = 0
            episode_count += 1
            # TODO(seungjaeryanlee): Use logging and wandb
            print('Episode {} Return: {}'.format(episode_count, episode_return))

        obs = next_obs

if __name__ == "__main__":
    main()
