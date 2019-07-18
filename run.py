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
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Hyperparameters
ENV_STEPS = 10000
REPLAY_BUFFER_SIZE = 1000
MIN_REPLAY_BUFFER_SIZE = 100
BATCH_SIZE = 32
DISCOUNT = 1


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
    optimizer = optim.SGD(q_net.parameters(), lr=0.1)

    episode_return = 0
    episode_count = 0
    for i in range(ENV_STEPS):
        # TODO(seungjaeryanlee): Use epsilon-greedy policy
        # Select and make action
        q_values = q_net(torch.FloatTensor(obs))
        action = q_values.argmax().item()
        next_obs, rew, done, info = env.step(action)

        # Update replay buffer and train QNetwork
        replay_buffer.append(Transition(obs=obs, action=action, rew=rew, next_obs=next_obs, done=done))
        if len(replay_buffer) >= MIN_REPLAY_BUFFER_SIZE:
            transition_b = random.sample(replay_buffer, BATCH_SIZE)
            obs_b, action_b, rew_b, next_obs_b, done_b = zip(*transition_b)
            obs_b = torch.FloatTensor(obs_b)
            action_b = torch.LongTensor(action_b)
            rew_b = torch.FloatTensor(rew_b)
            next_obs_b = torch.FloatTensor(next_obs_b)
            assert obs_b.shape == (BATCH_SIZE, env.observation_space.shape[0])
            assert action_b.shape == (BATCH_SIZE, )
            assert rew_b.shape == (BATCH_SIZE, )
            assert next_obs_b.shape == (BATCH_SIZE, env.observation_space.shape[0])

            # TODO(seungjaeryanlee): Use Target QNetwork
            target = rew_b + DISCOUNT * q_net(next_obs_b).max(dim=-1)[0]
            prediction = q_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
            assert target.shape == (BATCH_SIZE, )
            assert prediction.shape == (BATCH_SIZE, )

            td_loss = F.smooth_l1_loss(prediction, target)
            assert td_loss.shape == ()

            optimizer.zero_grad()
            td_loss.backward()
            optimizer.step()

        # Logging
        episode_return += rew
        if done:
            # TODO(seungjaeryanlee): Use logging and wandb
            print('Episode {} Return: {}'.format(episode_count, episode_return))
            env.reset()
            episode_return = 0
            episode_count += 1

        obs = next_obs

if __name__ == "__main__":
    main()
