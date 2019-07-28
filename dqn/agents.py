"""Reinforcement learning agents."""
import copy
import random
from typing import List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNAgent:
    def __init__(self, env: gym.Env, q_net: nn.Module, optimizer: optim.Optimizer):
        """
        Deep Q-Networks agent.

        Parameters
        ----------
        env : gym.Env
            Environment to train the agent in.
        q_net : nn.Module
            The Q-Network that returns Q-values of all actions given an observation.
        optimizer : optim.Optimzer
            Optimizer for training the DQN network.

        """
        self._env = env
        self._q_net = q_net
        self._optimizer = optimizer

        self._target_q_net = copy.deepcopy(q_net)

    def select_action(self, obs: torch.Tensor, epsilon: float = 0) -> int:
        """Select action based on epsilon-greedy policy.

        Parameters
        ----------
        obs : torch.Tensor
            Observation from the current timestep.
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
            return self._env.action_space.sample()

        # Greedy action
        q_values = self._dqn_net(torch.FloatTensor(obs))
        return q_values.argmax().item()

    def train(
        self,
        experiences: List[Tuple[np.array, int, float, np.array, bool]],
        discount: float,
    ) -> float:
        """Train Q-network with batch experience.

        Parameters
        ----------
        experiences : list of tuple
            Experience collected in replay buffer.

        Returns
        -------
        td_loss : float
            TD loss calculated from this experience batch.

        """
        obs_b, action_b, rew_b, next_obs_b, done_b = experiences
        assert obs_b.shape == (len(experiences), self._env.observation_space.shape[0])
        assert action_b.shape == (len(experiences),)
        assert rew_b.shape == (len(experiences),)
        assert next_obs_b.shape == (
            len(experiences),
            self._env.observation_space.shape[0],
        )
        assert done_b.shape == (len(experiences),)

        target = (
            rew_b
            + (1 - done_b) * discount * self._target_q_net(next_obs_b).max(dim=-1)[0]
        )
        prediction = self._q_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
        assert target.shape == (len(experiences),)
        assert prediction.shape == (len(experiences),)

        td_loss = F.smooth_l1_loss(prediction, target)
        assert td_loss.shape == ()

        self._optimizer.zero_grad()
        td_loss.backward()
        self._optimizer.step()

        return td_loss.item()

    def update_target_q_net(self):
        """Update target Q-network."""
        self._target_q_net = copy.deepcopy(self._q_net)
