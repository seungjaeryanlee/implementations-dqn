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
    def __init__(
        self,
        env: gym.Env,
        q_net: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
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
        device : torch.device
            The device used to train the agent.

        """
        self._env = env
        self.q_net = q_net
        self.optimizer = optimizer

        self._target_q_net = copy.deepcopy(q_net)

        self._device = device

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
        q_values = self.q_net(torch.FloatTensor(obs).to(self._device))
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
        assert obs_b.shape == (len(obs_b),) + self._env.observation_space.shape
        assert action_b.shape == (len(obs_b), 1)
        assert rew_b.shape == (len(obs_b), 1)
        assert next_obs_b.shape == (len(obs_b),) + self._env.observation_space.shape
        assert done_b.shape == (len(obs_b), 1)

        target = rew_b + (1 - done_b) * discount * self._target_q_net(next_obs_b).max(
            dim=-1
        )[0].unsqueeze(1)
        prediction = self.q_net(obs_b).gather(1, action_b)
        assert target.shape == (len(obs_b), 1)
        assert prediction.shape == (len(obs_b), 1)

        # NOTE(seungjaeryanlee): F.smooth_l1_loss is NOT the equivalent. (by scale of 2)
        # TODO(seungjaeryanlee): Add more details in comments.
        td_loss = F.mse_loss(prediction, target)
        assert td_loss.shape == ()

        self.optimizer.zero_grad()
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1)
        self.optimizer.step()

        return td_loss.item()

    def update_target_q_net(self):
        """Update target Q-network."""
        self._target_q_net = copy.deepcopy(self.q_net)
