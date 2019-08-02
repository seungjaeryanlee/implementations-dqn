"""Experience replay buffers for Deep Q-Network."""
import copy
import random
from collections import deque, namedtuple
from typing import Tuple

import gym
import numpy as np
import torch

Transition = namedtuple("Transition", ["obs", "action", "rew", "next_obs", "done"])


class ReplayBuffer:
    def __init__(self, maxlen: int, device: torch.device):
        """Initialize simple replay buffer.

        Parameters
        ----------
        maxlen : int
            The capacity of the replay buffer.
        device : torch.device
            Device that the agent will use to train with this replay buffer.

        """
        self.buffer = deque(maxlen=maxlen)
        self._device = device

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

    def get_numpy_batch(
        self, batch_size: int
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """Return a randomly selected batch in numpy.array format.

        Parameters
        ----------
        batch_size : int
            The size of the output batches.

        Returns
        -------
        obs_b : np.array
            Batched observations.
        action_b : np.array
            Batched actions.
        rew_b : np.array
            Batched rewards.
        next_obs_b : np.array
            Batched observations of the next step.
        done_b : np.array
            Batched terminal booleans.

        """
        transition_b = random.sample(self.buffer, batch_size)
        obs_b, action_b, rew_b, next_obs_b, done_b = zip(*transition_b)

        return obs_b, action_b, rew_b, next_obs_b, done_b

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
        obs_b = torch.FloatTensor(obs_b).to(self._device)
        action_b = torch.LongTensor(action_b).to(self._device)
        rew_b = torch.FloatTensor(rew_b).to(self._device)
        next_obs_b = torch.FloatTensor(next_obs_b).to(self._device)
        done_b = torch.FloatTensor(done_b).to(self._device)

        return obs_b, action_b, rew_b, next_obs_b, done_b


class CircularReplayBuffer:
    def __init__(self, env: gym.Env, maxlen: int, device: torch.device):
        """Initialize circular replay buffer.

        Parameters
        ----------
        maxlen : int
            The capacity of the replay buffer.
        device : torch.device
            Device that the agent will use to train with this replay buffer.

        """
        assert maxlen > 0

        # Fill buffer with random transitions to make sure enough memory exists
        stub_transition = Transition(
            env.observation_space.sample(),
            env.action_space.sample(),
            0,
            env.observation_space.sample(),
            False,
        )
        self.buffer = [None] * maxlen
        for i in range(maxlen):
            self.buffer[i] = copy.deepcopy(stub_transition)
        assert not (self.buffer[0] is self.buffer[1])
        self.buffer = np.array(self.buffer)

        self.maxlen = maxlen
        self.curlen = 0
        self.index = 0

        self._device = device

    def __len__(self):
        return self.curlen

    def append(self, transition: Transition):
        """Add new transition to the buffer.

        Parameters
        ----------
        transition: Transition
            The transition to add to the buffer.

        """
        assert type(transition) == Transition
        self.buffer[self.index] = transition
        if self.curlen <= self.maxlen:
            self.curlen += 1
        self.index = (self.index + 1) % self.maxlen

    def get_numpy_batch(
        self, batch_size: int
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """Return a randomly selected batch in numpy.array format.

        Parameters
        ----------
        batch_size : int
            The size of the output batches.

        Returns
        -------
        obs_b : np.array
            Batched observations.
        action_b : np.array
            Batched actions.
        rew_b : np.array
            Batched rewards.
        next_obs_b : np.array
            Batched observations of the next step.
        done_b : np.array
            Batched terminal booleans.

        """
        indices = np.random.randint(low=0, high=self.curlen, size=batch_size)
        transition_b = self.buffer[indices, ...]
        assert transition_b.shape == (batch_size, 5)

        obs_b, action_b, rew_b, next_obs_b, done_b = transition_b.T

        return obs_b, action_b, rew_b, next_obs_b, done_b

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
        obs_b, action_b, rew_b, next_obs_b, done_b = self.get_numpy_batch(batch_size)
        obs_b = torch.FloatTensor(obs_b).to(self._device)
        action_b = torch.LongTensor(action_b).to(self._device)
        rew_b = torch.FloatTensor(rew_b).to(self._device)
        next_obs_b = torch.FloatTensor(next_obs_b).to(self._device)
        done_b = torch.FloatTensor(done_b).to(self._device)

        return obs_b, action_b, rew_b, next_obs_b, done_b
