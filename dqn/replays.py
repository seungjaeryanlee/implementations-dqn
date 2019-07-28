"""Experience replay buffers for Deep Q-Network."""
import random
from collections import deque, namedtuple
from typing import Tuple

import torch

Transition = namedtuple("Transition", ["obs", "action", "rew", "next_obs", "done"])


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
