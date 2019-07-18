"""Implement Deep Q-Network.

Glossary
--------
obs : Observation
rew : Reward
"""

#!/usr/bin/env python
import gym
import torch
import torch.nn as nn


# Hyperparameters
ENV_STEPS = 10000


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

    episode_return = 0
    episode_count = 0
    for i in range(ENV_STEPS):
        # TODO(seungjaeryanlee): Use epsilon-greedy policy
        q_values = q_net(torch.FloatTensor(obs))
        action = q_values.argmax().item()
        obs, rew, done, info = env.step(action)
        episode_return += rew
        if done:
            env.reset()
            episode_return = 0
            episode_count += 1
            # TODO(seungjaeryanlee): Use logging and wandb
            print('Episode {} Return: {}'.format(episode_count, episode_return))


if __name__ == "__main__":
    main()
