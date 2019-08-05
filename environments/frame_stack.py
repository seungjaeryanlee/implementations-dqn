"""Frame stacking wrapper from TF-Agents.
https://github.com/tensorflow/agents/blob/master/tf_agents/environments/atari_wrappers.py
"""
# flake8: noqa

import collections

import gym
import numpy as np


class FrameStack(gym.Wrapper):
    """Stack previous N frames (must be applied to Gym env, not our envs)."""

    def __init__(self, env, stack_size=4):
        super(FrameStack, self).__init__(env)
        self._env = env
        self._frames = collections.deque(maxlen=stack_size)
        space = self._env.observation_space
        shape = (stack_size,) + space.shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=space.dtype
        )

        self.stack_size = stack_size

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._env, name)

    def _generate_observation(self):
        return np.array(self._frames)

    def reset(self):
        observation = self._env.reset()
        for _ in range(self.stack_size):
            self._frames.append(observation)
        return self._generate_observation()

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        self._frames.append(observation)
        return self._generate_observation(), reward, done, info
