"""Frame stacking wrapper from TF-Agents.
https://github.com/tensorflow/agents/blob/master/tf_agents/environments/atari_wrappers.py
"""
# flake8: noqa

import collections

import gym
import numpy as np


class FrameStack4(gym.Wrapper):
    """Stack previous four frames (must be applied to Gym env, not our envs)."""

    STACK_SIZE = 4

    def __init__(self, env):
        super(FrameStack4, self).__init__(env)
        self._env = env
        self._frames = collections.deque(maxlen=FrameStack4.STACK_SIZE)
        space = self._env.observation_space
        shape = (FrameStack4.STACK_SIZE,) + space.shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=space.dtype
        )

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._env, name)

    def _generate_observation(self):
        return np.array(self._frames)

    def reset(self):
        observation = self._env.reset()
        for _ in range(FrameStack4.STACK_SIZE):
            self._frames.append(observation)
        return self._generate_observation()

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        self._frames.append(observation)
        return self._generate_observation(), reward, done, info
