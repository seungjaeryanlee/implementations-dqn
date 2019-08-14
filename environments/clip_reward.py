"""Reward clipping wrapper from OpenAI Gym."""
# flake8: noqa
import gym
import numpy as np


class ClipReward(gym.RewardWrapper):
    r""""Clip reward to [min, max]. """

    def __init__(self, env, min_r, max_r):
        super(ClipReward, self).__init__(env)
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):

        return np.clip(reward, self.min_r, self.max_r)
