#!/usr/bin/env python
import gym


# Hyperparameters
ENV_STEPS = 10000


def main():
    env = gym.make('CartPole-v0')
    env.reset()

    episode_return = 0
    for i in range(ENV_STEPS):
    # TODO(seungjaeryanlee): Use Q-network to choose action
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    episode_return += reward
    if done:
        env.reset()
        episode_return = 0
        # TODO(seungjaeryanlee): Use logging and wandb
        print('Episode Return: ', episode_return)


if __name__ == "__main__":
    main()
