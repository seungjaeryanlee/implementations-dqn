#!/usr/bin/env python
"""Implement Deep Q-Network.

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


Running
-------
You can train the DQN agent on CartPole with the inluded
configuration file with the below command:
```
python train_eval.py -c cartpole.conf
```

For a reproducible run, use the RANDOM_SEED flag.
```
python train_eval.py -c cartpole.conf --RANDOM_SEED=1
```

To save a trained agent, use the SAVE_PATH flag.
```
python train_eval.py -c cartpole.conf --SAVE_PATH=saves/cartpole.pth
```

To load a trained agent, use the LOAD_PATH flag.
```
python train_eval.py -c cartpole.conf --LOAD_PATH=saves/cartpole.pth
```

To enable logging to TensorBoard or W&B, use appropriate flags.
```
python train_eval.py -c cartpole.conf --USE_TENSORBOARD --USE_WANDB
```


Logging
-------
1. You can view runs online via Weights & Biases (wandb):
https://app.wandb.ai/seungjaeryanlee/implementations-dqn/runs

2. You can use TensorBoard to view runs offline:
```
tensorboard --logdir=tensorboard_logs --port=2223
```


Glossary
--------
_b  : Batch
env : Environment
obs : Observation
rew : Reward
"""
import copy
import os
import random

import configargparse
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import QNetwork
from replays import ReplayBuffer, Transition
from utils import get_linear_anneal_func, get_logger, make_reproducible


def select_action(
    env: gym.Env, obs: torch.Tensor, q_net: nn.Module, epsilon: float = 0
) -> int:
    """Select action based on epsilon-greedy policy.

    Parameters
    ----------
    env : gym.Env
        Environment to train the agent in.
    obs : torch.Tensor
        Observation from the current timestep.
    q_net : nn.Module
        An action-value network to find the greedy action.
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
        return env.action_space.sample()

    # Greedy action
    q_values = q_net(torch.FloatTensor(obs))
    return q_values.argmax().item()


def main():
    """Run only when this file is called directly."""
    # Setup hyperparameters
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    parser.add("--ENV_STEPS", dest="ENV_STEPS", type=int)
    parser.add("--REPLAY_BUFFER_SIZE", dest="REPLAY_BUFFER_SIZE", type=int)
    parser.add("--MIN_REPLAY_BUFFER_SIZE", dest="MIN_REPLAY_BUFFER_SIZE", type=int)
    parser.add("--BATCH_SIZE", dest="BATCH_SIZE", type=int)
    parser.add("--DISCOUNT", dest="DISCOUNT", type=float)
    parser.add("--EPSILON_START", dest="EPSILON_START", type=float)
    parser.add("--EPSILON_END", dest="EPSILON_END", type=float)
    parser.add("--EPSILON_DURATION", dest="EPSILON_DURATION", type=int)
    parser.add("--RANDOM_SEED", dest="RANDOM_SEED", type=int)
    parser.add("--TARGET_NET_UPDATE_RATE", dest="TARGET_NET_UPDATE_RATE", type=int)
    parser.add("--EVAL_FREQUENCY", dest="EVAL_FREQUENCY", type=int)
    parser.add("--SAVE_PATH", dest="SAVE_PATH", type=str, default="")
    parser.add("--LOAD_PATH", dest="LOAD_PATH", type=str, default="")
    parser.add("--USE_TENSORBOARD", dest="USE_TENSORBOARD", action="store_true")
    parser.add("--USE_WANDB", dest="USE_WANDB", action="store_true")
    CONFIG = parser.parse_args()
    if not hasattr(CONFIG, "USE_TENSORBOARD"):
        CONFIG.USE_TENSORBOARD = False
    if not hasattr(CONFIG, "USE_WANDB"):
        CONFIG.USE_WANDB = False

    print()
    print("+--------------------------------+--------------------------------+")
    print("| Hyperparameters                | Value                          |")
    print("+--------------------------------+--------------------------------+")
    for arg in vars(CONFIG):
        print(
            "| {:30} | {:<30} |".format(
                arg, getattr(CONFIG, arg) if getattr(CONFIG, arg) is not None else ""
            )
        )
    print("+--------------------------------+--------------------------------+")
    print()

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    if CONFIG.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="tensorboard_logs")
    if CONFIG.USE_WANDB:
        import wandb

        wandb.init(project="implementations-dqn", config=CONFIG)

    # Setup environment
    env = gym.make("CartPole-v0")
    eval_env = gym.make("CartPole-v0")

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        make_reproducible(seed=CONFIG.RANDOM_SEED, use_random=True, use_torch=True)
        env.seed(CONFIG.RANDOM_SEED)
        eval_env.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    obs = env.reset()

    # Setup agent
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    target_q_net = copy.deepcopy(q_net)
    replay_buffer = ReplayBuffer(maxlen=CONFIG.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(q_net.parameters())
    get_epsilon = get_linear_anneal_func(
        CONFIG.EPSILON_START, CONFIG.EPSILON_END, CONFIG.EPSILON_DURATION
    )

    # Load trained agent
    if CONFIG.LOAD_PATH:
        state_dict = torch.load(CONFIG.LOAD_PATH)
        q_net.load_state_dict(state_dict["q_net"])
        optimizer.load_state_dict(state_dict["optimizer"])

    if CONFIG.USE_WANDB:
        wandb.watch(q_net)

    episode_return = 0
    episode_i = 0
    eval_episode_i = 0
    for step_i in range(CONFIG.ENV_STEPS + 1):
        # Select and make action
        epsilon = get_epsilon(step_i)
        action = select_action(env, obs, q_net, epsilon)
        next_obs, rew, done, info = env.step(action)

        # Update replay buffer and train QNetwork
        replay_buffer.append(Transition(obs, action, rew, next_obs, done))
        if len(replay_buffer) >= CONFIG.MIN_REPLAY_BUFFER_SIZE:
            obs_b, action_b, rew_b, next_obs_b, done_b = replay_buffer.get_torch_batch(
                CONFIG.BATCH_SIZE
            )
            assert obs_b.shape == (CONFIG.BATCH_SIZE, env.observation_space.shape[0])
            assert action_b.shape == (CONFIG.BATCH_SIZE,)
            assert rew_b.shape == (CONFIG.BATCH_SIZE,)
            assert next_obs_b.shape == (
                CONFIG.BATCH_SIZE,
                env.observation_space.shape[0],
            )
            assert done_b.shape == (CONFIG.BATCH_SIZE,)

            target = (
                rew_b
                + (1 - done_b)
                * CONFIG.DISCOUNT
                * target_q_net(next_obs_b).max(dim=-1)[0]
            )
            prediction = q_net(obs_b).gather(1, action_b.unsqueeze(1)).squeeze(1)
            assert target.shape == (CONFIG.BATCH_SIZE,)
            assert prediction.shape == (CONFIG.BATCH_SIZE,)

            td_loss = F.smooth_l1_loss(prediction, target)
            assert td_loss.shape == ()

            optimizer.zero_grad()
            td_loss.backward()
            optimizer.step()

            # Log td_loss
            logger.debug(
                "Episode {:4d}  Steps {:5d}  Loss {:6.6f}".format(
                    episode_i, step_i, td_loss.item()
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("td_loss", td_loss.item(), step_i)
            if CONFIG.USE_WANDB:
                wandb.log({"TD Loss": td_loss.item()}, step=step_i)

        # Evaluate agent periodically
        if step_i % CONFIG.EVAL_FREQUENCY == 0:
            eval_done = False
            eval_obs = eval_env.reset()
            eval_episode_return = 0
            while not eval_done:
                eval_action = select_action(env, eval_obs, q_net, epsilon=0)
                eval_obs, eval_rew, eval_done, info = eval_env.step(eval_action)
                eval_episode_return += eval_rew

            logger.info(
                "EVALUATION    Steps {:5d}  Return {:4d}".format(
                    step_i, int(eval_episode_return)
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar(
                    "eval/episode_return", eval_episode_return, eval_episode_i
                )
            if CONFIG.USE_WANDB:
                wandb.log(
                    {
                        "Evaluation Episode Return": eval_episode_return,
                        "Evaluation Episode Count": eval_episode_i,
                    },
                    step=step_i,
                )

            eval_episode_i += 1

        if step_i % CONFIG.TARGET_NET_UPDATE_RATE == 0:
            target_q_net = copy.deepcopy(q_net)

        episode_return += rew

        # If episode is finished
        if done:
            logger.info(
                "Episode {:4d}  Steps {:5d}  Return {:4d}".format(
                    episode_i, step_i, int(episode_return)
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("episode_return", episode_return, episode_i)
            if CONFIG.USE_WANDB:
                wandb.log(
                    {"Episode Return": episode_return, "Episode Count": episode_i},
                    step=step_i,
                )
            env.reset()
            episode_return = 0
            episode_i += 1

        # Prepare for next step
        obs = next_obs

    # Save trained agent
    if CONFIG.SAVE_PATH:
        # Create specified directory if it does not exist yet
        SAVE_DIRECTORY = "/".join(CONFIG.SAVE_PATH.split("/")[:-1])
        if not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY)

        torch.save(
            {"q_net": q_net.state_dict(), "optimizer": optimizer.state_dict()},
            CONFIG.SAVE_PATH,
        )


if __name__ == "__main__":
    main()
