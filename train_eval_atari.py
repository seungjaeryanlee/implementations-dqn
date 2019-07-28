#!/usr/bin/env python
"""Implement Deep Q-Network.

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


Running
-------
You can train the DQN agent on Atari Pong with the inluded
configuration file with the below command:
```
python train_eval_atari.py -c pong.conf
```

For a reproducible run, use the RANDOM_SEED flag.
```
python train_eval_atari.py -c pong.conf --RANDOM_SEED=1
```

To save a trained agent, use the SAVE_PATH flag.
```
python train_eval_atari.py -c pong.conf --SAVE_PATH=saves/pong.pth
```

To load a trained agent, use the LOAD_PATH flag.
```
python train_eval_atari.py -c pong.conf --LOAD_PATH=saves/pong.pth
```

To enable logging to TensorBoard or W&B, use appropriate flags.
```
python train_eval_atari.py -c pong.conf --USE_TENSORBOARD --USE_WANDB
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
import os

import configargparse
import gym
import numpy as np
import torch
import torch.optim as optim

from dqn.agents import DQNAgent
from dqn.networks import AtariQNetwork
from dqn.replays import ReplayBuffer, Transition
from environments import AtariPreprocessing, FrameStack4
from utils import get_linear_anneal_func, get_logger, make_reproducible


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
    # Deterministic-v4 variant: 4(or 3*) frameskip, No repeat action
    env = gym.make("PongDeterministic-v4")
    eval_env = gym.make("PongDeterministic-v4")
    # AtariPreprocessing:
    # - Max NOOP on start: 30
    # - Frameskip: 1
    # - Screen size: 84
    # - Terminal on life loss: True
    # - Grayscale obs: True
    env = AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=True)
    eval_env = AtariPreprocessing(eval_env, frame_skip=1, terminal_on_life_loss=True)
    # Stack four frames
    env = FrameStack4(env)
    eval_env = FrameStack4(eval_env)
    # TODO(seungajaeryanlee): Does "action repeat" and "update frequency" parameter
    #                         in paper mean separate things? Should I wait for
    #                         16 actions before each update?

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        make_reproducible(seed=CONFIG.RANDOM_SEED, use_random=True, use_torch=True)
        env.seed(CONFIG.RANDOM_SEED)
        eval_env.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    obs = env.reset()

    # Setup agent
    # TODO(seungjaeryanlee): = Number of Frames stacked
    # TODO(seungjaeryanlee): env.action_space.n returns 20?
    q_net = AtariQNetwork(4, 6)
    replay_buffer = ReplayBuffer(maxlen=CONFIG.REPLAY_BUFFER_SIZE)
    # TODO(seungjaeryanlee): Check parameters
    optimizer = optim.RMSprop(
        q_net.parameters(),
        lr=0.00025,
        alpha=0.95,
        eps=0.01,
        weight_decay=0,
        momentum=0,
        centered=True,
    )
    get_epsilon = get_linear_anneal_func(
        CONFIG.EPSILON_START, CONFIG.EPSILON_END, CONFIG.EPSILON_DURATION
    )
    dqn_agent = DQNAgent(env, q_net, optimizer)

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
        # Interact with the environment and save the experience
        epsilon = get_epsilon(step_i)
        action = dqn_agent.select_action(np.expand_dims(obs, 0), epsilon)
        next_obs, rew, done, info = env.step(action)
        replay_buffer.append(Transition(obs, action, rew, next_obs, done))

        # Train QNetwork
        # TODO(seungjaeryanlee): Only update after every four actions
        if len(replay_buffer) >= CONFIG.MIN_REPLAY_BUFFER_SIZE:
            experiences = replay_buffer.get_torch_batch(CONFIG.BATCH_SIZE)
            td_loss = dqn_agent.train(experiences, discount=CONFIG.DISCOUNT)

            # Log td_loss
            # TODO(seungjaeryanlee): Don't log too often: add parameter
            # TODO(seungjaeryanlee): Add option of disabling file logger
            logger.debug(
                "Episode {:4d}  Steps {:5d}  Loss {:6.6f}".format(
                    episode_i, step_i, td_loss
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("td_loss", td_loss, step_i)
            if CONFIG.USE_WANDB:
                wandb.log({"TD Loss": td_loss}, step=step_i)

        # Evaluate agent periodically
        if step_i % CONFIG.EVAL_FREQUENCY == 0:
            eval_done = False
            eval_obs = eval_env.reset()
            eval_episode_return = 0
            while not eval_done:
                eval_action = dqn_agent.select_action(
                    np.expand_dims(eval_obs, 0), epsilon=0
                )
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
            dqn_agent.update_target_q_net()

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