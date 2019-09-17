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

With default config, the model is saved to `saves/`.
To save in a different location, use the SAVE_DIR flag.
```
python train_eval_atari.py -c pong.conf --SAVE_DIR=saves2/
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
import gym
import torch
import torch.optim as optim

from dqn.agents import DQNAgent
from dqn.networks import AtariQNetwork
from dqn.replays import NATUREDQN_ATARI_PREPROCESS_BATCH, CircularReplayBuffer
from environments import AtariPreprocessing, FrameStack
from train_eval import get_config, train_eval
from utils import get_logger, load_models, set_env_random_seeds, set_global_random_seeds


def main():
    """Run only when this file is called directly."""
    # Setup hyperparameters
    CONFIG = get_config()

    # Log to File and Console
    logger = get_logger(log_to_console=True, log_to_file=CONFIG.LOG_TO_FILE)

    # Choose CPU and GPU
    torch.set_num_threads(CONFIG.CPU_THREADS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("GPU not available: this run could be slow.")

    # Setup environment
    # v4 variant: No repeat action
    env = gym.make(CONFIG.ENV_NAME)
    eval_env = gym.make(CONFIG.ENV_NAME)
    # AtariPreprocessing:
    # - Max NOOP on start: 30
    # - Frameskip: CONFIG.FRAME_SKIP
    #   - If Frameskip > 1, max pooling is done
    # - Screen size: 84
    # - Terminal on life loss: True
    # - Grayscale obs: True
    env = AtariPreprocessing(
        env, frame_skip=CONFIG.FRAME_SKIP, terminal_on_life_loss=True
    )
    # NOTE(seungjaeryanlee): In evaluation, episode does not end in life loss
    # https://github.com/deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/dqn/train_agent.lua#L119  # noqa: B950
    eval_env = AtariPreprocessing(
        eval_env, frame_skip=CONFIG.FRAME_SKIP, terminal_on_life_loss=False
    )
    # Stack frames to create observation
    env = FrameStack(env, stack_size=CONFIG.FRAME_STACK)
    eval_env = FrameStack(eval_env, stack_size=CONFIG.FRAME_STACK)

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        set_global_random_seeds(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
        # NOTE(seungjaeryanlee): Seed for env and eval_env are different for fair evaluation
        set_env_random_seeds(env, CONFIG.RANDOM_SEED)
        set_env_random_seeds(eval_env, CONFIG.RANDOM_SEED + 1)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    # Setup agent and replay buffer
    q_net = AtariQNetwork(CONFIG.FRAME_STACK, env.action_space.n).to(device)
    optimizer = optim.RMSprop(
        q_net.parameters(),
        lr=CONFIG.RMSPROP_LR,
        alpha=CONFIG.RMSPROP_DECAY,
        eps=CONFIG.RMSPROP_EPSILON,
        momentum=CONFIG.RMSPROP_MOMENTUM,
        weight_decay=CONFIG.RMSPROP_WEIGHT_DECAY,
        centered=CONFIG.RMSPROP_IS_CENTERED,
    )
    if CONFIG.LOAD_PATH:
        # Load parameters if possible
        load_models(CONFIG.LOAD_PATH, q_net=q_net, optimizer=optimizer)
    dqn_agent = DQNAgent(env, q_net, optimizer, device)
    replay_buffer = CircularReplayBuffer(
        env,
        maxlen=CONFIG.REPLAY_BUFFER_SIZE,
        device=device,
        preprocess_batch=NATUREDQN_ATARI_PREPROCESS_BATCH,
    )

    # Train and evaluate agent
    train_eval(dqn_agent, replay_buffer, env, eval_env, device, logger, CONFIG)


if __name__ == "__main__":
    main()
