#!/usr/bin/env python
"""Implement Deep Q-Network.

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


Running
-------
You can train the DQN agent on Atari Space Invaders with the inluded
configuration file with the below command:
```
python train_eval_atari.py -c space_invaders.conf
```

For a reproducible run, use the RANDOM_SEED flag.
```
python train_eval_atari.py -c space_invaders.conf --RANDOM_SEED=1
```

With default config, the model is saved to `saves/`.
To save in a different location, use the SAVE_DIR flag.
```
python train_eval_atari.py -c space_invaders.conf --SAVE_DIR=saves2/
```

To load a trained agent, use the LOAD_PATH flag.
```
python train_eval_atari.py -c space_invaders.conf --LOAD_PATH=saves/space_invaders.pth
```

To enable logging to TensorBoard or W&B, use appropriate flags.
```
python train_eval_atari.py -c space_invaders.conf --USE_TENSORBOARD --USE_WANDB
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
from utils import get_logger, load_models, make_reproducible


def main():
    """Run only when this file is called directly."""
    # Setup hyperparameters
    CONFIG = get_config()

    # Log to File and Console
    logger = get_logger(log_to_console=True, log_to_file=CONFIG.LOG_TO_FILE)

    # Choose CPU or GPU
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
    eval_env = AtariPreprocessing(
        eval_env, frame_skip=CONFIG.FRAME_SKIP, terminal_on_life_loss=True
    )
    # Stack frames to create observation
    env = FrameStack(env, stack_size=CONFIG.FRAME_STACK)
    eval_env = FrameStack(eval_env, stack_size=CONFIG.FRAME_STACK)

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        # NOTE(seungjaeryanlee): Seed for env and eval_env should be different
        make_reproducible(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
        env.seed(CONFIG.RANDOM_SEED + 1)
        env.observation_space.seed(CONFIG.RANDOM_SEED + 2)
        env.action_space.seed(CONFIG.RANDOM_SEED + 3)
        eval_env.seed(CONFIG.RANDOM_SEED + 4)
        eval_env.observation_space.seed(CONFIG.RANDOM_SEED + 5)
        eval_env.action_space.seed(CONFIG.RANDOM_SEED + 6)
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
