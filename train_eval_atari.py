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
import numpy as np
import torch
import torch.optim as optim

from dqn.agents import DQNAgent
from dqn.networks import AtariQNetwork
from dqn.replays import (
    NATUREDQN_ATARI_PREPROCESS_BATCH,
    CircularReplayBuffer,
    Transition,
)
from environments import AtariPreprocessing, FrameStack
from train_eval import get_config
from utils import (
    get_linear_anneal_func,
    get_logger,
    get_timestamp,
    load_models,
    make_reproducible,
    save_models,
)


def main():
    """Run only when this file is called directly."""
    # Setup hyperparameters
    CONFIG = get_config()

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger(log_to_console=True, log_to_file=CONFIG.LOG_TO_FILE)

    if CONFIG.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="tensorboard_logs")
    if CONFIG.USE_WANDB:
        import wandb

        wandb.init(project="implementations-dqn", config=CONFIG)

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
        make_reproducible(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
        env.seed(CONFIG.RANDOM_SEED)
        eval_env.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    obs = env.reset()

    # Choose CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("GPU not available: this run could be slow.")

    # Setup agent
    q_net = AtariQNetwork(CONFIG.FRAME_STACK, env.action_space.n).to(device)
    replay_buffer = CircularReplayBuffer(
        env,
        maxlen=CONFIG.REPLAY_BUFFER_SIZE,
        device=device,
        preprocess_batch=NATUREDQN_ATARI_PREPROCESS_BATCH,
    )
    optimizer = optim.RMSprop(
        q_net.parameters(),
        lr=CONFIG.RMSPROP_LR,
        alpha=CONFIG.RMSPROP_DECAY,
        eps=CONFIG.RMSPROP_EPSILON,
        momentum=CONFIG.RMSPROP_MOMENTUM,
        weight_decay=CONFIG.RMSPROP_WEIGHT_DECAY,
        centered=CONFIG.RMSPROP_IS_CENTERED,
    )
    get_epsilon = get_linear_anneal_func(
        CONFIG.EPSILON_START, CONFIG.EPSILON_END, CONFIG.EPSILON_DURATION
    )
    dqn_agent = DQNAgent(env, q_net, optimizer, device)

    # Load trained agent
    if CONFIG.LOAD_PATH:
        load_models(CONFIG.LOAD_PATH, q_net=q_net, optimizer=optimizer)
    # Check if SAVE_DIR is defined
    if not CONFIG.SAVE_DIR:
        logger.warning("No save path specified: the model will be lost!")
    else:
        # Episodic return of saved best model
        saved_model_eval_episode_return = -float("inf")
        # Unique save directory to prevent rewrite
        unique_save_dir = f"{CONFIG.SAVE_DIR}/{CONFIG.ENV_NAME}/{get_timestamp()}/"

    if CONFIG.USE_WANDB:
        wandb.watch(q_net)

    episode_return = 0
    episode_i = 0
    eval_i = 0
    for step_i in range(CONFIG.ENV_STEPS + 1):
        # Interact with the environment and save the experience
        epsilon = get_epsilon(step_i)
        action = dqn_agent.select_action(np.expand_dims(obs, 0), epsilon)
        next_obs, rew, done, info = env.step(action)
        replay_buffer.append(Transition(obs, action, rew, next_obs, done))

        # Train QNetwork
        if (
            step_i % CONFIG.UPDATE_FREQUENCY == 0
            and len(replay_buffer) >= CONFIG.MIN_REPLAY_BUFFER_SIZE
        ):
            experiences = replay_buffer.get_torch_batch(CONFIG.BATCH_SIZE)
            td_loss = dqn_agent.train(experiences, discount=CONFIG.DISCOUNT)

            # Log td_loss and epsilon
            if step_i % CONFIG.LOG_FREQUENCY == 0:
                logger.debug(
                    "Episode {:4d}  Steps {:5d}  Epsilon {:6.6f}  Loss {:6.6f}".format(
                        episode_i, step_i, epsilon, td_loss
                    )
                )
                if CONFIG.USE_TENSORBOARD:
                    writer.add_scalar("td_loss", td_loss, step_i)
                    writer.add_scalar("epsilon", epsilon, step_i)
                if CONFIG.USE_WANDB:
                    wandb.log({"td_loss": td_loss, "epsilon": epsilon}, step=step_i)

        # Update target QNetwork
        if step_i % CONFIG.TARGET_NET_UPDATE_FREQUENCY == 0:
            dqn_agent.update_target_q_net()

        # Prepare for next step
        episode_return += rew
        obs = next_obs

        # Prepare for next episode if episode is finished
        if done:
            logger.info(
                "Episode {:4d}  Steps {:5d}  Return {:4d}".format(
                    episode_i, step_i, int(episode_return)
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("train/episode_return", episode_return, episode_i)
            if CONFIG.USE_WANDB:
                wandb.log(
                    {
                        "train/episode_return": episode_return,
                        "train/episode_count": episode_i,
                    },
                    step=step_i,
                )
            env.reset()
            episode_return = 0
            episode_i += 1

        # Evaluate agent periodically
        if step_i % CONFIG.EVAL_FREQUENCY == 0:
            all_eval_episode_return = []

            # Run multiple evaluation episodes
            for _ in range(CONFIG.EVAL_EPISODES):
                # Run evaluation
                eval_done = False
                eval_obs = eval_env.reset()
                eval_episode_return = 0
                while not eval_done:
                    eval_action = dqn_agent.select_action(
                        np.expand_dims(eval_obs, 0), epsilon=0
                    )
                    eval_obs, eval_rew, eval_done, info = eval_env.step(eval_action)
                    eval_episode_return += eval_rew
                all_eval_episode_return.append(eval_episode_return)
            avg_eval_episode_return = (
                sum(all_eval_episode_return) / CONFIG.EVAL_EPISODES
            )

            # Log results
            logger.info(
                "EVALUATION    Steps {:5d}  Return {:7.2f}".format(
                    step_i, avg_eval_episode_return
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar(
                    "eval/avg_episode_return", avg_eval_episode_return, eval_i
                )
                writer.add_histogram(
                    "eval/episode_returns", np.array(all_eval_episode_return)
                )
            if CONFIG.USE_WANDB:
                wandb.log(
                    {
                        "eval/avg_episode_return": avg_eval_episode_return,
                        "eval/episode_returns": wandb.Histogram(
                            all_eval_episode_return
                        ),
                        "eval/episode_count": eval_i,
                    },
                    step=step_i,
                )

            # Update save file if necessary
            if (
                CONFIG.SAVE_DIR
                and saved_model_eval_episode_return <= eval_episode_return
            ):
                saved_model_eval_episode_return = eval_episode_return
                save_models(
                    unique_save_dir, filename="best", q_net=q_net, optimizer=optimizer
                )
                logger.info(f"Model succesfully saved at {unique_save_dir}")

            eval_i += 1

    # Save trained agent
    if CONFIG.SAVE_DIR:
        save_models(unique_save_dir, filename="last", q_net=q_net, optimizer=optimizer)
        logger.info(f"Model successfully saved at {unique_save_dir}")


if __name__ == "__main__":
    main()
