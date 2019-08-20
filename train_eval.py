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

To save a trained agent, use the SAVE_DIR flag.
```
python train_eval.py -c cartpole.conf --SAVE_DIR=saves/
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
import configargparse
import gym
import numpy as np
import torch
import torch.optim as optim

from dqn.agents import DQNAgent
from dqn.networks import QNetwork
from dqn.replays import CircularReplayBuffer, Transition
from utils import (
    get_linear_anneal_func,
    get_logger,
    get_timestamp,
    load_models,
    save_models,
    set_env_random_seeds,
    set_global_random_seeds,
)


def get_config():
    """Parse configuration from config file and arguments."""
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    parser.add(
        "--ENV_NAME",
        type=str,
        help="Full name of the environment, including the mode and version number.",
    )
    parser.add(
        "--ENV_STEPS",
        type=int,
        help="Number of environment steps to train the agent on.",
    )
    parser.add(
        "--FRAME_STACK",
        type=int,
        help=(
            "The number of most recent frames experienced by the agent"
            "that are given as input to the Q network."
        ),
    )
    parser.add(
        "--FRAME_SKIP",
        type=int,
        help=(
            "Repeat each action selected by the agent this many times. ",
            "Using a value of 4 results in the agent seeing only every 4th input frame.",
        ),
    )
    parser.add(
        "--REPLAY_BUFFER_SIZE",
        type=int,
        help="The capacity of the experience replay buffer.",
    )
    parser.add(
        "--MIN_REPLAY_BUFFER_SIZE",
        type=int,
        help="Minimum size of the replay buffer before starting training.",
    )
    parser.add(
        "--BATCH_SIZE",
        type=int,
        help="The size of the batch to train the agent on each step.",
    )
    parser.add(
        "--DISCOUNT",
        type=float,
        help="Discount factor that scales future rewards. Denoted with gamma",
    )
    parser.add(
        "--EPSILON_START",
        type=float,
        help="Starting value of a linearly annealing epsilon.",
    )
    parser.add(
        "--EPSILON_END",
        type=float,
        help="Terminal value of a linearly annealing epsilon.",
    )
    parser.add(
        "--EPSILON_DURATION",
        type=int,
        help="The duration of linear annealing for epsilon",
    )
    parser.add(
        "--RANDOM_SEED",
        type=int,
        help="Random seed to set to guarantee reproducibility.",
    )
    parser.add(
        "--TARGET_NET_UPDATE_FREQUENCY",
        type=int,
        help="How many steps to wait for each target network update.",
    )

    parser.add("--RMSPROP_LR", type=float, help="The learning rate for RMSprop.")
    parser.add("--RMSPROP_DECAY", type=float, help="Smoothing constant for RMSprop.")
    parser.add(
        "--RMSPROP_EPSILON",
        type=float,
        help="Term added to the denominator for numerical stability.",
    )
    parser.add("--RMSPROP_MOMENTUM", type=float, help="Momentum factor for RMSprop.")
    parser.add("--RMSPROP_WEIGHT_DECAY", type=float, help="RMSprop L2 penalty.")
    parser.add(
        "--RMSPROP_IS_CENTERED",
        action="store_true",
        help="If True, compute the centered RMSprop.",
    )
    parser.add(
        "--UPDATE_FREQUENCY",
        type=int,
        help="The number of actions selected by the agent between each successive SGD updates.",
    )
    parser.add(
        "--LOG_FREQUENCY",
        type=int,
        help=(
            "How frequently (in environment steps) to log various training "
            "metrics. Not relevant to metrics that are episode-specific."
        ),
    )
    parser.add(
        "--LOG_TO_FILE",
        action="store_true",
        help="If true, log everything to a run.log file.",
    )
    parser.add(
        "--EVAL_FREQUENCY",
        type=int,
        help="How frequently (in environment steps) the agent will be evaluated.",
    )
    parser.add(
        "--EVAL_EPISODES",
        type=int,
        help="How many episodes the agent will be evaluated on.",
    )
    parser.add(
        "--EVAL_EPSILON", type=float, help="Epsilon value while running evaluation."
    )
    parser.add("--SAVE_DIR", type=str, help="Save model to given directory.")
    parser.add("--LOAD_PATH", type=str, help="Load model from given file.")
    parser.add(
        "--USE_TENSORBOARD",
        action="store_true",
        help="Use TensorBoard for offline logging.",
    )
    parser.add(
        "--USE_WANDB",
        action="store_true",
        help="Use Weights & Biases for online logging.",
    )
    CONFIG = parser.parse_args()
    if not hasattr(CONFIG, "RMSPROP_IS_CENTERED"):
        CONFIG.RMSPROP_IS_CENTERED = False
    if not hasattr(CONFIG, "LOG_TO_FILE"):
        CONFIG.LOG_TO_FILE = False
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

    return CONFIG


def train_eval(dqn_agent, replay_buffer, env, eval_env, device, logger, CONFIG):
    """Train and evaluate agent on given environments according to given configuration."""
    # Log to TensorBoard and W&B
    if CONFIG.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="tensorboard_logs")
    if CONFIG.USE_WANDB:
        import wandb

        wandb.init(project="implementations-dqn", config=CONFIG)
        wandb.watch(dqn_agent.q_net)

    # Check if SAVE_DIR is defined
    if not CONFIG.SAVE_DIR:
        logger.warning("No save directory specified: the model will be lost!")
    else:
        # Episodic return of saved best model
        saved_model_eval_episode_return = -float("inf")
        # Unique save directory to prevent rewrite
        unique_save_dir = f"{CONFIG.SAVE_DIR}/{CONFIG.ENV_NAME}/{get_timestamp()}/"

    # Setup epsilon decay function
    get_epsilon = get_linear_anneal_func(
        CONFIG.EPSILON_START, CONFIG.EPSILON_END, CONFIG.EPSILON_DURATION
    )

    # Main training loop
    episode_return = 0
    episode_i = 0
    eval_i = 0
    obs = env.reset()
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

        # Update target network periodically
        # NOTE(seungjaeryanlee): The paper specifies frequency measured in the number
        # of parameter updates, which is dependent on UPDATE_FREQUENCY. However, in
        # the code the frequency is measured in number of perceived states. We follow
        #  the latter.
        # https://github.com/deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/dqn/NeuralQLearner.lua#L356  # noqa: B950
        if step_i % CONFIG.TARGET_NET_UPDATE_FREQUENCY == 0:
            dqn_agent.update_target_q_net()

        # Prepare for next step
        episode_return += rew
        obs = next_obs

        # Prepare for next episode if episode is finished
        if done:
            # Log episode metrics
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

            # Prepare for new episode
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
                        np.expand_dims(eval_obs, 0), epsilon=CONFIG.EVAL_EPSILON
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
                    unique_save_dir,
                    filename="best",
                    q_net=dqn_agent.q_net,
                    optimizer=dqn_agent.optimizer,
                )
                logger.info(f"Model succesfully saved at {unique_save_dir}")

            eval_i += 1

    # Save trained agent
    if CONFIG.SAVE_DIR:
        save_models(
            unique_save_dir,
            filename="last",
            q_net=dqn_agent.q_net,
            optimizer=dqn_agent.optimizer,
        )
        logger.info(f"Model successfully saved at {unique_save_dir}")


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
    env = gym.make(CONFIG.ENV_NAME)
    eval_env = gym.make(CONFIG.ENV_NAME)

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        set_global_random_seeds(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
        # NOTE(seungjaeryanlee): Seed for env and eval_env are different for fair evaluation
        set_env_random_seeds(env, CONFIG.RANDOM_SEED)
        set_env_random_seeds(eval_env, CONFIG.RANDOM_SEED + 1)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    # Setup agent and replay buffer
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
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
        env, maxlen=CONFIG.REPLAY_BUFFER_SIZE, device=device
    )

    # Train and evaluate agent
    train_eval(dqn_agent, replay_buffer, env, eval_env, device, logger, CONFIG)


if __name__ == "__main__":
    main()
