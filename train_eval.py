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
import torch
import torch.optim as optim

from dqn.agents import DQNAgent
from dqn.networks import QNetwork
from dqn.replays import CircularReplayBuffer, Transition
from utils import (
    get_linear_anneal_func,
    get_logger,
    load_models,
    make_reproducible,
    save_models,
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
    env = gym.make(CONFIG.ENV_NAME)
    eval_env = gym.make(CONFIG.ENV_NAME)

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
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    replay_buffer = CircularReplayBuffer(
        env, maxlen=CONFIG.REPLAY_BUFFER_SIZE, device=device
    )
    optimizer = optim.Adam(q_net.parameters())
    get_epsilon = get_linear_anneal_func(
        CONFIG.EPSILON_START, CONFIG.EPSILON_END, CONFIG.EPSILON_DURATION
    )
    dqn_agent = DQNAgent(env, q_net, optimizer, device)

    # Load trained agent
    if CONFIG.LOAD_PATH:
        load_models(CONFIG.LOAD_PATH, q_net=q_net, optimizer=optimizer)
    # Check if SAVE_DIR is defined
    if not CONFIG.SAVE_DIR:
        logger.warning("No save directory specified: the model will be lost!")

    if CONFIG.USE_WANDB:
        wandb.watch(q_net)

    episode_return = 0
    episode_i = 0
    eval_episode_i = 0
    for step_i in range(CONFIG.ENV_STEPS + 1):
        # Interact with the environment and save the experience
        epsilon = get_epsilon(step_i)
        action = dqn_agent.select_action(obs, epsilon)
        next_obs, rew, done, info = env.step(action)
        replay_buffer.append(Transition(obs, action, rew, next_obs, done))

        # Train QNetwork
        if len(replay_buffer) >= CONFIG.MIN_REPLAY_BUFFER_SIZE:
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
                    wandb.log({"TD Loss": td_loss, "Epsilon": epsilon}, step=step_i)

        # Update target network periodically
        if step_i % CONFIG.TARGET_NET_UPDATE_FREQUENCY == 0:
            dqn_agent.update_target_q_net()

        # Prepare for next step
        obs = next_obs
        episode_return += rew

        # Prepare for next episode if episode is finished
        if done:
            # Log episode metrics
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

            # Prepare for new episode
            env.reset()
            episode_return = 0
            episode_i += 1

        # Evaluate agent periodically
        if step_i % CONFIG.EVAL_FREQUENCY == 0:
            eval_done = False
            eval_obs = eval_env.reset()
            eval_episode_return = 0
            while not eval_done:
                eval_action = dqn_agent.select_action(eval_obs, epsilon=0)
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

    # Save trained agent
    if CONFIG.SAVE_DIR:
        unique_save_dir = save_models(
            CONFIG.SAVE_DIR, suffix="last", q_net=q_net, optimizer=optimizer
        )
        logger.info(f"Model succesfully saved at {unique_save_dir}")


if __name__ == "__main__":
    main()
