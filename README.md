# Human-level control through Deep Reinforcement Learning

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=black)](https://travis-ci.com/seungjaeryanlee/implementations-dqn)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=flake8)](https://travis-ci.com/seungjaeryanlee/implementations-dqn)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=isort)](https://travis-ci.com/seungjaeryanlee/implementations-dqn)
[![pytest Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=pytest)](https://travis-ci.com/seungjaeryanlee/implementations-dqn)

This repository is a implementation of the paper [Human-level control through Deep Reinforcement Learning](/papers.pdf).

For implementations of other deep learning papers, check the centralized [implementations](https://github.com/seungjaeryanlee/implementations) repository!

### Summary 📝

![DQN Architecture](https://user-images.githubusercontent.com/6107926/61592574-ed770d00-ac0f-11e9-85f2-328aea8a84a6.png)

Deep Q-Network (DQN) is a reinforcement learning algorithm that extends the tabular Q-Learning algorithm to large complex environments using neural networks. To train the algorithm efficiently, the authors suggest using **Experience Replay** and **Target Networks**. 

Instead of the traditional Q-Learning algorithm that discards the interaction experience after learning from it once, DQN saves all these experience into a "replay buffer." This allows minibatch learning, which lowers variance and accelerates learning. Target network slows down the update of the Q-network that is used to compute the target of the MSE loss, which also lowers variance.

### Results 📊

This repository uses **TensorBoard** for offline logging and **Weights & Biases** for online logging. You can see the all the metrics in [my summary report at Weights & Biases](https://app.wandb.ai/seungjaeryanlee/implementations-dqn/reports?view=seungjaeryanlee%2FSummary)!

| | | |
|-|-|-|
| ![Individual Episode Return](https://user-images.githubusercontent.com/6107926/61592376-85bfc280-ac0d-11e9-9e04-c49cb43b91ce.png) | ![Individual Evaluation Episode Return](https://user-images.githubusercontent.com/6107926/61592377-85bfc280-ac0d-11e9-8571-e9f6725d3561.png) | ![Individual TD Loss](https://user-images.githubusercontent.com/6107926/61592378-85bfc280-ac0d-11e9-9a9a-1b5adaf71835.png) |




### Installation 🧱

This repository has [requirements.txt](/requirements.txt) for simple installation of PyPI packages.

```bash
pip install -r requirements.txt
```
