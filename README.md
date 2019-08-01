# Human-level control through Deep Reinforcement Learning

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=black)](https://black.readthedocs.io/en/stable/)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=flake8)](http://flake8.pycqa.org/en/latest/)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=isort)](https://pypi.org/project/isort/)
[![pytest Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-dqn.svg?label=pytest)](https://docs.pytest.org/en/latest/)

[![numpydoc Docstring Style](https://img.shields.io/badge/docstring-numpydoc-blue.svg)](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue.svg)](https://pre-commit.com/)

This repository is a implementation of the paper [Human-level control through Deep Reinforcement Learning](/papers.pdf).

**Please ⭐ this repository if you found it useful!**


---

### Table of Contents 📜

- [Summary](#summary-)
- [Installation](#installation-)
- [Running](#running-)
- [Results](#results-)
- [Differences from the Paper](#differences-from-the-paper-)
- [Reproducibility](#reproducibility-)

For implementations of other deep learning papers, check the **[implementations](https://github.com/seungjaeryanlee/implementations) repository**!

---

### Summary 📝

![DQN Architecture](https://user-images.githubusercontent.com/6107926/61592574-ed770d00-ac0f-11e9-85f2-328aea8a84a6.png)

Deep Q-Network (DQN) is a reinforcement learning algorithm that extends the tabular Q-Learning algorithm to large complex environments using neural networks. To train the algorithm efficiently, the authors suggest using **Experience Replay** and **Target Networks**.

Instead of the traditional Q-Learning algorithm that discards the interaction experience after learning from it once, DQN saves all these experience into a "replay buffer." This allows minibatch learning, which lowers variance and accelerates learning. Target network slows down the update of the Q-network that is used to compute the target of the MSE loss, which also lowers variance.

### Installation 🧱

First, clone this repository from GitHub. Since this repository contains submodules, you should use the `--recursive` flag.

```bash
git clone --recursive https://github.com/seungjaeryanlee/implementations-dqn.git
```

If you already cloned the repository without the flag, you can download the submodules separately with the `git submodules` command:

```bash
git clone https://github.com/seungjaeryanlee/implementations-dqn.git
git submodule update --init --recursive
```

After cloing the repository, use the [requirements.txt](/requirements.txt) for simple installation of PyPI packages.

```bash
pip install -r requirements.txt
```

### Running 🏃

### Results 📊

This repository uses **TensorBoard** for offline logging and **Weights & Biases** for online logging. You can see the all the metrics in [my summary report at Weights & Biases](https://app.wandb.ai/seungjaeryanlee/implementations-dqn/reports?view=seungjaeryanlee%2FSummary)!

<p align="center">
  <img alt="Train Episode Return" src="https://user-images.githubusercontent.com/6107926/61592376-85bfc280-ac0d-11e9-9e04-c49cb43b91ce.png" width="32%">
  <img alt="Evaluation Episode Return" src="https://user-images.githubusercontent.com/6107926/61592377-85bfc280-ac0d-11e9-8571-e9f6725d3561.png" width="32%">
  <img alt="TD Loss" src="https://user-images.githubusercontent.com/6107926/61592378-85bfc280-ac0d-11e9-9a9a-1b5adaf71835.png" width="32%">
</p>

### Differences from the Paper 👥

### Reproducibility 🎯
