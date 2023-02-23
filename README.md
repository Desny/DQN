# DQN

## Introduction
This repository contains DQN related algorithms implemented in PyTorch. There are tutorial videos in Chinese on [bilibili](https://www.bilibili.com/video/BV1Rq4y1b7ML/).

## Subpackages
### original_dqn
 - Algorithm: DQN
 - Original paper: Human-level control through deep reinforcement learning
 - Reference code: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
 - Tutorial video: https://www.bilibili.com/video/BV1Rq4y1b7ML/

It contains 2 codes that `dqn_Nature_gpu.py` for training and `dqn_Nature_eval.py` for testing the model. You can run the code directly to train or test, but please be noticed of the path for saving or loading `.pth` files.

### double_dqn
 - Algorithm: Double Q-Learning
 - Original paper: Deep Reinforcement Learning with Double Q-learning
 - Reference code: https://github.com/deepmind/dqn_zoo
 - Tutorial video: https://www.bilibili.com/video/BV1i94y1X7bt/

It refers to the code named dqn_zoo implemented by DeepMind in Jax. Double_dqn in this repository is converted from Jax to PyTorch. You can run `main.py` with the config variable `--mode` to choose *train* or *eval*. It is able to restart training at breakpoints by modifying the variable *net_file* in `main.py`.

#### To be noticed:
 - The working directory needs to be configured of *DQN*. It is recommended to run code on PyCharm that the configuration for running codes would be easier, referring to the picture below.
![image](https://github.com/Desny/DQN/pics/pycharm_run_config.png)

 - If *net_file*(.pth) can not be found, the training process will start at the begining rather than breakpoints.

### prioritized
 - Algorithm: Prioritized Experience Replay
 - Original paper: Prioritized Experience Replay
 - Reference code: https://github.com/deepmind/dqn_zoo
 - Tutorial video: https://www.bilibili.com/video/BV1nG4y1S7db/

Its code structure is similar to double_dqn. You can take steps above in double_dqn to run the code.

## Dependencies
#### python3.7
```
gym==0.19.0
numpy==1.21.6
dm-env==1.5
torch==1.13.1
pyglet==1.5.21
```
