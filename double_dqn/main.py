import argparse
import torch
import random
from double_dqn.agent import DoubleDqnAgent
from double_dqn.env import WrappedEnv
from double_dqn.replay import ReplayMemory

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description='hyperparameters')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--eps_start', default=1.0, type=float)
    parser.add_argument('--eps_end', default=0.1, type=float)
    parser.add_argument('--eps_decay', default=10000, type=int)
    parser.add_argument('--target_update_period', default=10, type=int)
    parser.add_argument('--num_episodes', default=1000000, type=int)
    parser.add_argument('--max_noop_steps', default=30, type=int)
    parser.add_argument('--min_noop_steps', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--environment_height', default=84, type=int)
    parser.add_argument('--environment_width', default=84, type=int)
    parser.add_argument('--capacity', default=10000, type=int, help='capacity of replay memory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg = get_args()
    random.seed(cfg.seed)
    env = WrappedEnv('SpaceInvaders-v0', cfg.max_noop_steps, cfg.min_noop_steps)
    n_actions = env.action_spec.n
    memory = ReplayMemory(cfg.capacity)
    agent = DoubleDqnAgent(cfg, n_actions, memory)


    # for i in range(cfg.num_episodes):




