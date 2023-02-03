import argparse
import torch
import random
from double_dqn.agent import DoubleDqnAgent
from double_dqn.env import WrappedEnv
from double_dqn.replay import ReplayMemory
import double_dqn.preprocessor as preprocessor
import dm_env
from collections import namedtuple
import time

StepType = dm_env.StepType
device = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'discount'))


def get_args():
    parser = argparse.ArgumentParser(description='hyperparameters')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--eps_start', default=1.0, type=float)
    parser.add_argument('--eps_end', default=0.1, type=float)
    parser.add_argument('--target_update_period', default=6400, type=int)
    parser.add_argument('--num_episodes', default=100000, type=int)
    parser.add_argument('--num_train_frames', default=1000000, type=int)
    parser.add_argument('--exploration_epsilon_decay_frame_fraction', default=0.02, type=float)
    parser.add_argument('--max_noop_steps', default=30, type=int)
    parser.add_argument('--min_noop_steps', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--environment_height', default=84, type=int)
    parser.add_argument('--environment_width', default=84, type=int)
    parser.add_argument('--capacity', default=50000, type=int, help='capacity of replay memory')
    parser.add_argument('--additional_discount', default=0.99, type=float, help='scale discount value in preprocess')
    parser.add_argument('--max_abs_reward', default=1., type=float)
    parser.add_argument('--resize_height', default=84, type=int, help='resize shape in preprocess')
    parser.add_argument('--resize_width', default=84, type=int)
    parser.add_argument('--num_action_repeats', default=4, type=int)
    parser.add_argument('--num_stacked_frames', default=4, type=int)
    parser.add_argument('--learn_period', default=16, type=int, help='time interval for updating an agent')
    parser.add_argument('--mode', default='eval', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg = get_args()
    random.seed(cfg.seed)
    env = WrappedEnv('SpaceInvaders-v0', cfg.max_noop_steps, cfg.min_noop_steps)
    n_actions = env.action_spec.n
    memory = ReplayMemory(cfg.capacity)
    net_file = 'double_dqn/weights/policy_net_weights_1214.pth'
    agent = DoubleDqnAgent(cfg, n_actions, memory, net_file)

    def preprocessor_builder():
        return preprocessor.preprocess(
            additional_discount=cfg.additional_discount,
            max_abs_reward=cfg.max_abs_reward,
            resize_shape=(cfg.resize_height, cfg.resize_width),
            num_action_repeats=cfg.num_action_repeats,
            num_pooled_frames=2,
            zero_discount_on_life_loss=True,
            num_stacked_frames=cfg.num_stacked_frames,
            grayscaling=True,
        )
    preprocess = preprocessor_builder()

    if cfg.mode == 'train':
        for episode in range(cfg.num_episodes):
            timestep = env.reset()
            preprocessor.reset(preprocess)
            a_t = None
            while True:
                timestep_group = preprocess(timestep)
                if timestep_group is None:
                    timestep = env.step(a_t)
                    continue

                action = agent.select_action(timestep_group)
                timestep = env.step(action)
                if timestep_group.first():
                    s_t = timestep_group.observation
                    a_t = action
                    continue

                s_t1 = timestep_group.observation
                r_t1 = timestep_group.reward
                d_t1 = timestep_group.discount
                memory.add(Transition(s_t, a_t, s_t1, r_t1, d_t1))

                s_t = s_t1
                a_t = action

                if agent.frame_t % cfg.learn_period == 0:
                    agent.update()

                if timestep_group.last():
                    break

            print('episode:{0} memory_size:{1} frame_t:{2}'.format(episode, memory.size, agent.frame_t))
    else:
        timestep = env.reset()
        a_t = None
        while True:
            timestep_group = preprocess(timestep)
            if timestep_group is None:
                timestep = env.step(a_t)
                env.render()
                continue
            a_t = action = agent.predict_action(timestep_group.observation)
            timestep = env.step(action)
            env.render()
            time.sleep(0.05)
            if timestep_group.last():
                break
