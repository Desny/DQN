import argparse
import torch
import random
from prioritized.agent import DoubleDqnAgent
from prioritized.env import WrappedEnv
from prioritized.replay import PrioritizedReplayMemory
import prioritized.preprocessor as preprocessor
from prioritized.preprocessor import LinearSchedule
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
    parser.add_argument('--priority_exponent', default=0.6, type=float, help='alpha')
    parser.add_argument('--importance_sampling_exponent_begin_value', default=0.4, type=float, help='beta_start')
    parser.add_argument('--importance_sampling_exponent_end_value', default=1., type=float)
    parser.add_argument('--min_replay_capacity_fraction', default=0.05, type=float)
    parser.add_argument('--uniform_sample_probability', default=1e-3, type=float)
    parser.add_argument('--normalize_weights', default=True, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--exploration_epsilon_begin_value', default=1., type=float)
    parser.add_argument('--exploration_epsilon_end_value', default=0.01, type=float)
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

    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(cfg.min_replay_capacity_fraction * cfg.capacity *
                    cfg.num_action_repeats),
        decay_steps=int(cfg.exploration_epsilon_decay_frame_fraction *
                        cfg.num_episodes * cfg.num_train_frames),
        begin_value=cfg.exploration_epsilon_begin_value,
        end_value=cfg.exploration_epsilon_end_value)

    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(cfg.min_replay_capacity_fraction * cfg.capacity),
        end_t=(cfg.num_episodes *
               int(cfg.num_train_frames / cfg.num_action_repeats)),
        begin_value=cfg.importance_sampling_exponent_begin_value,
        end_value=cfg.importance_sampling_exponent_end_value)

    memory = PrioritizedReplayMemory(
        cfg.capacity,
        cfg.priority_exponent,
        importance_sampling_exponent_schedule,
        cfg.uniform_sample_probability,
        cfg.normalize_weights)

    net_file = 'prioritized/weights/policy_net_weights_1214.pth'
    agent = DoubleDqnAgent(cfg, n_actions, memory, exploration_epsilon_schedule, net_file)

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
                memory.add(Transition(s_t, a_t, s_t1, r_t1, d_t1), agent.max_seen_priority)

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
