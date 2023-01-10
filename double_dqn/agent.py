from double_dqn.network import DQN
import math
import random
from collections import namedtuple
import torch
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'discount'))


class DoubleDqnAgent:
    def __init__(self, cfg, n_actions, memory):
        self.n_actions = n_actions
        self.policy_net = DQN(cfg.environment_height, cfg.environment_width, n_actions)
        self.target_net = DQN(cfg.environment_height, cfg.environment_width, n_actions)
        self.memory = memory
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.target_update_period = cfg.target_update_period
        self.learn_steps = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        if sample > eps_threshold:
            return self.predict_action(state)
        else:
            return random.randint(0, self.n_actions-1)

    def predict_action(self, state):
        action = self.policy_net(state).detach().cpu().numpy()
        return action

    def update(self):
        if self.memory.size < self.batch_size:
            return
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(self.batch_size, 1)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, 1)
        discount_batch = torch.cat(batch.discount).view(self.batch_size, 1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
            q_max = self.target_net(next_state_batch).gather(1, argmax_action)
            expected_state_action_values = reward_batch + self.gamma * q_max * discount_batch

        loss = loss_fn(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        self.learn_steps += 1

        if self.learn_steps % self.target_update_period == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save(self.policy_net.state_dict(), 'weights/policy_net_weights_{0}.pth'.format(self.learn_steps))
