from prioritized.network import DQN
import random
from collections import namedtuple
import torch
import dm_env
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'discount'))


class DoubleDqnAgent:
    def __init__(
        self,
        cfg,
        n_actions,
        memory,
        exploration_epsilon,
        net_file=None
    ):
        self.n_actions = n_actions
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.memory = memory
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.target_update_period = cfg.target_update_period
        self.exploration_epsilon = exploration_epsilon
        self.frame_t = -1
        self.action = 0
        self.num_net_file = 0
        self.max_seen_priority = 1.

        if net_file and os.path.exists(net_file):
            print('load net_file:', net_file)
            self.policy_net.load_state_dict(torch.load(net_file, map_location=torch.device('cpu')))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        self.frame_t += 1
        if timestep is None:
            action = self.action
        else:
            sample = random.random()
            eps_threshold = self.exploration_epsilon(self.frame_t)
            if sample > eps_threshold:
                state = timestep.observation
                action = self.action = self.predict_action(state)
            else:
                action = self.action = random.randint(0, self.n_actions - 1)

        return action

    def predict_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action = self.policy_net(state).max(1)[1].detach().cpu().numpy()
        return action[0]

    def update(self):
        if self.memory.size < self.batch_size:
            return
        optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(tuple(map(lambda s: torch.from_numpy(s), batch.state))).to(device)
        action_batch = torch.tensor(batch.action).view(self.batch_size, 1).to(device)
        next_state_batch = torch.stack(tuple(map(lambda s: torch.from_numpy(s), batch.next_state))).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).view(self.batch_size, 1).to(device)
        discount_batch = torch.tensor(batch.discount, dtype=torch.float32).view(self.batch_size, 1).to(device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
            q_max = self.target_net(next_state_batch).gather(1, argmax_action)
            expected_state_action_values = reward_batch + self.gamma * q_max * discount_batch

        td_errors = state_action_values - expected_state_action_values
        td_errors_square = torch.square(td_errors)
        loss = torch.mean(td_errors_square * weights)
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        priorities = torch.abs(td_errors)
        max_priority = priorities.max()
        self.max_seen_priority = torch.max(torch.tensor(max_priority, self.max_seen_priority)).item()
        self.memory.update_priorities(indices, priorities)

        if self.frame_t % self.target_update_period == 0:
            self.num_net_file += 1
            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save(self.policy_net.state_dict(), 'prioritized/weights/policy_net_weights_{0}.pth'.format(self.num_net_file))
