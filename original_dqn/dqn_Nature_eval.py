# -*- coding: utf-8 -*-

import gym
import numpy as np
from collections import deque
from itertools import count
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


env = gym.make('SpaceInvaders-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# DQN algorithm

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.l1 = nn.Linear(linear_input_size, 512)
        self.l2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        return self.l2(x.view(-1, 512))


######################################################################
# Input extraction

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((84, 84), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor()])

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


######################################################################
# Evaluation
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.load_state_dict(torch.load('weights/policy_net_weights_100.pth', map_location=torch.device('cpu')))
# policy_net.load_state_dict(torch.load('weights/policy_net_weights.pth'))
policy_net.eval()

env.reset()

m = 4
state_queue = deque([], maxlen=m)
# 先存储m帧图像
for _ in range(m):
    action = env.action_space.sample()
    env.step(action)
    state_queue.append(get_screen())

total_reward = 0
rewards = []
duration = 0
state = torch.cat(tuple(state_queue), dim=1)
for t in count():
    reward = 0
    done = False
    # 每m帧完成一次action
    with torch.no_grad():
        action = policy_net(state).max(1)[1].view(1, 1)
        print('action =', action.item(), 'action_values =', policy_net(state))

    _, reward, done, _ = env.step(action.item())
    total_reward += reward
    if reward != 0:
        rewards.append(reward)
    env.render()
    if not done:
        state_queue.append(get_screen())
        state = torch.cat(tuple(state_queue), dim=1)
    else:
        duration = t + 1
        break

    time.sleep(0.05)


print('Complete, total reward = {0}  duration = {1}'.format(total_reward, duration))
print('rewards =', rewards)
env.close()
