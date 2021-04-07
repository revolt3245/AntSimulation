import gym
from Actor import Actor
from Critic import Critic

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make("Pendulum-v0")

actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
critic = Critic(env.observation_space.shape[0])

criterion = nn.MSELoss().to(device)

Actor_Optimizer = optim.SGD(actor.parameters(), lr = 1e-4)
Critic_Optimizer = optim.SGD(critic.parameters(), lr = 1e-4)

for episode in range(5):
    state = env.reset()
    returns = np.zeros(1000)
    for step in range(1000):
        pass