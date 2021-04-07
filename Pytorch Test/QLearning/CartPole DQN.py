import gym

import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("CartPole-v0")
s = env.reset()
env.render()
env.close()
