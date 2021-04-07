import gym
from gym.envs.classic_control import rendering

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

'''
env = gym.make("CartPole-v0")
state = env.reset()
'''

class WheelPendulum(gym.Env):
    def __init__(self):
        self.Gravity = 9.8
        self.MassPendulum = 1.0
        self.MassReactionWheel = 0.5
        
        self.LengthPendulum = 1.0
        self.RadiusReactionWheel = 0.5
        
        self.MotorConst = 3.0
        self.Resistance = 1
        
        self.WheelInertia = self.MassReactionWheel*self.RadiusReactionWheel**2/2
        self.PendulumInertia = (self.MassPendulum/4 + self.MassReactionWheel)*self.LengthPendulum**2 + self.MassReactionWheel * self.WheelInertia
        
        self.GravitationalCoef = self.Gravity * self.LengthPendulum * (self.MassPendulum/2+self.MassReactionWheel)
        
        self.MotorDamping = self.MotorConst**2 / self.Resistance
        
    def step(self, action):
        pass
    
    def reset(self):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass