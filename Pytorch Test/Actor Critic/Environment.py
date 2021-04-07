import gym
from gym import utils
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np

class CartPoleEnv(gym.Env):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masscart + self.masspole)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)/2
        self.inertia = self.masspole * self.length**2/4
        self.force_mag = 10.0
        
        self.x_threshold = 2.4
        
        self.theta_threshold = np.pi
        
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold * 2, np.finfo(np.float32).max])
        
        action = 1.0
        self.action_space = spaces.Box(action, -action, dtype = np.float64)
        self.observation_space = spaces.Box(-high, high, dtype = np.float64)
        
        self.seed();
        
        self.viewer = None
        self.state = None
        
        self.step_beyond_done = None
        
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * action
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        
    
    def reset(self):
        pass
    
    def render(self):
        pass
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None