import torch
import torch.nn as nn
import torch.optim as optim

hidden_size = 64

# V(s)
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        
        self.Layer1 = nn.Linear(input_size, hidden_size)
        self.Layer2 = nn.Linear(hidden_size, hidden_size)
        self.Layer3 = nn.Linear(hidden_size, 1)
        
        self.Layer1 = nn.Sequential(self.Layer1, nn.Tanh())
        self.Layer2 = nn.Sequential(self.Layer2, nn.Tanh())
        
    def forward(self, x):
        out = self.Layer1(x)
        out = self.Layer2(out)
        out = self.Layer3(out)
        
        return out
    