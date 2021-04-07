import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class DynamicsNet(nn.Module):
    def __init__(self):
        super(DynamicsNet, self).__init__()
        
        self.l1 = nn.LSTMCell(16, 12)
        self.l2 = nn.LSTMCell(12, 12)
        self.l3_mean = nn.LSTMCell(12, 12)
        self.l3_dev = nn.LSTMCell(12, 12)
        
        nn.init.xavier_uniform_(self.l1.weight_hh)
        nn.init.xavier_uniform_(self.l2.weight_hh)
        nn.init.xavier_uniform_(self.l3_mean.weight_hh)
        nn.init.xavier_uniform_(self.l3_dev.weight_hh)
        
        nn.init.xavier_uniform_(self.l1.weight_ih)
        nn.init.xavier_uniform_(self.l2.weight_ih)
        nn.init.xavier_uniform_(self.l3_mean.weight_ih)
        nn.init.xavier_uniform_(self.l3_dev.weight_ih)
        
        self.l1 = nn.Sequential(self.l1, nn.relu())
        self.l2 = nn.Sequential(self.l2, nn.SELU())
        self.l3_mean = nn.Sequential(self.l3_mean, nn.Tanh())
        self.l3_dev = nn.Sequential(self.l3_dev, nn.Tanh())
        
    def forward(self, x):
        distribution_input = self.l1(x)
        distribution_input = self.l2(distribution_input)
        m_out = self.l3_mean(distribution_input)
        d_out = self.l3_dev(distribution_input)
        
        out = torch.cat([m_out, d_out], 0)
        
        return out