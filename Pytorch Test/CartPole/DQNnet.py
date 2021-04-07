import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from InvertedPendulum import InvertedPendulum

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayMemory(object):
    def __init__(self, capacity = 10000):
        self.capacity = capacity
        self.memory = []
        
    def push(self, Data):
        if(len(self.memory) < self.capacity):
            self.memory += [Data]
        else:
            del self.memory[0]
            self.memory += [Data]
        return self
            
    def sample(self, batch_size):
        state, action, next_state, reward = zip(*random.sample(self.memory, batch_size))
        #return random.sample(self.memory, batch_size)
        return torch.vstack(state), torch.tensor(action).to(device).view(-1,1), torch.vstack(next_state), torch.Tensor(reward).to(device).view(-1,1)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.l1 = nn.Linear(4,32, bias = True)
        self.l2 = nn.Linear(32,64, bias = True)
        self.l3 = nn.Linear(64,2, bias = True)
        
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)
        
        self.l1 = nn.Sequential(self.l1, nn.Tanh())
        self.l2 = nn.Sequential(self.l2, nn.Tanh())
        self.l3 = nn.Sequential(self.l3, nn.Sigmoid())
        
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        
        return out

def reward(state, done):
    reward = 0
    if(abs(state[0]) <= 1e-2):
        reward += 1
        if(abs(state[1]) <= 1):
            reward += 1
    
    if(abs(state[2]) <= np.pi*1e-2):
        reward += 1
        if(abs(state[3]) <= np.pi):
            reward += 1
    if(np.pi - abs(state[2]) <= np.pi*1e-2):
        reward -= 10
        
    if(done):
        reward -= 10
    return reward
    
Environment = InvertedPendulum(init_state = np.array([0., 0., 0., 0.]), is_stochastic = True)

policy_net = DQN().to(device)#play
target_net = DQN().to(device)#update
target_net.load_state_dict(policy_net.state_dict())
    
ReplayBuffer = ReplayMemory()

criterion = nn.MSELoss().to(device)
#criterion = nn.KLDivLoss().to(device)
optimizer = optim.Adam(policy_net.parameters(), lr =1e-4)

episode = 1000
gamma = 0.9

batch_size = 500

Environment.is_Animate(False)

#train
for i in range(episode):
    #initialize
    Environment.Initialize(init_state = np.array([0., 0., 0., 0.]), is_stochastic = True)
    
    cstate = torch.Tensor(Environment.getState()).to(device)
    done = False
    step = 0
    #loop
    while((not done) and (step < 1000)):
        if random.random() < 1/(i+1):
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = policy_net(cstate).argmax().item()
            
        if(action == 0):
            force = -100
        else:
            force = 100
            
        #apply action
        nstate, done = Environment.Update(iforce = force)
        nstate[2] = (nstate[2] + np.pi)%(2*np.pi)-np.pi
        
        #reward
        r = torch.Tensor([reward(nstate, done)]).to(device)
        
        nstate_T = torch.Tensor(nstate).to(device)
        
        #push to replay buffer
        ReplayBuffer.push((cstate, action, nstate_T, r))
        
        cstate = nstate_T.clone()
        step += 1
    #train
    if(len(ReplayBuffer) >= batch_size):
        s, a, ns, r = ReplayBuffer.sample(batch_size)
        optimizer.zero_grad()
        qas = policy_net(s).gather(1, a)
        nqas = target_net(ns).max(1)[0].detach().view(-1,1)
        expect = r + gamma * nqas
        
        loss = criterion(qas, expect)
        loss.backward()
        optimizer.step()
        
    if(i%10 == 9):
        target_net.load_state_dict(policy_net.state_dict())
        print("loss : {}".format(loss))

Environment.Initialize(init_state = np.array([0., 0., 0., 0.]), is_stochastic = True)
Environment.is_Animate(True)

gstate = torch.Tensor(Environment.getState()).to(device)
count = 0

def animated(*args):
    global count, gstate
    action = policy_net(gstate).argmax().item()
    if(action == 0):
        force = 100
    else:
        force = -100
        
    if(count == 180):
        state, _ = Environment.Update(iforce = force, disturbance = np.array([0., 100.]))
    else:
        state, _ = Environment.Update(iforce = force)
        
    gstate = torch.Tensor(state).to(device)
    count += 1
    return Environment.graphs

anim = animation.FuncAnimation(Environment.EnvFig, func = animated, interval = 17, blit = True, frames = 900)

anim.save("InvertedPendulum DQN.mp4", fps = 60, dpi = 144)