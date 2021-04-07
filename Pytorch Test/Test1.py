import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ivalue = torch.Tensor([[0,0,1,1],[0,1,0,1]]).to(device).T
ovalue = torch.Tensor([0,1,1,0]).to(device).reshape(4,1)

epoch = 5000

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.l1 = nn.Linear(2,3, bias = True)
        self.l2 = nn.Linear(3,1, bias = True)
        
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        
        self.l1 = nn.Sequential(self.l1, nn.ReLU())
        self.l2 = nn.Sequential(self.l2, nn.Sigmoid())
        
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return out
    
t = Test().to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(t.parameters(), lr =1e-2)

for i in range(epoch):
    optimizer.zero_grad()
    output = t(ivalue)
    cost = criterion(output, ovalue)
    cost.backward()
    optimizer.step()
    
