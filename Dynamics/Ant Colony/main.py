import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''
Empty - 0
Food - 1
Wall - 2
'''
class Environment:
    def __init__(self, GridWorld, XGrid, YGrid, AntNum = 100, Colony = np.array([-10, -10])):
        self.GridWorld = GridWorld
        self.XGrid = XGrid
        self.YGrid = YGrid
        
        self.PheromoneBuffer = []
        
        self.AntBuffer = [Ant(Colony, dev = 100) for i in range(AntNum)]
        
        self.PheromoneIndexBuffer = [[[]for i in range(len(YGrid))] for j in range(len(XGrid))]
    
    def Update(self, step):
        for i in self.PheromoneBuffer:
            i.Update(step)
        self.EliminatePheromone()
        
        for i in self.AntBuffer:
            i.Update(self, step)
    
    def Observe(self, pos):
        # return Gridworld[x,y]
        x, y = self.PosToIndex(pos)
        return self.GridWorld[x,y]
    
    def EliminatePheromone(self):
        b_num = len(self.PheromoneBuffer)
        self.PheromoneBuffer = [i for i in self.PheromoneBuffer if i.lifecount <= i.halflife * 5]
        a_num = len(self.PheromoneBuffer)
        
        elimination = b_num - a_num
        
        self.PheromoneIndexBuffer = [[[k-elimination for k in j if k >= elimination] for j in i] for i in self.PheromoneIndexBuffer]
    
    def ShowPheromone(self, pos):
        Addition = 0
        weight = 0
        x, y = self.PosToIndex(pos)
        for i in self.PheromoneIndexBuffer[x][y]:
            w = self.PheromoneBuffer[i].intensity
            weight += w
            Addition += w * self.PheromoneBuffer[i].dir
        return Addition, weight
    
    def PosToIndex(self, pos):
        xpos = pos[0]
        ypos = pos[1]
        
        Xg = (self.XGrid[:-1] + self.XGrid[1:])/2
        Yg = (self.YGrid[:-1] + self.YGrid[1:])/2
        
        x_ind = len([i for i in Xg if i < xpos])
        y_ind = len([i for i in Yg if i < ypos])
        
        return x_ind, y_ind
    
    def AddPheromone(self, pos, direction):
        idx = len(self.PheromoneBuffer)
        self.PheromoneBuffer.append(pheromone(pos, direction))
        x, y = self.PosToIndex(pos)
        self.PheromoneIndexBuffer[x][y].append(idx)
        
    def GetAntPos(self):
        return [i.pos for i in self.AntBuffer]
    
    def GetPheromone(self):
        return [i.pos for i in self.PheromoneBuffer]
        
class Ant:
    def __init__(self, pos, dev = 5):
        self.ActionBuffer = [pos];
        self.State = False
        self.dev = dev
        
        self.pos = pos
        
    def Update(self, Env, step):
        if Env.Observe(self.pos) == 1:
            self.State = True
            n_pos = self.ActionBuffer[-1]
            direction = self.pos - n_pos
            self.pos = n_pos
            self.ActionBuffer.pop(-1)
            Env.AddPheromone(self.pos, direction)
        elif Env.Observe(self.pos) == 2:
            self.pos = self.ActionBuffer[-1]
        else:
            if self.State:
                if len(self.ActionBuffer) != 0:
                    action_ph, weight_ph = Env.ShowPheromone(self.pos)
                    n_pos = self.ActionBuffer[-1]
                    direction = (self.pos - n_pos)/step
                    direction = (direction * 0.5 + action_ph)/(weight_ph + 0.5)
                    self.pos = n_pos
                    self.ActionBuffer.pop(-1)
                    Env.AddPheromone(self.pos, direction)
                else:
                    self.State = False
                    self.ActionBuffer.append(self.pos)
            else:
                action_ph, weight_ph = Env.ShowPheromone(self.pos)
                action = np.random.normal(0, self.dev, 2)
                action = (action * 0.5 + action_ph)/(weight_ph + 0.5)
                pos = self.pos + step * action
                if Env.Observe(pos) != 2:
                    self.pos = pos
                    self.ActionBuffer.append(pos)
                    
class pheromone:
    def __init__(self, pos, direction, halflife = 20):
        self.pos = pos
        self.intensity = 1
        self.dir = direction
        self.halflife = halflife
        
        self.lifecount = 0
        
    def Update(self, step):
        self.lifecount += 1
        
        self.intensity *= 2**(-1/self.halflife)
        
x = np.linspace(-25, 25, 51)
y = np.linspace(-25, 25, 51)

X, Y = np.meshgrid(x, y)

Z = np.zeros([51, 51])

Z[39, 39] = 1
Z[11,11] = 0

Z[0,:] = 2
Z[50,:] = 2
Z[:,0] = 2
Z[:,50] = 2

env = Environment(Z, x, y, AntNum = 300)

A = np.vstack(env.GetAntPos())

fig, ax = plt.subplots(figsize = (9,9))
g = ax.contourf(X, Y, -Z, cmap = "gray")
ants = ax.scatter(A[:,0], A[:,1], s = 10, c = [1., 0., 0.])
pheromones = ax.scatter([], [], s = [], c = [0., 1., 0.])

def animated(*args):
    global env
    env.Update(1/60)
    
    A = np.vstack(env.GetAntPos())
    P = env.GetPheromone()
    
    ants.set_offsets(A)
    if(len(P) > 0):
        P = np.vstack(P)
        pheromones.set_offsets(P)
    return [ants, pheromones]

anim = animation.FuncAnimation(fig, func = animated, frames = 900, blit = True, interval = 17)