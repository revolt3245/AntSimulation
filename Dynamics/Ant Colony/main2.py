import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''
Empty - 0
Food - 1
Wall - 2
'''
class Environment:
    def __init__(self, GridWorld, XGrid, YGrid, AntNum = 100, Colony = np.array([-10, -10]), FoodBuffer = []):
        self.GridWorld = GridWorld
        self.XGrid = XGrid
        self.YGrid = YGrid
        
        self.PheromoneBuffer = []
        self.HomemarkerBuffer = []
        
        self.AntBuffer = [Ant(np.copy(Colony), Colony, dev = 100) for i in range(AntNum)]
        
        self.FoodBuffer = FoodBuffer
    
    def Update(self, step):
        self.AddPheromone(np.array([14,14]))
        self.AddHomemarker(np.array([-14, -14]))
        for i in self.PheromoneBuffer:
            i.Update(step)
        self.EliminatePheromone()
        
        for i in self.HomemarkerBuffer:
            i.Update(step)
            
        for i in self.AntBuffer:
            i.Update(self, step)
    
    def Observe(self, pos):
        # return Gridworld[x,y]
        x, y = self.PosToIndex(pos)
        return self.GridWorld[x,y]
    
    def EliminatePheromone(self):
        self.PheromoneBuffer = [i for i in self.PheromoneBuffer if i.lifecount <= i.halflife * 5]
        self.HomemarkerBuffer = [i for i in self.HomemarkerBuffer if i.lifecount <= i.halflife * 5]
    
    def ShowPheromone(self, pos, radius):
        Addition = sum([i.intensity * (i.pos - pos)/lin.norm(i.pos - pos) * radius for i in self.PheromoneBuffer if 1e-3 < lin.norm(i.pos - pos) and lin.norm(i.pos - pos) < radius])
        weight = sum([i.intensity for i in self.PheromoneBuffer if 1e-3 < lin.norm(i.pos - pos) and lin.norm(i.pos - pos) < radius])
        
        return Addition, weight
    
    def ShowHomemarker(self, pos, radius):
        Addition = sum([i.intensity * (i.pos - pos)/lin.norm(i.pos - pos) * radius for i in self.HomemarkerBuffer if 1e-3 < lin.norm(i.pos - pos) and lin.norm(i.pos - pos) < radius])
        weight = sum([i.intensity for i in self.HomemarkerBuffer if 1e-3 < lin.norm(i.pos - pos) and lin.norm(i.pos - pos) < radius])
        
        return Addition, weight
    
    def PosToIndex(self, pos):
        xpos = pos[0]
        ypos = pos[1]
        
        Xg = (self.XGrid[:-1] + self.XGrid[1:])/2
        Yg = (self.YGrid[:-1] + self.YGrid[1:])/2
        
        x_ind = len([i for i in Xg if i < xpos])
        y_ind = len([i for i in Yg if i < ypos])
        
        return x_ind, y_ind
    
    def AddPheromone(self, pos):
        self.PheromoneBuffer.append(pheromone(pos))
        
    def AddHomemarker(self, pos):
        self.HomemarkerBuffer.append(homemarker(pos))
    def GetAntPos(self):
        return [i.pos for i in self.AntBuffer]
    
    def GetPheromone(self):
        return [i.pos for i in self.PheromoneBuffer]
    
    def GetPheromoneIntensity(self):
        return [i.intensity for i in self.PheromoneBuffer]
    
    def GetHomemarker(self):
        return [i.pos for i in self.HomemarkerBuffer]
    
    def GetHomemarkerIntensity(self):
        return [i.intensity for i in self.HomemarkerBuffer]
        
class Ant:
    def __init__(self, pos, home, dev = 10):
        self.ActionBuffer = [pos];
        self.State = False
        self.dev = dev
        
        self.pos = pos
        
        self.home = home
        
    def Update(self, Env, step):
        if Env.Observe(self.pos) == 1:
            self.State = True
            action_ph, weight_ph = Env.ShowHomemarker(self.pos, self.dev*step*10)
            action = (self.home - self.pos)/lin.norm(self.home - self.pos) * self.dev
            action = (action * 5 + action_ph/step/10)/(weight_ph + 5)
            self.pos = self.pos+step * action
            Env.AddPheromone(self.pos)
        elif Env.Observe(self.pos) == 2:
            self.pos = self.ActionBuffer[-1]
        else:
            if self.State:
                if lin.norm(self.pos - self.home) > self.dev * step:            
                    action_ph, weight_ph = Env.ShowHomemarker(self.pos, self.dev*step*10)
                    r_action = np.random.normal(0, self.dev, 2)
                    action = (self.home - self.pos)/lin.norm(self.home - self.pos) * self.dev
                    action = (action * 5 + action_ph/step/10 + r_action * 5)/(weight_ph + 10)
                    self.pos = self.pos + step * action
                    Env.AddPheromone(self.pos)
                else:
                    self.State = False
                    Env.AddHomemarker(self.pos)
            else:
                action_ph, weight_ph = Env.ShowPheromone(self.pos, self.dev*step*10)
                action = np.random.normal(0, self.dev, 2)
                action = (action * 5 + action_ph/step/10)/(weight_ph + 5)
                pos = self.pos + step * action
                if Env.Observe(pos) != 2:
                    self.pos = pos
                    Env.AddHomemarker(self.pos)
                    
class Food:
    def __init__(self, pos, capacity = 10):
        self.pos = pos
        self.capacity = capacity
        
    def Reduce(self):
        self.capacity -= 1
                    
class pheromone:
    def __init__(self, pos, halflife = 30):
        self.pos = pos
        self.intensity = 1
        self.halflife = halflife
        
        self.lifecount = 0
        
    def Update(self, step):
        self.lifecount += 1
        
        self.intensity *= 2**(-1/self.halflife)
        
class homemarker:
    def __init__(self, pos, halflife = 30):
        self.pos = pos
        self.intensity = 1
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

env = Environment(Z, x, y, AntNum = 50)

A = np.vstack(env.GetAntPos())

fig, ax = plt.subplots(figsize = (9,9))
g = ax.contourf(X, Y, -Z, cmap = "gray")
ants = ax.scatter(A[:,0], A[:,1], s = 10, c = [1., 0., 0.])
pheromones = ax.scatter([], [], s = [], c = [0., 1., 0.])
homemarkers = ax.scatter([], [], s = [], c = [0., 0., 1.])

def animated(*args):
    global env
    env.Update(1/60)
    
    A = np.vstack(env.GetAntPos())
    P = env.GetPheromone()
    Pi = env.GetPheromoneIntensity()
    H = env.GetHomemarker()
    Hi = env.GetHomemarkerIntensity()
    
    ants.set_offsets(A)
    if(len(P) > 0):
        P = np.vstack(P)
        pheromones.set_offsets(P)
        pheromones.set_sizes(Pi)
        
    if(len(H) > 0):
        H = np.vstack(H)
        homemarkers.set_offsets(H)
    return [ants, pheromones, homemarkers]

anim = animation.FuncAnimation(fig, func = animated, frames = 900, blit = True, interval = 17)

#anim.save("Ant.mp4", fps = 60, dpi = 144)