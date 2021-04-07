import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#2Dof
class Manipulator:
    def __init__(self):
        self.Base = np.array([0.,0.,0.,1.]).T
        self.__Rod1Length__ = 1
        
        self.__BaseRotz__ = 0
        self.__BaseRotx__ = 0
        
        self.__Rod1Load__ = 10
        
    def __Translate__(self, trans):
        res = np.eye(4)
        res[0:3,-1] = trans
        
        return res
    
    def __Rotx__(self, rotx):
        res = np.eye(4)
        res[1:3, 1:3] = [[np.cos(rotx), -np.sin(rotx)],[np.sin(rotx),np.cos(rotx)]]
        return res
    
    def __Roty__(self, roty):
        res = np.eye(4)
        res[(2,0), (2,0)] = [[np.cos(roty), -np.sin(roty)],[np.sin(roty),np.cos(roty)]]
        
        return res
    
    def __Rotz__(self, rotz):
        res = np.eye(4)
        res[0:2, 0:2] = [[np.cos(rotz), -np.sin(rotz)],[np.sin(rotz),np.cos(rotz)]]
        return res
    
    def __Translation__(self, trans, rot):
        res = self.__Translate__(trans)
        res = np.matmul(self.__Rotx__(rot[0]), res)
        res = np.matmul(self.__Roty__(rot[1]), res)
        res = np.matmul(self.__Rotz__(rot[2]), res)
        
        return res
    
    def getRod1Pos(self):
        res = np.matmul(self.__Translate__([0,0,self.__Rod1Length__]), self.Base)
        res = np.matmul(self.__Rotx__(self.__BaseRotx__), res)
        res = np.matmul(self.__Rotz__(self.__BaseRotz__), res)
        
        return res
    
    def getBaseRotz(self):
        return self.__BaseRotz__
    
    def getBaseRotx(self):
        return self.__BaseRotx__
    
    def getLoad(self):
        return self.__Rod1Load__
    
    def setBaseRotz(self, rotz):
        self.__BaseRotz__ = rotz
        
    def mutBaseRotz(self, mutz):
        self.__BaseRotz__ += mutz
        
    def setBaseRotx(self, rotx):
        if(rotx > np.pi/2):
            self.__BaseRotx__ = np.pi/2
        elif(rotx < -np.pi/2):
            self.__BaseRotx__ = -np.pi/2
        else:
            self.__BaseRotx__ = rotx
            
    def mutBaseRotx(self, mutx):
        if(mutx >= np.pi/2):
            self.__BaseRotx__ = np.pi/2
        elif(mutx <= -np.pi/2):
            self.__BaseRotx__ = -np.pi/2
        else:
            self.__BaseRotx__ += mutx
            
    def Jacobian(self):
        """
        from rotx, rotz to rod pos(3x2) matrix
        input vec = [rotx, rotz]^T
        output vec = [rodx, rody, rodz]^T
        """
        res = np.zeros((3,2))
        res[0,0] = np.cos(self.__BaseRotx__) * np.sin(self.__BaseRotz__)
        res[0,1] = np.sin(self.__BaseRotx__) * np.cos(self.__BaseRotz__)
        res[1,0] = -np.cos(self.__BaseRotx__) * np.cos(self.__BaseRotz__)
        res[1,1] = np.sin(self.__BaseRotx__) * np.sin(self.__BaseRotz__)
        res[2,0] = -np.sin(self.__BaseRotz__)
        res[2,1] = 0
        
        return res * self.__Rod1Length__
    
X = Manipulator()

base = X.Base
rod1 = X.getRod1Pos()

Xomega = np.pi/4
Zomega = np.pi/2

Dim = np.vstack([base, rod1])

fig = plt.gcf()
ax = fig.add_subplot(111, projection = '3d')

ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([0.0, 1.1])

graphs, = ax.plot(Dim[:,0], Dim[:,1], Dim[:,2])

def animated(*args):
    global X, Xomega
    rx = X.getBaseRotx()
    if(rx * Xomega > 0) and (abs(rx) >= np.pi/2):
        Xomega = -Xomega
        
    X.mutBaseRotx(Xomega/60)
    X.mutBaseRotz(Zomega/60)
    
    print(X.Jacobian())
    
    rod1 = X.getRod1Pos()
    Dim = np.vstack([base, rod1])
    
    graphs.set_data(Dim[:,0], Dim[:,1])
    graphs.set_3d_properties(Dim[:,2], zdir = 'z')
    
    return graphs,

#anim = animation.FuncAnimation(fig, func = animated, interval = 17, blit = True, frames = 900)