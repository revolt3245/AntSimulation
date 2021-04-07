import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class Constraints:
    def __init__(self, Range, Mode = 'block'):
        self.Range = Range
        self.Mode = Mode

#6DOF Manipulator
class Manipulator:
    def __init__(self):
        #Base Dimension
        self.__Base__ = np.array([0,0,0,1])
        self.__RodLengths__ = np.array([1,1,1])
        
        #Rotation State
        self.__BaseRotation__ = np.array([0,0])#xz
        
        self.__RodRotation__ = np.zeros([2,2])#xy
        
        #Constraints
        self.__BaseConstraints__ = [Constraints([-np.pi/2, np.pi/2], 'block'), Constraints([0, 2*np.pi], 'periodic')]
        self.__RodConstraints__ = [[Constraints([-np.pi/2, np.pi/2], 'block'),Constraints([-np.pi/2, np.pi/2], 'block')],
                                   [Constraints([-np.pi/2, np.pi/2], 'block'),Constraints([-np.pi/2, np.pi/2], 'block')]]
    
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
        idx_row = [[2,2],[0,0]]
        idx_col = [[2,0],[2,0]]
        res[idx_row, idx_col] = [[np.cos(roty), -np.sin(roty)],[np.sin(roty),np.cos(roty)]]
        
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
    
    def getBase(self):
        return self.__Base__
    
    def getRodLocation(self):
        #sequence r1->r2->r3
        res = np.zeros((4,3))
        res[-1,:] = 1
        
        for i in [2,1]:
            res[:, i:] = np.matmul(self.__Translate__([0,0,self.__RodLengths__[i]]), res[:, i:])
            res[:, i:] = np.matmul(self.__Rotx__(self.__RodRotation__[i-1, 0]), res[:, i:])
            res[:, i:] = np.matmul(self.__Roty__(self.__RodRotation__[i-1, 1]), res[:, i:])
            
        res[:,:] = np.matmul(self.__Translate__([0,0,self.__RodLengths__[0]]), res[:,:])
        res[:,:] = np.matmul(self.__Rotx__(self.__BaseRotation__[0]), res[:,:])
        res[:,:] = np.matmul(self.__Rotz__(self.__BaseRotation__[1]), res[:,:])
        
        return res
    
    def setRotBase(self, rot):
        self.__BaseRotation__ = rot
        
    def setRotRod(self, rot):
        self.__RodRotation__ = rot
    
X = Manipulator()

X.setRotBase(np.array([np.pi/6, np.pi/6]))
X.setRotRod(np.array([[np.pi/3, np.pi/3],[np.pi/4, np.pi/4]]))

base = X.getBase().reshape(4,1)
rod = X.getRodLocation()

Xomega = np.pi/4
Zomega = np.pi/2

Dim = np.hstack([base, rod])

fig = plt.gcf()
ax = fig.add_subplot(111, projection = '3d')

ax.set_xlim([-3.1, 3.1])
ax.set_ylim([-3.1, 3.1])
ax.set_zlim([0.0, 3.1])

graphs, = ax.plot(Dim[0,:], Dim[1,:], Dim[2,:])
