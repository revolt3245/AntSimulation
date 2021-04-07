import os

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg as scilin

from InvertedPendulum import InvertedPendulum

count = 0

x = InvertedPendulum(init_state = np.array([0., 0., np.pi, 0.]), is_stochastic = True, is_animate = False)
state, _ = x.Update()
A, B = x.SSModel()

Q = np.array([[1,0,0,0],[0,10,0,0],[0,0,100,0],[0,0,0,100]])
R = np.array([[0.01]])

P = scilin.solve_continuous_are(A,B,Q,R)
K = np.matmul(lin.inv(R),B.T)
K = np.matmul(K, P).reshape(4)

x.is_Animate(True)
def animated(*args):
    global count, state
    
    if abs(state[2]) > np.pi/3:
        force = np.sign(state[3]) * 50
        #force = -state[3] * (abs(state[2])-np.pi)*15
    else:
        force = -np.dot(state - np.array([0,0,0,0]), K)
        
    state, _ = x.Update(iforce = force)
    if(abs(state[2]) > np.pi):
        state[2] = (state[2] + np.pi) % (2*np.pi) - np.pi
    
    return x.graphs

anim = animation.FuncAnimation(x.EnvFig, func = animated, interval = 17, blit = True, frames = 900)

anim.save("SwingUp2.mp4", fps = 60, dpi = 144)

os.system(".\SwingUp2.mp4")