import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg as scilin

from InvertedPendulum import InvertedPendulum

count = 0

x = InvertedPendulum(init_state = np.array([0., 0., 0., 0.]), is_stochastic = True)
state, _ = x.Update()
A, B = x.SSModel()

Q = np.array([[1,0,0,0],[0,10,0,0],[0,0,100,0],[0,0,0,100]])
R = np.array([[0.01]])

P = scilin.solve_continuous_are(A,B,Q,R)
K = np.matmul(lin.inv(R),B.T)
K = np.matmul(K, P).reshape(4)

def animated(*args):
    global count, state
    force = np.dot(state - np.array([0,0,0,0]), K)
    if(count == 180):
        state, _ = x.Update(iforce = -force, disturbance = np.array([0., 1000.]))
    else:
        state, _ = x.Update(iforce = -force)
        
    count += 1
    return x.graphs

anim = animation.FuncAnimation(x.EnvFig, func = animated, interval = 17, blit = True, frames = 900)

anim.save("InvertedPendulum LQR.mp4", fps = 60, dpi = 144)