import os

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg as scilin

from InvertedPendulum import InvertedPendulum

title = "BackStepping.mp4"

x = InvertedPendulum(init_state = np.array([0., 0., np.pi, 0.]), is_stochastic = True)
state, _ = x.Update()
A, B = x.SSModel()

A, B = x.SSModel()

Q = np.array([[1,0,0,0],[0,10,0,0],[0,0,100,0],[0,0,0,100]])
R = np.array([[0.01]])

P = scilin.solve_continuous_are(A,B,Q,R)
K = np.matmul(lin.inv(R),B.T)
K = np.matmul(K, P).reshape(4)


bm, pm, pl = x.getSystemParameter()

totmass = bm + pm
momentum1 = pm*pl/2

a1, a2 = 1, 1

a1x, a2x = a1 * 0.5, a2 * 0.5
def animated(*args):
    global state
    
    ct = np.cos(state[2])
    st = np.sin(state[2])
    
    if ct < np.cos(np.pi/4) and ct > -np.cos(np.pi/4):
        tt = np.tan(np.pi/4) * np.sign(ct) * np.sign(st)
        ct = np.cos(np.pi/4) * np.sign(ct)
    else:
        tt = st/ct
        
    force = -momentum1*state[3]**2*st + (totmass)*9.8*tt +(bm + pm*st**2)*pl/(2*ct) * (a1 * state[3] + a2*(state[3] + a1*state[2]))
    state, _ = x.Update(iforce = force)
    if(abs(state[2]) > np.pi):
        state[2] = (state[2] + np.pi) % (2*np.pi) - np.pi
    
    return x.graphs

anim = animation.FuncAnimation(x.EnvFig, func = animated, interval = 17, blit = True, frames = 900)

anim.save(title, fps = 60, dpi = 144)

os.system(".\{}".format(title))