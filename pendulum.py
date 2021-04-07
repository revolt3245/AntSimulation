import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize = (12, 12));

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

graphs, = ax.plot([], [])

theta = np.pi/2 * np.ones([2,1])
omega = np.zeros([2,1])

g = 9.8
l = 0.5 * np.ones([2,1])
m = 0.5 * np.ones([2,1])
d = 0.1 * np.ones([2,1])

tick = 1/6000

def Dynamics(theta, omega):
    dtheta = omega
    SEquation = -m*g*l*np.sin(theta)
    return dtheta, SEquation