import numpy as np
from ModelPredictivePathIntegral import MPPI

def dynamics(x, u, Timestep):
    dx = np.zeros(x.shape)
    dxtot = np.zeros(x.shape)
    for i in range(4):
        if i == 0:
            step = 0
            mul = 1/6
        elif i == 3:
            step = Timestep
            mul = 1/6
        else:
            step = Timestep/2
            mul = 1/3
            
        xtemp = x + dx * step
        dx[0] = xtemp[1]
        dx[1] = 9.8 * np.sin(xtemp[0]) + u
        dxtot += dx * mul
    return dxtot

u = MPPI(dynamics, np.array([np.pi*1/3, 0]), np.zeros(2), np.array([0]), 100, 10, 1/60, 20)
print(u)