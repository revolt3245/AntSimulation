import numpy as np

def CostFunction(x, xr):
    return np.dot(x - xr, x - xr)/2

def MPPI(dynamics, xinit, xr, uinit, Trajcount, Ocount, Timestep, deviation):
    Stot = []
    utot = []
    Smin = 0
    for t in range(Trajcount):
        x = np.copy(xinit)
        S = 0
        for i in range(Ocount):
            u = uinit + np.random.normal(0, deviation, uinit.shape)
            x += dynamics(x, u, Timestep) * Timestep
            S += CostFunction(x, xr)
        Stot += [S]
        utot += [u]
        if t == 0:
            Smin = S
        elif Smin > S:
            Smin = S
            
    Stot = np.array(Stot) - Smin
    Sexp = np.exp(-Stot).reshape(-1)
    utot = np.array(utot).reshape(-1)
    print(Sexp)
    print(utot)
    u = np.sum(utot * Sexp)/np.sum(Sexp)
    
    return u
    