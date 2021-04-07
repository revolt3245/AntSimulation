import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#inverted Pendulum Dynacmis
class InvertedPendulum:
    def __init__(self, init_state = np.zeros(4), is_stochastic = False, is_drawed = False, is_animate = True):
        #Dynamic Const
        self.__BodyMass__ = 10
        self.__PendulumMass__ = 5
        self.__PendulumLength__ = 2
        
        self.__Gravity__ = 9.8
        
        self.__Dampers__ = np.array([1, 0.5])
        
        #Drawing Const
        self.__BoxWidth__ = 1
        self.__BoxHeight__ = 0.5
        self.__PendulumWidth__ = 0.2
        self.__AnchorBox__ = np.zeros(2)
        self.__AnchorPendulum__ = np.array([0, -0.95])
        
        #State
        '''
            State[0] - Body Pos
            State[1] - Body Vel
            State[2] - Pendulum Angle
            State[3] - Pendulum Revolution
        '''
        
        if(is_drawed):
            self.__UpdateGraph__()
        else:
            self.__State__ = init_state
            self.__is_stochastic__ = is_stochastic
            
            self.__StateQueue__ = []
            self.__QueueCapacity = 600
            
            self.EnvFig = plt.figure(figsize = (9,12))
            
            self.ax = self.EnvFig.add_subplot(2,1,1)
            self.ax.set_xlim([-7.6, 7.6])
            self.ax.set_ylim([-3.1, 3.1])
            self.ax.set_aspect('equal', adjustable = 'box')
            
            self.StatePlotAx = self.EnvFig.add_subplot(2,1,2)
            self.StatePlotAx.set_xlim([0,10])
            self.StatePlotAx.set_ylim([-100, 100])
            
            self.__InitDraw__()
            
        self.__is_Drawed__ = True
        
        self.__is_Animate__ = is_animate
        
    def __InitDraw__(self):
        seqx = np.array([self.__BoxWidth__, -self.__BoxWidth__])/2
        seqy = -self.__BoxHeight__ * np.ones([2])/2
        seqx = np.hstack([seqx, -seqx]) + self.__State__[0]
        seqy = np.hstack([seqy, -seqy])
        seqx = np.hstack([seqx, seqx[0]])
        seqy = np.hstack([seqy, seqy[0]])
        
        self.Body = self.ax.plot(seqx, seqy, color = [0,0,0])[0]
        
        seqx = np.array([self.__PendulumWidth__, -self.__PendulumWidth__])/2
        seqy = -self.__PendulumLength__ * np.ones([2])/2
        seqx = np.hstack([seqx, -seqx])
        seqy = np.hstack([seqy, -seqy])
        seq = np.vstack([seqx, seqy])
        
        seq -= (self.__AnchorPendulum__ - self.__AnchorBox__).reshape(2,1)
        seq = np.matmul(self.__Rotation__(self.__State__[2]), seq)
        seq += np.array([self.__State__[0], 0]).reshape(2,1)
        seq = np.hstack([seq, seq[:,0].reshape(2,1)])
        
        self.Pendulum = self.ax.plot(seq[0,:], seq[1,:], color = [0,0,1])[0]
        
        self.StatePlot = self.StatePlotAx.plot([], [], color = [0,0,1])[0]
        
        self.graphs = [self.Body, self.Pendulum, self.StatePlot]
        
    def __UpdateGraph__(self):
        seqx = np.array([self.__BoxWidth__, -self.__BoxWidth__])/2
        seqy = -self.__BoxHeight__ * np.ones([2])/2
        seqx = np.hstack([seqx, -seqx]) + self.__State__[0]
        seqy = np.hstack([seqy, -seqy])
        seqx = np.hstack([seqx, seqx[0]])
        seqy = np.hstack([seqy, seqy[0]])
        
        self.Body.set_data(seqx, seqy)
        
        seqx = np.array([self.__PendulumWidth__, -self.__PendulumWidth__])/2
        seqy = -self.__PendulumLength__ * np.ones([2])/2
        seqx = np.hstack([seqx, -seqx])
        seqy = np.hstack([seqy, -seqy])
        seq = np.vstack([seqx, seqy])
        
        seq -= (self.__AnchorPendulum__ - self.__AnchorBox__).reshape(2,1)
        seq = np.matmul(self.__Rotation__(self.__State__[2]), seq)
        seq += np.array([self.__State__[0], 0]).reshape(2,1)
        seq = np.hstack([seq, seq[:,0].reshape(2,1)])
        
        self.Pendulum.set_data(seq[0,:],seq[1,:])
        
        timetable = np.arange(0, len(self.__StateQueue__)) * 1/60
        self.StatePlot.set_data(timetable, self.__StateQueue__)
        
        plt.draw()
        
    def __Rotation__(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s],[s, c]])
    
    def __Dynamics__(self, state, iforce, disturbance):
        pos = state[0::2]
        if(self.__is_stochastic__):
            vel = state[1::2] + (2*np.random.rand(2)-1)*5
        else:
            vel = state[1::2]
        
        force = np.array([iforce, 0]) + disturbance
        
        dpos = vel
        dvel_coef = self.__PendulumMass__*self.__PendulumLength__*np.sin(pos[1])/2
        dvel_raw = dvel_coef * np.array([vel[1]**2, self.__Gravity__]) + force - self.__Dampers__ * vel 
        
        tot_mass = self.__BodyMass__ + self.__PendulumMass__
        correl = self.__PendulumMass__ * self.__PendulumLength__*np.cos(pos[1])/2
        inertia = self.__PendulumMass__*self.__PendulumLength__**2/4
        
        dvel_mat = np.array([[tot_mass, correl], [correl, inertia]])
        
        dvel = np.matmul(lin.inv(dvel_mat), dvel_raw.reshape(2,1))
        
        dstate = np.zeros(4)
        dstate[0::2] = dpos
        dstate[1::2] = dvel.reshape(2)
        
        return dstate
    
    def __RK4__(self, mstep, iforce, disturbance):
        dstate = np.zeros(4)
        dstate_res = np.zeros(4)
        for i in range(4):
            if i == 0:
                step = 0
                mul = 1/6
            elif i == 3:
                step = mstep
                mul = 1/3
            else:
                step = mstep/2
                mul = 1/6
                
            cstate = self.__State__ + step * dstate
            dstate = self.__Dynamics__(cstate, iforce, disturbance)
            
            dstate_res += dstate * mul
            
        return dstate_res
        
    def Update(self, iforce = 0, step = 1/60, resolution = 100, disturbance = np.zeros(2)):
        mstep = step/resolution
        done = False
        
        self.__StateQueue__ += [iforce]
        
        if(len(self.__StateQueue__) > self.__QueueCapacity):
            del self.__StateQueue__[0]
            
        for i in range(resolution):
            dstate = self.__RK4__(mstep, iforce, disturbance)
            self.__State__ += dstate*mstep
        
        if (abs(self.__State__[0]) >= 5):
            done = True
        if(self.__is_Animate__):
            self.__UpdateGraph__()
        
        return self.__State__, done
    
    def SSModel(self):
        A = np.zeros([4,4])
        B = np.zeros([4,1])
        
        tot_mass = self.__BodyMass__ + self.__PendulumMass__
        correl = self.__PendulumMass__ * self.__PendulumLength__/2
        inertia = self.__PendulumMass__*self.__PendulumLength__**2/4
        
        dvel_mat = np.array([[tot_mass, correl], [correl, inertia]])
        
        dvel_coef = self.__PendulumMass__*self.__PendulumLength__/2
        dvel_raw = dvel_coef * np.array([0, self.__Gravity__])
        
        dvel_raw_F = np.array([1,0])
        dvel_F = np.matmul(lin.inv(dvel_mat), dvel_raw_F.reshape(2,1))
        
        dvel_theta = np.matmul(lin.inv(dvel_mat), dvel_raw.reshape(2,1))
        
        A[0,1] = 1
        A[2,3] = 1
        A[[1,3],[2,2]] = dvel_theta.reshape(2)
        B[[1,3], 0] = dvel_F.reshape(2)
        
        return A, B
    
    def Initialize(self, init_state = np.zeros(4), is_stochastic = False):
        self.__State__ = init_state
        self.__is_stochastic__ = is_stochastic
        
        self.__StateQueue__ = []
        self.__QueueCapacity = 600
        
        if(self.__is_Drawed__):
            self.__UpdateGraph__()
        else:
            self.EnvFig = plt.figure()
            
            self.ax = self.EnvFig.add_subplot(2,1,1)
            self.ax.set_xlim([-5.1, 5.1])
            self.ax.set_ylim([-3.1, 3.1])
            self.ax.set_aspect('equal', adjustable = 'box')
            
            self.StatePlotAx = self.EnvFig.add_subplot(2,1,2)
            self.StatePlotAx.set_xlim([0,10])
            self.StatePlotAx.set_ylim([-100, 100])
            
            self.__InitDraw__()
            
        self.__is_Drawed__ = True
        
    def is_Animate(self, is_animate):
        self.__is_Animate__ = is_animate
        return self
    
    def getState(self):
        return self.__State__
    
    def getSystemParameter(self):
        return self.__BodyMass__, self.__PendulumMass__, self.__PendulumLength__