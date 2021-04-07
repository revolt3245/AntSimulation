import gym
from gym.envs.registration import register
import numpy as np
import random

#register(id = 'FrozenLake-v2', entry_point = 'gym.envs.toy_text:FrozenLakeEnv', kwargs = {'map_name':'4x4', 'is_slippery':False})

env = gym.make("FrozenLake-v2")

Q = np.zeros([env.observation_space.n,env.action_space.n])
Q4x4 = Q.reshape(4,4,4)
episode = 1000

lr = 0.9
gamma = 0.9

env.reset()

#learning
for i in range(episode):
    state = env.reset()
    done = False
    step = 0
    t_reward = 0
    while not done and step < 250:
        #epsilon greedy
        if(random.random() < 1/(i//100+1)):
            action = random.randrange(0, env.action_space.n)
        else:
            action = np.argmax(Q[state,:])
            
        nstate, reward, done, _ = env.step(action)
        
        if(reward <= 0 and done):
            reward = -1
        
        Q[state, action] = (1-lr)*Q[state, action] + lr*(reward + gamma * np.max(Q[nstate,:]))
        state = nstate
        step += 1
        
        t_reward += reward
        
    print("episode {} : total reward is {}".format(i+1, t_reward))
    
print(Q)

done = False
state = env.reset()
while not done:
    action = np.argmax(Q[state,:])
    nstate, reward, done, _ = env.step(action)
    
    state = nstate
    env.render()