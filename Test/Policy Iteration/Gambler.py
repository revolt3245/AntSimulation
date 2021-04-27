import numpy as np
import matplotlib.pyplot as plt

s = np.arange(100)
V = np.zeros(100)

trial = 10000
probability = 0.4

learning_rate = 0.1
gamma = 1
reset = True

s_current = 0

for i in range(trial):
    if reset:
        s_current = np.random.randint(1, 100, 1)
        reset = False

    if(s_current < 100 - s_current):
        action = np.random.randint(1, s_current + 1)
    else:
        action = np.random.randint(1, 101 - s_current)

    #coin tossing
    if np.random.random() < probability:
        s_next = s_current + action
    else:
        s_next = s_current - action

    #transition
    if s_next >= 100:
        reward = 1
        reset = True
        s_next = 100
    elif s_next <= 1:
        reward = 0
        reset = True
        s_next = 1
    else:
        reward = 0

    V[s_current-1] = (1-learning_rate) * V[s_current-1] + learning_rate * (reward + gamma * V[s_next-1])
    s_current = s_next

plt.plot(s, V)

plt.show()