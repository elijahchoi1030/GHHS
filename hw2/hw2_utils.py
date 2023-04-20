import numpy as np
import random
import matplotlib.pyplot as plt

class env():
    def __init__(self, alpha=0.1, gamma=1, num_states=5, use_penalty=False):
        if not use_penalty:
            self.ground_truth = np.append(np.arange(1+num_states)/(1+num_states), 0)
            self.value = np.concatenate([[0], np.ones(num_states)/2, [0]])
        else:
            self.ground_truth = np.concatenate([[0], (np.arange(1+num_states)/(1+num_states)*2-1)[1:], [0]])
            self.value = np.zeros(num_states+2)

        self.use_penalty = use_penalty
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rms_hist = [self.RMS()]

    def learnMC(self):
        for i in range(100):
            state = 3
            visit = []
            while 1 <= state <= self.num_states:
                visit.append(state)
                state += (random.random() > 0.5)*2-1
            returnG = state // (self.num_states+1)
            for state in visit:
                self.value[state] += self.alpha*(returnG - self.value[state])
            self.rms_hist.append(self.RMS())
        return self.rms_hist
    
    def learnTD(self):
        for i in range(100):
            new = 3
            while 1 <= new <= self.num_states:
                state = new
                new = state + (random.random()>0.5)*2-1
                reward = new // (self.num_states+1)
                self.value[state] += self.alpha*(reward + \
                                    self.gamma*self.value[new] - self.value[state])
            self.rms_hist.append(self.RMS())
        return self.rms_hist
    
    def learnTDN(self, n):
        for i in range(10):
            state = 3
            visit = []
            while 1 <= state <= self.num_states:
                visit.append(state)
                state += (random.random() > 0.5)*2-1
            if not self.use_penalty:
                reward = (state // (self.num_states+1)) 
            else:
                reward = (state // (self.num_states+1))*2-1 # Assuming use penalty
            for i in range(len(visit)):
                state = visit[i]
                if i+n >= len(visit):
                    future = reward
                else:
                    future = self.value[visit[i+n]]
                self.value[state] += self.alpha*(future - self.value[state])
        return self.RMS()

    def RMS(self):
        return np.sqrt(np.sum(np.square(self.value - self.ground_truth))/self.num_states)


class windyGrid():
    def __init__(self):
        self.H, self.W, self.C = 7, 10, 4
        self.Q = np.zeros([self.H, self.W, self.C])
        self.actions = [(1,0), (0,1), (-1,0), (0,-1)]
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.epsilon = 0.1
        self.initial_state = (3, 0)
        self.terminal_state = (3, 7)
        self.living_reward = -1
        self.gamma = 1
        self.alpha = 0.5

    def chooseActions(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.randint(0, 3)
        
    def doActions(self, state, action):
        next_x = min(max(state[0] + action[0] - self.wind[state[1]], 0), self.H-1)
        next_y = min(max(state[1] + action[1], 0), self.W-1)
        return (next_x, next_y)

    def learnSarsa(self, total_time_step=8000):
        epi = 0
        episodes = []
        state = self.initial_state
        action = self.chooseActions(state)
        for i in range(total_time_step):
            next_state = self.doActions(state, self.actions[action])
            next_action = self.chooseActions(next_state)
            self.Q[state][action] += self.alpha*(self.living_reward + \
                                     self.gamma*self.Q[next_state][next_action] - self.Q[state][action])
            episodes.append(epi)
            
            if next_state == self.terminal_state:
                state = self.initial_state
                action = action = self.chooseActions(state)
                epi += 1
            else:
                state = next_state
                action = next_action
                
        return episodes
    
    def printPolicy(self):
        arrow = ('↓', '→', '↑', '←')
        policy = []
        for valueline in self.Q:
            for value in valueline:
                policy.append(arrow[np.argmax(value)])
        self.policy = np.reshape(np.array(policy), (self.H, self.W))
        self.policy[self.terminal_state] = '■'
        print(self.policy)