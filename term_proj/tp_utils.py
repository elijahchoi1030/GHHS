import gym
import math
import random
import numpy as np

class agentCartPole():
    def __init__(self, method, **kwargs):
        self.method = method
        env = gym.make('CartPole-v1', render_mode='rgb_array')

        if method == 'Q Learning':
            self.chooseAction = self._action_Q
            self.updateWeight = self._update_Q
            self.saveWeight = self._save_Q
            self.discrete_size = kwargs.get('discrete_size', (30, 30, 30, 30))
            self.epsilon = kwargs.get('epsilon', 0.1)
            self.lr = kwargs.get('lr', 0.01)
            self.discount_factor = kwargs.get('discount_factor', 1)
            self.discretizer = discretizer(self.discrete_size)
            # self.Q = np.zeros(self.discrete_size + (env.action_space.n,))
            self.Q = 100*np.ones(self.discrete_size + (env.action_space.n,))

    def _action_Q(self, state):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        state = self.discretizer(state)
        if random.random() > self.epsilon: # exploration
            return np.argmax(self.Q[state])
        else: # exploitation
            return env.action_space.sample()

    def _update_Q(self, state, action, reward):
        state = self.discretizer(state)
        next_Q = np.argmax(self.Q[state])
        self.Q[state][action] += self.lr*(reward + self.discount_factor*next_Q - self.Q[state][action])

    def _save_Q(self, header=""):
        with open("./data.csv", 'r') as f:
            data = f.readlines()
        my_line = "weight," + header + "," + str(self.Q.shape)
        for val in np.reshape(self.Q, (-1,)):
            my_line += str(val) + ","
        line = -1
        for i, datum in enumerate(data):
            datum = datum.split(',')
            if datum[0] == "weight" and datum[1] == header:
                line = i
        if line == -1:
            data.append(my_line)
        else:
            data[line] = my_line
        with open("./data.csv", 'w') as f:
            for datum in data:
                f.write(datum)

    def _load_Q(self, header=""):
        with open("./data.csv", 'r') as f:
            data = f.readlines()
        line = -1
        for i, datum in enumerate(data):
            datum = datum.split(',')
            if datum[0] == "weight" and datum[1] == header:
                line = i
        if line != -1:
            pass



        

class discretizer():
    def __init__(self, space_size):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.state_len = 4
        self.state_bound = list(zip(env.observation_space.low, env.observation_space.high))
        self.state_bound[1] = (-0.5, 0.5)
        self.state_bound[3] = (-math.radians(50), math.radians(50))
        self.scaler = []
        for i in range(self.state_len):
            self.scaler.append(space_size[i]/(self.state_bound[i][1]-self.state_bound[i][0]))

    def __call__(self, state):
        discrete_state = []
        for i in range(self.state_len):
            stt = min(max(state[i], self.state_bound[i][0]), self.state_bound[i][1] - self.scaler[i]*0.0001)
            discrete_state.append(int(math.floor((stt - self.state_bound[i][0])*self.scaler[i])))
        return discrete_state

