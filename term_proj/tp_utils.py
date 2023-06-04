import gym
import math
import random
import numpy as np

from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras.models import Model

class agentCartPole():
    def __init__(self, method, **kwargs):
        self.method = method
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.sample_action = env.action_space.sample

        if method == 'Q Learning':
            self.chooseAction = self._action_Q
            self.train = self._update_Q
            self.saveWeight = self._save_Q
            self.lr = kwargs.get('lr', 0.01)
            self.discrete_size = kwargs.get('discrete_size', (30, 30, 30, 30))
            self.epsilon = kwargs.get('epsilon', 0.1)
            self.discount_factor = kwargs.get('discount_factor', 1)
            self.discretizer = discretizer(self.discrete_size)
            self.Q = np.zeros(self.discrete_size + (env.self.action_size,))

        if method == 'DQN':
            self.chooseAction = self._action_DQN
            self.train = self._update_DQN
            self.lr = kwargs.get('lr', 0.001)
            self.discount_factor = kwargs.get('discount_factor', 0.9)
            # self.saveWeight = self._save_Q

    def _action_Q(self, state):
        state = self.discretizer(state)
        if random.random() > self.epsilon: # exploration
            return np.argmax(self.Q[state])
        else: # exploitation
            return self.sample_action()

    def _update_Q(self, state, action, reward, next_state):
        state = self.discretizer(state)
        next_state = self.discretizer(next_state)
        pseudo_return = reward + self.discount_factor*np.max(self.Q[next_state])
        self.Q[state][action] += self.lr*(pseudo_return - self.Q[state][action])

    def _save_Q(self, header=""):
        with open("./data.csv", 'r') as f:
            data = f.readlines()
        my_line = "weight," + header + "," + str(self.Q.shape)
        for val in np.reshape(self.Q, (-1,)):
            my_line += "," + str(val)
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
                f.write(datum+"\n")

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

    def _build_DQN(self):
        network_input = Input(shape=(self.state_size,), name='network_input')
        A1 = Dense(24, activation='relu', name='A1')(network_input)
        A2 = Dense(24, activation='relu', name ='A2')(A1)
        A3 = Dense(self.action_size, activation='linear', name='A3')(A2)
        V3 = Dense(1, activation='linear', name='V3')(A2)
        network_output = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1], output_shape=(self.action_size,))([A3,V3])
        model = Model(network_input, network_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        model.summary()
        return model


    def _action_DQN(self, state):
        if random.random() > self.epsilon: # exploration
            q_value = self.model.predict(state)
            return np.argmax(q_value)
        else: # exploitation
            return self.sample_action()

    def _update_DQN(self):
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
        return tuple(discrete_state)

