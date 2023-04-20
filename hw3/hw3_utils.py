import gym
import random
import numpy as np

def tileCoding(obs, num_tile=8, place_bound=(-1.2, 0.5), speed_bound=(-0.07, 0.07)):
    plc_offset = (place_bound[1]-place_bound[0])/(num_tile*(num_tile-1)+1)
    spd_offset = (speed_bound[1]-speed_bound[0])/(num_tile*(num_tile-1)+1)

    feature_vector = []
    for i in range(num_tile):
        feature_vector.append(((obs[0] - (place_bound[0] + (1-num_tile+i)*plc_offset))/(plc_offset*num_tile))//1)
        feature_vector.append(((obs[1] - (speed_bound[0] + (1-num_tile+3*i%num_tile)*spd_offset))/(spd_offset*num_tile))//1)
    return feature_vector

def getActionValue(state, action, W):
    return np.dot(W, np.append(tileCoding(state), action))

def getNextAction(state, W, epsilon=0.1):
    action_val = []
    for act in range(3):
        action_val.append(getActionValue(state, act, W))

    if random.random() > epsilon:
        return np.argmin(action_val)
    else:
        return random.randint(0, 2)