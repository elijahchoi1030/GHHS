import random
import numpy as np
from math import floor

class functionApproximation:
    def __init__(self, num_tile, epsilon, alpha, gamma, conv, env):
        self.len_weight = num_tile**4
        self.iht = IHT(self.len_weight)
        self.weight = np.zeros(self.len_weight)
        self.num_tile = num_tile
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.randomAction = env.action_space.sample
        self.num_action = env.action_space.n
        self.pos_len = env.observation_space.high[0] - env.observation_space.low[0]
        self.vel_len = env.observation_space.high[1] - env.observation_space.low[1]
        self.steps_per_epi = []
        self.epi = 1
        self.converge = False
        self.conv = conv

    def featureVector(self, state, action):
        projection = [self.num_tile*state[0]/self.pos_len, self.num_tile*state[1]/self.pos_len]
        tile_idx = tiles(self.iht, self.num_tile, projection, [action])
        feature_vector = np.zeros(self.len_weight)
        for i in range(len(tile_idx)):
            feature_vector[512*i + tile_idx[i]] = 1
        return feature_vector
    
    def actionValue(self, state, action):
        return np.dot(self.weight, self.featureVector(state, action))
    
    def chooseAction(self, state):
        '''
            By epsilon-greedy choosing. 
            Probability of e=0.9, exploitation.
            Probability of e=0.1, exploration.
        '''
        if random.random() > self.epsilon:
            action_val = []
            for act in range(self.num_action):
                action_val.append(self.actionValue(state, act))
            return np.argmax(action_val)
        else:
            return self.randomAction()
        
    def updateWeight(self, state, action, next_state, next_action, terminal):
        if terminal:
            delta_weight = (-1 - self.actionValue(state, action))*self.featureVector(state, action)
            self.epi += 1
        else:
            delta_weight = (-1 + self.gamma*self.actionValue(next_state, next_action) \
                        - self.actionValue(state, action))*self.featureVector(state, action)
        if np.sqrt(np.mean(delta_weight**2)) <= self.conv:
            self.converge = True
        self.weight += self.alpha*delta_weight
    
def saveData(item, header=""):
    with open("./datas.csv", 'r') as f:
        data = f.readlines()
    line = None
    for i in range(len(data)):
        if data[i].split(',')[0] == header:
            line = i
    new_line = f"{header}"
    for it in item:
        new_line += f", {it}"
    new_line += "\n"
    if line is None:
        data.append(new_line)
    else:
        data[line] = new_line
    with open("./datas.csv", 'w') as f:
        for lin in data:
            f.write(lin)

def loadData(header=""):
    with open("./datas.csv", 'r') as f:
        data = f.readlines()
    for item in data:
        item = item.strip().split(',')
        if item[0] == header:
            return np.array(item[1:]).astype(np.float)

    


""" 
Tile Coding from http://incompleteideas.net/tiles/tiles3.html
"""

basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates


def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

