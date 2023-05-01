import numpy as np
from math import floor

class functionApproximation:
    def __init__(self, num_tile, alpha, gamma, conv, env):
        self.len_weight = num_tile**4
        self.iht = IHT(self.len_weight)
        self.iht.dictionary = loadData(header="iht")
        self.weight = np.zeros(self.len_weight)
        self.num_tile = num_tile
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
        projection = [self.num_tile*state[0]/self.pos_len, self.num_tile*state[1]/self.vel_len]
        tile_idx = tiles(self.iht, self.num_tile, projection, [action])
        feature_vector = np.zeros(self.len_weight)
        for idx in tile_idx:
            feature_vector[idx] = 1
        return feature_vector
    
    def actionValue(self, state, action):
        return np.dot(self.weight, self.featureVector(state, action))
    
    def chooseAction(self, state):
        action_val = []
        for act in range(self.num_action):
            action_val.append(self.actionValue(state, act))
        return np.argmax(action_val)
        
    def updateWeight(self, state, action, next_state, next_action, terminal):
        if terminal:
            delta_weight = (0 - self.actionValue(state, action))*self.featureVector(state, action)
            self.epi += 1
        else:
            delta_weight = (-1 + self.gamma*self.actionValue(next_state, next_action) \
                        - self.actionValue(state, action))*self.featureVector(state, action)
        # if np.sqrt(np.mean(delta_weight**2)) <= self.conv:
        #     self.converge = True
        if self.epi == 2000:
            self.converge = True
        self.weight += self.alpha*delta_weight
    
def saveData(item, header=""):
    if header == "iht":
        with open("./iht.txt", 'w') as f:
            f.write(str(item))
        return
    
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

def loadData(header="", find=False, header2=""):
    if header == "iht":
        with open("./iht.txt", 'r') as f:
            data = f.read()
        return eval(data)
    
    with open("./datas.csv", 'r') as f:
        data = f.readlines()
    for i, item in enumerate(data):
        item = item.strip().split(',')
        if find == True:
            if header in item[0] and header2 in item[0]:
                line = i
        else:
            if header == item[0]:
                return np.array(item[1:]).astype(np.float32)
    if line is None:
        print("error occured!")
    return np.array(data[line].strip().split(',')[1:]).astype(np.float32)

    


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

