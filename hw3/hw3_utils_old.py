import random
import numpy as np
import pygame

class agent:
    def __itit__(self, ):
        self.iht = IHT(4096)

def featureVector(obs, action, iht, num_tile=8, num_act=3, pos_bound=(-1.2, 0.6), vel_bound=(-0.07, 0.07)):
    """
    '''
        Tile coding: Check the textbook page 217.
        Using 8 tiles divided by 8 of each dimension, boundary is divided by 57 pieces in each dimension.
        By asymmetrical offsets, we use (0, 0), (1, 3), (2, 6)... = (i, 3*i%num_tile) [textbook 219-220]
        To find the block that the state is in, first scale and offset the bound into [0, 8), than floor.
    '''
    pos_offset = (pos_bound[1]-pos_bound[0])/(num_tile*(num_tile-1)+1)
    vel_offset = (vel_bound[1]-vel_bound[0])/(num_tile*(num_tile-1)+1)
    state_vector = []
    for i in range(num_tile):
        projected_pos = (obs[0] - (pos_bound[0] + (1-num_tile+i)*pos_offset))/(pos_offset*num_tile)
        state_vector.append(projected_pos//1)
        
        projected_vel = (obs[1] - (vel_bound[0] + (1-num_tile+3*i%num_tile)*vel_offset))/(vel_offset*num_tile)
        if projected_vel == num_tile:
            state_vector.append(num_tile-1)
        else:
            state_vector.append(projected_speed//1)
    
    '''
        The state vector is created by tile coding.
        The feature vector has 3 times dimension of the state vector, each region only used for each actions.
        For example,    action 0: [2, 3, 0, 0, 0, 0]
                        action 1: [0, 0, 2, 3, 0, 0]
                        action 2: [0, 0, 0, 0, 2, 3]
    '''
    feature_vector = np.zeros(len(state_vector)*num_act)
    feature_vector[action*len(state_vector):(action+1)*len(state_vector)] = state_vector
    
    # feature_vector = np.append(state_vector, action)
    
    # feature_vector = np.array(state_vector) + np.ones(len(state_vector))*num_tile*action
    return feature_vector
    """

    # Used External Tile Coding
    pos_len, vel_len = pos_bound[1]-pos_bound[0], vel_bound[1] - vel_bound[0]
    return tiles(iht, num_tile, [num_tile*obs[0]/pos_len, num_tile*obs[1]/pos_len], action)


def actionValue(state, action, W, iht):
    return np.dot(W, featureVector(state, action, iht))


def chooseAction(state, W, env, epsilon=0.1):
    '''
        By epsilon-greedy choosing. 
        Probability of e=0.9, exploitation.
        Probability of e=0.1, exploration.
    '''
    if random.random() > epsilon:
        action_val = []
        for act in range(env.action_space.n):
            action_val.append(actionValue(state, act, W))
        return np.argmax(action_val)
    else:
        return env.action_space.sample()
    
    
def showScreen(env):
    if env.screen is None:
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode(
            (env.screen_width, env.screen_height)
        )
    env.render()
    pygame.event.pump()
    env.clock.tick(env.metadata["render_fps"])
    pygame.display.flip()
    


"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the 
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed. 
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have 
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)), 
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile 
coordinates are to be returned without being converted to indices).
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

from math import floor, log
from itertools import zip_longest

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

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles
