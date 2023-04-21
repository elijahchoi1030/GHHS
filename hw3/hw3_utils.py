import gym
import random
import numpy as np
import pygame

def tileCoding(obs, num_tile, place_bound=(-1.2, 0.5), speed_bound=(-0.07, 0.07)):
    '''
        Tile coding: Check the textbook page 217.
        Using 8 tiles divided by 8 of each dimension, boundary is divided by 57 pieces in each dimension.
        By asymmetrical offsets, we use (0, 0), (1, 3), (2, 6)... = (i, 3*i%num_tile) [textbook 219-220]
        To find the block that the state is in, first scale and offset the bound into [0, 8), than floor.
    '''
    plc_offset = (place_bound[1]-place_bound[0])/(num_tile*(num_tile-1)+1)
    spd_offset = (speed_bound[1]-speed_bound[0])/(num_tile*(num_tile-1)+1)

    state_vector = []
    for i in range(num_tile):
        projected_place = (obs[0] - (place_bound[0] + (1-num_tile+i)*plc_offset))/(plc_offset*num_tile)
        state_vector.append(projected_place//1)
        
        projected_speed = (obs[1] - (speed_bound[0] + (1-num_tile+3*i%num_tile)*spd_offset))/(spd_offset*num_tile)
        if projected_speed == num_tile:
            state_vector.append(num_tile-1)
        else:
            state_vector.append(projected_speed)
    return state_vector


def actionValue(state, action, W, num_tile=8, num_act=3):
    '''
        The state vector is created by tile coding.
        The feature vector has 3 times dimension of the state vector, each region only used for each actions.
        For example,    action 0: [2, 3, 0, 0, 0, 0]
                        action 1: [0, 0, 2, 3, 0, 0]
                        action 2: [0, 0, 0, 0, 2, 3]
    '''
    state_vector = tileCoding(state, num_tile)
    feature_vector = np.zeros(len(state_vector)*num_act)
    feature_vector[action*len(state_vector):(action+1)*len(state_vector)] = state_vector
    if feature_vector.shape != (48,):
        print("SE")
    return np.dot(W, feature_vector)


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
    