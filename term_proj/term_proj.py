import gym
import keyboard
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="human")
terminated = True
while True:
    if terminated:
        #env.reset()
        terminated = False
        print(terminated)
    else:
        if keyboard.is_pressed('left'):
            state, _, terminated, truncated, _ = env.step(0)
        elif keyboard.is_pressed('right'):
            state, _, terminated, truncated, _ = env.step(1)
