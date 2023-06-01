import gym
import keyboard
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tp_utils import agentCartPole

NUM_EPHISODE = 100000

# Training
env = gym.make('CartPole-v1', render_mode="rgb_array")
agent = agentCartPole("Q Learning", discrete_size=(10, 10, 10, 10))
terminated = False
for eph in tqdm(range(NUM_EPHISODE)):
    state, _ = env.reset()
    action = agent.chooseAction(state)
    while not terminated:
        next_state, reward, terminated, _, _ = env.step(action)
        next_action = agent.chooseAction(next_state)
        agent.updateWeight(state, action, reward)
        state, action = next_state, next_action
agent.saveWeight()

# Rendering
env = gym.make('CartPole-v1', render_mode="human")
for eph in range(10):
    state, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.chooseAction(state)
        state, _, terminated, truncated, _ = env.step(action)
