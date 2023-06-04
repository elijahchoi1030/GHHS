import gym
import time
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tp_utils import agentCartPole

parser = argparse.ArgumentParser()
parser.add_argument('--Q', action='store_true') 
parser.add_argument('--DQN', action='store_true') 
# parser.add_argument('--plot', nargs='+', default=[])
args = parser.parse_args() 

if args.Q:
    NUM_EPHISODE = 100000

    # Training
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    agent = agentCartPole("Q Learning", discrete_size=(1, 1, 6, 3))
    for eph in tqdm(range(NUM_EPHISODE)):
        state, _ = env.reset()
        action = agent.chooseAction(state)
        terminated = False
        while not terminated:
            next_state, reward, terminated, _, _ = env.step(action)
            agent.train(state, action, reward, next_state)
            next_action = agent.chooseAction(next_state)
            state, action = next_state, next_action

    agent.saveWeight(header="original")

    # Rendering
    env = gym.make('CartPole-v1', render_mode="human")
    for eph in range(10):
        state, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.chooseAction(state)
            state, _, terminated, truncated, _ = env.step(action)

if args.DQN:
    NUM_EPHISODE = 1000

    env = gym.make('CartPole-v1', render_mode="rgb_array")
    agent = agentCartPole("DQN")
    for eph in tqdm(range(NUM_EPHISODE)):
        state, _ = env.reset()
        state = np.reshape(state, [1, -1])
        terminated = False
        while not terminated:
            action = agent.chooseAction(state)
            next_state, reward, terminated, _, _ = env.step(action)
