import gym
import numpy as np
from hw3_utils import getNextAction, getActionValue

NUM_TILE = 8
NUM_EPI = 1
EPSILON = 0.1
ALPHA = 0.5/8
GAMMA = 1

# env = gym.make('MountainCar-v0', render_mode="rgb_array")
env = gym.make('MountainCar-v0', render_mode="human")
# weight = np.zeros(NUM_TILE*2 + 1)
weight = np.random.random(NUM_TILE*2 + 1)
steps_per_epi = []

for _ in range(NUM_EPI):
    num_steps = 0
    state, _ = env.reset()
    action = getNextAction(state, weight, epsilon=EPSILON)
    terminated = False
    while not terminated:
        next_state, reward, terminated, _, _ = env.step(action)
        next_action = getNextAction(next_state, weight, epsilon=EPSILON)
        weight += ALPHA*(reward + GAMMA*getActionValue(next_state, next_action, weight) \
                          - getActionValue(state, action, weight))*weight
        num_steps += 1
        state = next_state
        action = next_action
        print(num_steps)
    steps_per_epi.append(num_steps)

env.close()


