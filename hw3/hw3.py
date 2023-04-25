import gym
import numpy as np
from hw3_utils import featureVector, chooseAction, actionValue

NUM_TILE = 8
NUM_EPI = 500
EPSILON = 0.1
ALPHA = 0.01/8
GAMMA = 1

# render_mode = "rgb_array" : rgb array를 return하며 이를 plot할 수 있음. 빠름.
# render_mode = "human"     : 자동으로 render()가 실행되어 pygame으로 보여줌. 느림. 
env = gym.make('MountainCar-v0', render_mode="human")
env.metadata["render_fps"] = 120

# weight = np.zeros(NUM_TILE*2*env.action_space.n)
# weight = np.zeros(NUM_TILE*2+1)
weight = np.zeros(NUM_TILE*2)
steps_per_epi = []

for _ in range(NUM_EPI):
    # Each ephisode
    num_steps = 0
    state, _ = env.reset()
    action = chooseAction(state, weight, env, epsilon=EPSILON)
    while True:
        next_state, reward, terminal, _, _ = env.step(action)
        
        if num_steps%300 == 2:
            print(weight)
            print(featureVector(state, action))
            print(actionValue(next_state, next_action, weight))
            print(actionValue(state, action, weight))
            print(ALPHA*(reward + GAMMA*actionValue(next_state, next_action, weight) \
                            - actionValue(state, action, weight))*featureVector(state, action))
            print("CUT", num_steps, '\n')
            # break
        
        if terminal:
            # Assuming terminal reward is 0
            weight += ALPHA*(0 - actionValue(state, action, weight))*featureVector(state, action)
            break
        next_action = chooseAction(next_state, weight, env, epsilon=EPSILON)
        weight += ALPHA*(reward + GAMMA*actionValue(next_state, next_action, weight) \
                        - actionValue(state, action, weight))*featureVector(state, action)
        state, action = next_state, next_action
        num_steps += 1
        
    steps_per_epi.append(num_steps)

env.close()


