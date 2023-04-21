import gym
import numpy as np
from hw3_utils import chooseAction, actionValue, showScreen

NUM_TILE = 8
NUM_EPI = 1
EPSILON = 0.1
ALPHA = 0.1/8
GAMMA = 1

# render_mode = "rgb_array" : rgb array를 return하며 이를 plot할 수 있음. 빠름.
# render_mode = "human"     : 자동으로 render()가 실행되어 pygame으로 보여줌. 느림. 
env = gym.make('MountainCar-v0', render_mode="human")
env.metadata["render_fps"] = 1000

weight = np.ones(NUM_TILE*2*env.action_space.n)
steps_per_epi = []

for _ in range(NUM_EPI):
    # Each ephisode
    num_steps = 0
    state, _ = env.reset()
    action = chooseAction(state, weight, env, epsilon=EPSILON)
    while True:
        next_state, reward, terminal, _, _ = env.step(action)
        if terminal:
            # Assuming terminal reward is 0
            weight += ALPHA*( - actionValue(state, action, weight))*weight
            break
        next_action = chooseAction(next_state, weight, env, epsilon=EPSILON)
        weight += ALPHA*(reward + GAMMA*actionValue(next_state, next_action, weight) \
                        - actionValue(state, action, weight))*weight
        state = next_state.copy()
        action = next_action
        num_steps += 1
        print(num_steps)
        # break
    steps_per_epi.append(num_steps)

env.close()


