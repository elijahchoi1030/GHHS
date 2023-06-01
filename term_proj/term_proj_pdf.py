import gym
import numpy as np
import random, math
import matplotlib.pyplot as plt

env= gym.make('CartPole-v1', render_mode='human')
no_buckets = (1,1,6,3)
no_actions = env.action_space.n
state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_value_bounds[1] = (-0.5, 0.5)
state_value_bounds[3] = (-math.radians(50), math.radians(50))
# define q_value_table - it has a dimension of 1 x 1 x 6 x 3 x 2
q_value_table = np.zeros(no_buckets + (no_actions,))
# user-defined parameters
min_explore_rate = 0.1; min_learning_rate = 0.1; max_episodes = 1000
max_time_steps = 250; streak_to_end = 120; solved_time = 199; discount = 0.99
no_streaks = 0

# Select an action using epsilon-greedy policy
def select_action(state_value, explore_rate): # omitted
    if random.random() < explore_rate:
        action = env.action_space.sample() # explore
    else: # exploit
        action = np.argmax(q_value_table[state_value])
    return action

# change the exploration rate over time.
def select_explore_rate(x):
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x+1)/25)))

# Change learning rate over time
def select_learning_rate(x):
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x+1)/25)))

# Bucketize the state_value
def bucketize(state_value):
    bucket_indices = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:
            # violates lower bound
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            # violates upper bound
            # put in the last bucket
            bucket_index = no_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i]-1) * state_value_bounds[i][0] / bound_width
            scaling = (no_buckets[i]-1) / bound_width
            bucket_index = int(round(scaling*state_value[i] -offset))
        bucket_indices.append(bucket_index)
    return(tuple(bucket_indices))


# train the system
totaltime = 0
for episode_no in range(max_episodes):
    #learning rate and explore rate diminishes
    # monotonically over time
    explore_rate = select_explore_rate(episode_no)
    learning_rate = select_learning_rate(episode_no)
    # initialize the environment
    observation = env.reset()
    state_value = bucketize(observation[0])
    previous_state_value = state_value
    done = False
    time_step = 0
    while not done:
        #env.render()
        # select action using epsilon-greedy policy
        action = select_action(previous_state_value, explore_rate)
        # record new observations
        observation, reward_gain, done, truncate, info = env.step(action)
        #update q_value_table
        best_q_value = np.max(q_value_table[state_value])
        q_value_table[previous_state_value][action] += learning_rate * (reward_gain + discount * best_q_value - q_value_table[previous_state_value][action])
        # update the states for next iteration
        state_value = bucketize(observation)
        previous_state_value = state_value
        time_step += 1
        # while loop ends here

    if time_step >= solved_time:
        no_streaks += 1
    else:
        no_streaks = 0
    if no_streaks > streak_to_end:
        print('CartPole problem is solved after {} episodes.', episode_no)
        break
    print(episode_no)
env.close()