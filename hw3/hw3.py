import argparse
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from hw3_utils import functionApproximation, saveData, loadData

NUM_TILE = 8
NUM_EPI = 1000
# ALPHAS = [0.1/8, 0.2/8, 0.5/8]
ALPHAS = [0.5/8]*10
GAMMA = 1
CONVERGE = 1E-6

parser = argparse.ArgumentParser()
parser.add_argument('--reset_data', action='store_true') 
parser.add_argument('--save_data', action='store_true') 
parser.add_argument('--run_simulation', action='store_true') 
parser.add_argument('--plot', nargs='+', default=[])
parser.add_argument('--render', action='store_true')
args = parser.parse_args() 

if args.reset_data:
    with open(f"./datas.csv", 'w') as f:
        f.write("")

if args.run_simulation:
    for i, ALPHA in enumerate(ALPHAS):
        env = gym.make('MountainCar-v0', render_mode="rgb_array")
        env.metadata['render_fps'] = 120
        agent = functionApproximation(NUM_TILE, ALPHA, GAMMA, CONVERGE, env)

        for _ in tqdm(range(5000)):
            # Each episode
            num_steps = 0
            terminal = False
            state, _ = env.reset()
            action = agent.chooseAction(state)
            while not terminal:
                next_state, _, terminal, _, _ = env.step(action)
                next_action = agent.chooseAction(next_state)
                agent.updateWeight(state, action, next_state, next_action, terminal)
                state, action = next_state, next_action
                num_steps += 1
            agent.steps_per_epi.append(num_steps)

            if agent.epi in [10, 100, 1000] and args.save_data:
                saveData(agent.weight, header=f"weight epi={agent.epi}&a=0.{ALPHA*80}/8")
                pass

        if args.save_data:
            saveData(agent.iht.dictionary, header="iht")
            saveData(agent.weight, header=f"weight epi={agent.epi}&a=0.{ALPHA*80}/8")
            # saveData(agent.steps_per_epi, header=f"steps a=0.{ALPHA*80}/8")
            saveData(agent.steps_per_epi, header=f"steps a=0.{ALPHA*80}/8&t={i}")

        print(f"The weight converged in runs of {agent.epi}.")
        env.close()


if args.render:
    for ALPHA in ALPHAS:
        env = gym.make('MountainCar-v0', render_mode="human")
        env.metadata["render_fps"] = 120
        env.reset()
        # time.sleep(10) For Recording
        agent = functionApproximation(NUM_TILE, ALPHA, GAMMA, CONVERGE, env)
        agent.weight = loadData(header=f"a=0.{ALPHA*80}/8", find=True, header2="weight")
        print(f"alpha = {ALPHA}")
        for _ in tqdm(range(10)):
            terminal, trun = False, False
            state, _ = env.reset()
            while not (terminal or trun):
                action = agent.chooseAction(state)
                state, _, terminal, trun, _ = env.step(action)
        env.close()


if "steps_per_epi" in args.plot:
    for i ,ALPHA in enumerate(ALPHAS):
        plt.subplot(1, len(ALPHAS), i+1)
        plt.plot(loadData(header=f"a=0.{ALPHA*80}/8", find=True, header2="steps"))
        plt.yscale('log')
        plt.title(f"steps a=0.{ALPHA*80:.0f}/8")
        plt.xlabel("episodes")
        plt.ylabel("steps")
    plt.show()

if "steps_per_epi (avg)" in args.plot:
    steps_per_epi = []
    for i ,ALPHA in enumerate(ALPHAS):
        steps_per_epi.append(loadData(header=f"steps a=0.{ALPHA*80}/8&t={i}")[0:1500])
    avg_steps_per_epi = np.mean(steps_per_epi, axis=0)
    plt.plot(avg_steps_per_epi)
    plt.yscale('log')
    plt.title(f"steps per episode (average) alpha=0.{ALPHA*80:.0f}/8")
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.show()

if "cost_to_go (alpha)" in args.plot:
    fig = plt.figure(figsize=plt.figaspect(1/len(ALPHAS)))
    for k ,ALPHA in enumerate(ALPHAS):
        env = gym.make('MountainCar-v0', render_mode="rgb_array")
        agent = functionApproximation(NUM_TILE, ALPHA, GAMMA, CONVERGE, env)
        agent.weight = loadData(header=f"a=0.{ALPHA*80}/8", find=True, header2="weight")
        resolution = 200
        pos_offset = (env.observation_space.high[0] - env.observation_space.low[0])/resolution
        vel_offset = (env.observation_space.high[1] - env.observation_space.low[1])/resolution
        position = np.arange(env.observation_space.low[0], env.observation_space.high[0], pos_offset) + pos_offset/2
        velocity = np.arange(env.observation_space.low[1], env.observation_space.high[1], vel_offset) + vel_offset/2

        cost_to_go = np.zeros([resolution, resolution])
        for i, pos in enumerate(position):
            for j, vel in enumerate(velocity):
                action_val = []
                for act in range(agent.num_action):
                    action_val.append(agent.actionValue((pos, vel), act))
                cost_to_go[j, i] = -max(action_val)

        position, velocity = np.meshgrid(position, velocity) 
        ax = fig.add_subplot(1, len(ALPHAS), k+1, projection='3d')
        ax.plot_surface(position, velocity, cost_to_go)
        ax.set_title(f"cost to go a=0.{ALPHA*80}/8")
    plt.show()

if "cost_to_go (epi)" in args.plot:
    ALPHA = ALPHAS[0]
    episode = [10, 100, 1000, 10001]
    fig = plt.figure()
    for k, epi in enumerate(episode):
        env = gym.make('MountainCar-v0', render_mode="rgb_array")
        agent = functionApproximation(NUM_TILE, ALPHA, GAMMA, CONVERGE, env)
        agent.weight = loadData(header=f"weight epi={epi}&a=0.{ALPHA*80}/8")
        resolution = 57
        pos_offset = (env.observation_space.high[0] - env.observation_space.low[0])/resolution
        vel_offset = (env.observation_space.high[1] - env.observation_space.low[1])/resolution
        position = np.arange(env.observation_space.low[0], env.observation_space.high[0], pos_offset) + pos_offset/2
        velocity = np.arange(env.observation_space.low[1], env.observation_space.high[1], vel_offset) + vel_offset/2

        cost_to_go = np.zeros([resolution, resolution])
        for i, pos in enumerate(position):
            for j, vel in enumerate(velocity):
                action_val = []
                for act in range(agent.num_action):
                    action_val.append(agent.actionValue((pos, vel), act))
                cost_to_go[j, i] = -max(action_val)

        position, velocity = np.meshgrid(position, velocity)
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        ax.plot_surface(position, velocity, cost_to_go, linewidth=1, cmap=cm.gist_earth)
        ax.set_title(f"cost to go episode={epi}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("-max{q(s, a, w)}")
    plt.show()

if "cost_to_go (epi_wire)" in args.plot:
    ALPHA = ALPHAS[0]
    episode = [10, 100, 1000, 10001]
    # episode = [1]
    fig = plt.figure()
    for k, epi in enumerate(episode):
        env = gym.make('MountainCar-v0', render_mode="rgb_array")
        agent = functionApproximation(NUM_TILE, ALPHA, GAMMA, CONVERGE, env)
        agent.weight = loadData(header=f"weight epi={epi}&a=0.{ALPHA*80}/8")
        print(f"weight epi={epi}&a=0.{ALPHA*80}/8")
        print(agent.weight[0])
        resolution = 57
        pos_offset = (env.observation_space.high[0] - env.observation_space.low[0])/resolution
        vel_offset = (env.observation_space.high[1] - env.observation_space.low[1])/resolution
        position = np.arange(env.observation_space.low[0], env.observation_space.high[0], pos_offset) + pos_offset/2
        velocity = np.arange(env.observation_space.low[1], env.observation_space.high[1], vel_offset) + vel_offset/2

        cost_to_go = np.zeros([resolution, resolution])
        for i, pos in enumerate(position):
            for j, vel in enumerate(velocity):
                action_val = []
                for act in range(agent.num_action):
                    action_val.append(agent.actionValue((pos, vel), act))
                cost_to_go[j, i] = -max(action_val)

        position, velocity = np.meshgrid(position, velocity)
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        # ax = fig.add_subplot(1, 1, k+1, projection='3d')
        ax.plot_wireframe(position, velocity, cost_to_go)
        ax.set_title(f"cost to go episode={epi}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("-max{q(s, a, w)}")
    plt.show()

if "optimum_policy" in args.plot:   
    ALPHA = ALPHAS[0]

    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    agent = functionApproximation(NUM_TILE, ALPHA, GAMMA, CONVERGE, env)
    agent.weight = loadData(header=f"a=0.{ALPHA*80}/8", find=True, header2="weight")
    resolution = 114
    pos_offset = (env.observation_space.high[0] - env.observation_space.low[0])/resolution
    vel_offset = (env.observation_space.high[1] - env.observation_space.low[1])/resolution
    position = np.arange(env.observation_space.low[0], env.observation_space.high[0], pos_offset) + pos_offset/2
    velocity = np.arange(env.observation_space.low[1], env.observation_space.high[1], vel_offset) + vel_offset/2

    optimum_policy = np.zeros([resolution, resolution])
    for i, pos in enumerate(position):
        for j, vel in enumerate(velocity):
            action_val = []
            for act in range(agent.num_action):
                action_val.append(agent.actionValue((pos, vel), act))
            optimum_policy[j, i] = np.argmax(action_val)

    position, velocity = np.meshgrid(position, velocity)
    fig, ax = plt.subplots()
    scatter = ax.scatter(position, velocity, c=optimum_policy)
    ax.set_title("Opimum Policy")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    plt.legend(*scatter.legend_elements(),)
    plt.show()