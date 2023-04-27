import argparse
import gym
import matplotlib.pyplot as plt
from hw3_utils import functionApproximation, saveData, loadData

NUM_TILE = 8
NUM_EPI = 1000
EPSILON = 0
ALPHAS = [0.01/8, 0.02/8, 0.05/8, 0.1/8, 0.2/8, 0.5/8]
GAMMA = 1
CONVERGE = 1E-6

parser = argparse.ArgumentParser()
parser.add_argument('-r', action='store_true') 
parser.add_argument('-n', action='store_true') 
args = parser.parse_args() 

if args.n:
    with open(f"./datas.csv", 'w') as f:
        f.write("")

if args.r:
    for ALPHA in ALPHAS:
        ALPHA = 0.5/8
        env = gym.make('MountainCar-v0', render_mode="human")
        env.metadata["render_fps"] = 120
        agent = functionApproximation(NUM_TILE, EPSILON, ALPHA, GAMMA, CONVERGE, env)
        while True:

            # Each ephisode
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

            if agent.epi in [10, 100, 1000]:
                saveData(agent.weight, header=f"weight epi={agent.epi}&a=0.{ALPHA*80}/8")
            
            if agent.epi % 100 == 0:
                print(agent.epi)

            if agent.converge:
                saveData(agent.weight, header=f"weight epi={agent.epi}&a=0.{ALPHA*80}/8")
                break
        print(f"The weight converged in runs of {agent.epi}.")
        env.close()
        saveData(agent.steps_per_epi, header=f"steps a=0.{ALPHA*80}/8")

for i ,ALPHA in enumerate(ALPHAS):
    plt.subplot(2, 3, i+1)
    plt.plot(loadData(header=f"steps a=0.{ALPHA*80}/8"))
    plt.yscale('log')
    plt.title(f"steps a=0.{ALPHA*80}/8")
# plt.show()



ALPHA = ALPHAS[2]
env = gym.make('MountainCar-v0', render_mode="human")
agent = functionApproximation(NUM_TILE, EPSILON, ALPHA, GAMMA, CONVERGE, env)
agent.weight = loadData(header=f"weight epi=1969&a=0.0.2/8")
for _ in range(10):
    terminal = False
    state, _ = env.reset()
    while not terminal:
        action = agent.chooseAction(state)
        state, _, terminal, _, _ = env.step(action)
env.close()

