from hw2_utils import env
import matplotlib.pyplot as plt
import numpy as np

# Monte Carlo
for a in [0.01, 0.02, 0.03, 0.04]:
    runs = []
    for i in range(100):
        runs.append(env(alpha=a).learnMC())
    print(f"Learning Completed (MC alpha = {a})")
    plt.plot(np.average(runs, axis=0), label=f'MC alpha = {a}')

# Temporal Difference
for a in [0.05, 0.1, 0.15]:
    runs = []
    for i in range(100):
        runs.append(env(alpha=a).learnTD())
    print(f"Learning Completed (TD alpha = {a})")
    plt.plot(np.average(runs, axis=0), label=f'TD alpha = {a}')

plt.legend()
plt.show()




