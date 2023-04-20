from hw2_utils import env
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

alpha = np.append(np.arange(0, 0.1, 0.01), np.arange(0.1, 1, 0.05))
for N in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], desc='N', position=0):
    averageRMS = []
    for a in tqdm(alpha, desc='a', position=1, leave=False):
        runs = []
        for i in range(500):
            runs.append(env(alpha=a, num_states=19).learnTDN(n=N))
        averageRMS.append(np.average(runs))
    plt.plot(alpha, averageRMS, label=f'TD({N})')

plt.legend()
plt.ylim(0.1, 0.3)
plt.show()

