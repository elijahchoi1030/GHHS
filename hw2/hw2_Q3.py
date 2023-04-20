from hw2_utils import windyGrid
import matplotlib.pyplot as plt

Q3 = windyGrid()
plt.plot(Q3.learnSarsa())
Q3.printPolicy()
plt.show()