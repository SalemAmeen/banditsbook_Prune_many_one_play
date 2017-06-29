import matplotlib.pyplot as plt
import numpy as np

AccuracyAftrerPrune = np.load('AccuracyAftrerPrune.npy')

plt.plot(AccuracyAftrerPrune)
plt.xlabel('Number of Neurons  pruned')
plt.ylabel('Accuracy after pruning')
plt.title('UCB1 Algorithm')
plt.axis([0, 470, 0.8, 1])
plt.grid(True)
plt.show()