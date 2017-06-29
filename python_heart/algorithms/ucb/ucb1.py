import math

import numpy as np

class UCB1():
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return

  def initialize(self, n_arms):
    self.counts = [1 for col in range(n_arms)]
    self.values = [1.0 for col in range(n_arms)]
    return

  def select_arm(self):
    n_arms = len(self.counts)
    #for arm in range(n_arms):
      #if self.counts[arm] == 0:
        #return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    a = np.array(ucb_values)
    return a.argsort()[-2:]

  def update(self, arms, reward):
    for chosen_arm in arms:
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
    return
