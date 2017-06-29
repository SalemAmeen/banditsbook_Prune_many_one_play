#execfile("core.py")
from core import *

algo = UCB1([], [])
arms=[1,2,3,4,5]
horizon=100 # Playing times
num_sims=1 # How many times want to play at one time. With Particulr arm a
           # How many other arms or weights want to check to see if they work
           # well with this arm or weigh.
chosen_arms = [0.0 for i in range(num_sims * horizon)]
rewards = [0.0 for i in range(num_sims * horizon)]
cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
sim_nums = [0.0 for i in range(num_sims * horizon)]
times = [0.0 for i in range(num_sims * horizon)]

for sim in range(num_sims):
    sim = sim + 1
    algo.initialize(len(arms))
    
    for t in range(horizon):
      t = t + 1
      index = (sim - 1) * horizon + t - 1
      sim_nums[index] = sim
      times[index] = t
      
      chosen_arm = algo.select_arm()
      chosen_arms[index] = chosen_arm
      
      #reward = arms[chosen_arms[index]].draw()
      #rewards[index] = reward
      reward = 1
      rewards[index] = reward
      if t == 1:
        cumulative_rewards[index] = reward
      else:
        cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
      
      algo.update(chosen_arm, reward)