import matplotlib.pyplot as plt
import numpy as np
from experiments.multitask import Multitask

num_obstacles = 1
num_groups = 5
targets_per_group = 2
K = 26
Multitask_ = Multitask(K, num_obstacles, num_groups, targets_per_group, seed=0)

trajectory_map = np.load("trajectory_map.npy")

f = plt.gca()
f.set_aspect('equal')
Multitask_.add_to_plot(f)

i = 5
for j in range(0, 3):
    tmp = trajectory_map[j, i, :, :]
    f.scatter(tmp[0, :], tmp[1, :], label='trust_region='+str((j + 1) * 10))

plt.legend()
plt.show()
