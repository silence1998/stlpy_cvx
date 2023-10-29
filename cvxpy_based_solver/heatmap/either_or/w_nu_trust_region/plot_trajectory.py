import matplotlib.pyplot as plt
import numpy as np
from experiments.either_or_main import EitherOr

K = 25
Multitask_ = EitherOr(K)

trajectory_map = np.load("trajectory_map.npy")

f = plt.gca()
f.set_aspect('equal')
Multitask_.add_to_plot(f)

i = 6
for j in range(0, 5):
    tmp = trajectory_map[i, j, :, :]
    f.scatter(tmp[0, :], tmp[1, :], label='trust_region='+str((j + 1) * 10))

plt.legend()
plt.show()
