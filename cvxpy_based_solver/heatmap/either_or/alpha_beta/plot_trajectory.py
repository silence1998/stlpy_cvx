import matplotlib.pyplot as plt
import numpy as np
from experiments.either_or_main import EitherOr

K = 25
Multitask_ = EitherOr(K)

trajectory_map = np.load("trajectory_map.npy")

f = plt.gca()
f.set_aspect('equal')
Multitask_.add_to_plot(f)

for i in range(0, 7):
    for j in range(0, 7):
        tmp = trajectory_map[i, j, :, :]
        f.scatter(tmp[0, :], tmp[1, :])

plt.legend()
plt.show()
