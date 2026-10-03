import os
import sys

# 数据目录：默认读取仓库中已提交的 results/ 数据。
# 若要画自己运行 sweep/计时脚本后得到的新数据，把 outputs 目录作为第一个参数传入，例如：
#   python plots/heatmap/multitask/plot_w_nu_trust_region_trajectory.py outputs
# （参数先按当前目录解析，不存在时再按本子项目根目录解析。）
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if len(sys.argv) > 1:
    _ROOT = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.join(_PROJ_DIR, sys.argv[1])
else:
    _ROOT = os.path.join(_PROJ_DIR, "results")
DATA_DIR = os.path.join(_ROOT, "heatmap", "multitask", "w_nu_trust_region")

import matplotlib.pyplot as plt
import numpy as np
from experiments.multitask import Multitask

num_obstacles = 1
num_groups = 5
targets_per_group = 2
K = 26
Multitask_ = Multitask(K, num_obstacles, num_groups, targets_per_group, seed=0)

trajectory_map = np.load(os.path.join(DATA_DIR, "trajectory_map.npy"))

f = plt.gca()
f.set_aspect('equal')
Multitask_.add_to_plot(f)

i = 5
for j in range(0, 3):
    tmp = trajectory_map[j, i, :, :]
    f.scatter(tmp[0, :], tmp[1, :], label='trust_region='+str((j + 1) * 10))

plt.legend()
plt.show()
