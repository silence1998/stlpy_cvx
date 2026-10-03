import os
import sys

# 数据目录：默认读取仓库中已提交的 results/ 数据。
# 若要画自己运行 sweep/计时脚本后得到的新数据，把 outputs 目录作为第一个参数传入，例如：
#   python plots/heatmap/either_or/plot_alpha_beta_trajectory.py outputs
# （参数先按当前目录解析，不存在时再按本子项目根目录解析。）
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if len(sys.argv) > 1:
    _ROOT = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.join(_PROJ_DIR, sys.argv[1])
else:
    _ROOT = os.path.join(_PROJ_DIR, "results")
DATA_DIR = os.path.join(_ROOT, "heatmap", "either_or", "alpha_beta")

import matplotlib.pyplot as plt
import numpy as np
from experiments.either_or_main import EitherOr

K = 25
Multitask_ = EitherOr(K)

trajectory_map = np.load(os.path.join(DATA_DIR, "trajectory_map.npy"))

f = plt.gca()
f.set_aspect('equal')
Multitask_.add_to_plot(f)

for i in range(0, 7):
    for j in range(0, 7):
        tmp = trajectory_map[i, j, :, :]
        f.scatter(tmp[0, :], tmp[1, :])

plt.legend()
plt.show()
