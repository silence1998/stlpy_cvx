import os
import sys

# 数据目录：默认读取仓库中已提交的 results/ 数据。
# 若要画自己运行 sweep/计时脚本后得到的新数据，把 outputs 目录作为第一个参数传入，例如：
#   python plots/heatmap/multitask/plot_w_nu_trust_region.py outputs
# （参数先按当前目录解析，不存在时再按本子项目根目录解析。）
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if len(sys.argv) > 1:
    _ROOT = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.join(_PROJ_DIR, sys.argv[1])
else:
    _ROOT = os.path.join(_PROJ_DIR, "results")
DATA_DIR = os.path.join(_ROOT, "heatmap", "multitask", "w_nu_trust_region")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
plt.rcParams['text.usetex'] = True
map_ = np.load(os.path.join(DATA_DIR, "map_robustness.npy"))
list_trust_region = np.load(os.path.join(DATA_DIR, "list_trust_region.npy"))
w_nu_list = np.load(os.path.join(DATA_DIR, "w_nu_list.npy"))
map_ = map_.transpose()
print(map_)
column_names = [10, 20, 30, 40, 50, 60, 70, 80, 90]
row_indices = ['1e2', '5e2', '1e3', '5e3', '1e4', '5e4', '1e5', '5e5', '1e6']
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0, vmax=0.2)
ax.set_xlabel(r'$r^{(1)}$', fontsize=16)
ax.set_ylabel(r'$\lambda$', fontsize=16)
plt.title('robustness')
plt.show()

map_ = np.load(os.path.join(DATA_DIR, "map_cost.npy"), allow_pickle=True)
map_ = map_.transpose()
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=-0.02, vmax=0.02)
ax.set_xlabel(r'$r^{(1)}$', fontsize=16)
ax.set_ylabel(r'$\lambda$', fontsize=16)
plt.title('optimal cost')
plt.show()

map_ = np.load(os.path.join(DATA_DIR, "trajectory_map_simularity.npy"), allow_pickle=True)
map_ = map_.transpose()
print(map_)
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0, vmax=0.4)
ax.set_xlabel(r'$r^{(1)}$', fontsize=16)
ax.set_ylabel(r'$\lambda$', fontsize=16)
plt.title('Similarity of different trajectories')
plt.show()
