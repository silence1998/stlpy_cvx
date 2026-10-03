import os
import sys

# 数据目录：默认读取仓库中已提交的 results/ 数据。
# 若要画自己运行 sweep/计时脚本后得到的新数据，把 outputs 目录作为第一个参数传入，例如：
#   python plots/heatmap/either_or/plot_alpha_beta.py outputs
# （参数先按当前目录解析，不存在时再按本子项目根目录解析。）
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if len(sys.argv) > 1:
    _ROOT = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.join(_PROJ_DIR, sys.argv[1])
else:
    _ROOT = os.path.join(_PROJ_DIR, "results")
DATA_DIR = os.path.join(_ROOT, "heatmap", "either_or", "alpha_beta")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
plt.rcParams['text.usetex'] = True
map_ = np.load(os.path.join(DATA_DIR, "map_robustness.npy"), allow_pickle=True)
rho_1_list = np.load(os.path.join(DATA_DIR, "alpha_list.npy"))
rho_2_list = np.load(os.path.join(DATA_DIR, "beta_list.npy"))
map_ = map_[0:8, 0:8]
print(map_)
column_names = rho_1_list[0: 8]
row_indices = rho_2_list[0: 8]
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0, vmax=0.2)
ax.set_xlabel(r'$\alpha$', fontsize=16)
ax.set_ylabel(r'$\beta$', fontsize=16)
plt.title('robustness')
plt.show()

map_ = np.load(os.path.join(DATA_DIR, "map_cost.npy"), allow_pickle=True)
rho_1_list = np.load(os.path.join(DATA_DIR, "alpha_list.npy"))
rho_2_list = np.load(os.path.join(DATA_DIR, "beta_list.npy"))
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
column_names = rho_1_list
row_indices = rho_2_list
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0.02, vmax=0.1)
ax.set_xlabel(r'$\alpha$', fontsize=16)
ax.set_ylabel(r'$\beta$', fontsize=16)
plt.title('optimal cost')
plt.show()

map_ = np.load(os.path.join(DATA_DIR, "map_solve_time.npy"), allow_pickle=True)
rho_1_list = np.load(os.path.join(DATA_DIR, "alpha_list.npy"))
rho_2_list = np.load(os.path.join(DATA_DIR, "beta_list.npy"))
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
column_names = rho_1_list
row_indices = rho_2_list
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0.15, vmax=0.45)
ax.set_xlabel(r'$\alpha$', fontsize=16)
ax.set_ylabel(r'$\beta$', fontsize=16)
plt.title('solve time [s]')
plt.show()

