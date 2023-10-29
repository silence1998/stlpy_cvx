import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
plt.rcParams['text.usetex'] = True
map_ = np.load("map_robustness.npy")
list_trust_region = np.load("list_trust_region.npy")
w_nu_list = np.load("w_nu_list.npy")
print(map_)
column_names = [10, 20, 30, 40, 50, 60, 70, 80, 90]
row_indices = ['1e2', '5e2', '1e3', '5e3', '1e4', '5e4', '1e5', '5e5', '1e6']
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0, vmax=0.3)
ax.set_xlabel(r'$r^{(1)}$', fontsize=16)
ax.set_ylabel(r'$\lambda$', fontsize=16)
plt.title('robustness')
plt.show()

map_ = np.load("map_cost.npy", allow_pickle=True)
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=-0.1, vmax=0.1)
ax.set_xlabel(r'$r^{(1)}$', fontsize=16)
ax.set_ylabel(r'$\lambda$', fontsize=16)
plt.title('optimal cost')
plt.show()
