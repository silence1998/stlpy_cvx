import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
plt.rcParams['text.usetex'] = True
map_ = np.load("map_robustness.npy", allow_pickle=True)
rho_1_list = np.load("rho_1_list.npy")
rho_2_list = np.load("rho_2_list.npy")
map_ = map_
print(map_)
column_names = rho_1_list
row_indices = rho_2_list
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0, vmax=0.3)
ax.set_xlabel(r'$\psi_1$', fontsize=16)
ax.set_ylabel(r'$\psi_2$', fontsize=16)
plt.title('robustness')
plt.show()

map_ = np.load("map_cost.npy", allow_pickle=True)
rho_1_list = np.load("rho_1_list.npy")
rho_2_list = np.load("rho_2_list.npy")
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
column_names = rho_1_list
row_indices = rho_2_list
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0.02, vmax=0.03)
ax.set_xlabel(r'$\psi_1$', fontsize=16)
ax.set_ylabel(r'$\psi_2$', fontsize=16)
plt.title('optimal cost')
plt.show()

map_ = np.load("map_solve_time.npy", allow_pickle=True)
rho_1_list = np.load("rho_1_list.npy")
rho_2_list = np.load("rho_2_list.npy")
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
column_names = rho_1_list
row_indices = rho_2_list
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0.15, vmax=0.45)
ax.set_xlabel(r'$\psi_1$', fontsize=16)
ax.set_ylabel(r'$\psi_2$', fontsize=16)
plt.title('solve time [s]')
plt.show()

