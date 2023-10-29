import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


map_ = np.load("map_cost.npy", allow_pickle=True)
rho_1_list = np.load("alpha_list.npy")
rho_2_list = np.load("beta_list.npy")
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
column_names = rho_1_list
row_indices = rho_2_list
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=-0.1, vmax=0)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('optimal cost')
plt.show()
