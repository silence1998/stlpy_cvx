import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9]
row_indices = ['1e2', '5e2', '1e3', '5e3', '1e4', '5e4', '1e5', '5e5', '1e6']
map_ = np.load("map_cost.npy", allow_pickle=True)
map_ = map_.transpose()
print(map_)
index = (np.isnan(map_))
map_[index] = 1000
data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
f, ax = plt.subplots()
ax = sns.heatmap(data_df, vmin=0, vmax=2.1e-6)
plt.xlabel('initial trust region')
plt.ylabel('penalty coefficient')
plt.title('optimal cost')
plt.show()
