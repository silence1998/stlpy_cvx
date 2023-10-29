import numpy as np

from Models.double_integral_with_sub_state import DoubleIntegral

from SCvx_solver import SCvxSolverFixTime
from experiments.either_or_main import EitherOr

map_robustness = np.zeros((7, 7))
map_cost = np.zeros((7, 7))
map_solve_time = np.zeros((7, 7))
trajectory_map = np.zeros((7, 7, 4, 25))

list_trust_region = []
list_robustness = []
alpha_list = [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
beta_list = [2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5]
for j in range(0, 7):
    for i in range(0, 7):
        alpha = alpha_list[j]
        beta = beta_list[i]
        K = 25
        iterations = 20
        tr_radius = 20
        sigma = 24
        list_trust_region.append(tr_radius)

        num_obstacles = 1
        num_groups = 5
        targets_per_group = 2
        Multitask_ = EitherOr(K)

        m = DoubleIntegral(K, sigma, Multitask_.spec, max_k=10, smin_C=0.1,
                           x_init=np.array([2.0, 2.0, 0, 0]), x_final=np.array([7.5, 8.5, 0, 0]))

        m.settingStateBoundary(x_min=np.array([0.0, 0.0, -1.0, -1.0]), x_max=np.array([10.0, 10.0, 1.0, 1.0]))
        m.settingControlBoundary(u_min=np.array([-1, -1]), u_max=np.array([1, 1]))
        m.settingWeights(u_weight=1.0, velocity_weight=0.1)
        solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius, w_nu=5e4, rho_1=0.1, rho_2=0.7, alpha=alpha, beta=beta)

        X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()

        #list_robustness.append(X_robust[-1, 0])
        map_robustness[j, i] = X_robust[-1, 0]
        map_cost[j, i] = solver.optimal_cost
        map_solve_time[j, i] = solver.solving_time
        trajectory_map[j, i, :, :] = X

np.save("map_robustness.npy", map_robustness)
np.save("map_cost.npy", map_cost)
np.save("map_solve_time.npy", map_solve_time)
np.save("alpha_list.npy", np.array(alpha_list))
np.save("beta_list.npy", np.array(beta_list))
np.save("trajectory_map.npy", trajectory_map)

# map_ = np.load("map.npy", allow_pickle=True)
# w_nu_list = np.load("w_nu_list.npy")
# list_trust_region = np.load("list_trust_region.npy")
# map_ = map_[0:8, 0:8]
# print(map_)
# column_names = list_trust_region[0:8]
# w_nu_list_ = ['1e2', '5e2', '1e3', '5e3', '1e4', '5e4', '1e5', '5e5', '1e6']
# row_indices = w_nu_list_[0: 8]
# data_df = pd.DataFrame(map_, index=row_indices, columns=column_names)
# f, ax = plt.subplots()
# #ax.set_yscale("log")
# # ax.imshow(-map_[0:-1, :], cmap='hot', interpolation='nearest')
# # plt.show()
# ax = sns.heatmap(data_df, vmin=0, vmax=0.2)
# plt.xlabel('initial trust region')
# plt.ylabel('penalty coefficient')
# plt.title('robustness')
# plt.show()

