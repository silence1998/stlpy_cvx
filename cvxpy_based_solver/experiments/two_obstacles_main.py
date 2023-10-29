from time import time
import numpy as np

from Models.two_obstacles import DoubleIntegral

import matplotlib.pyplot as plt

from SCvx_solver import SCvxSolverFixTime
t_total_1 = time()
K = 41
iterations = 20

# initial trust region radius
tr_radius = 15

# Weight constants
w_nu = 1e5
# trust region variables
rho_0 = 0.00
rho_1 = 0.6  ### very good for 0.01
rho_2 = 0.9
alpha = 2.5
beta = 3.2

sigma = 4

m = DoubleIntegral(K, sigma)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius, w_nu, rho_1, rho_2, alpha, beta)
X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
print('Solve time: ', solver.solving_time, 's')
print('Compile time: ', solver.compile_time, 's')
X_converge = X
X_sub_converge = X_sub
U_converge = U

f = plt.gca()
f.set_aspect('equal')
m.add_to_plot(f)
f.scatter(X_converge[0, :], X_converge[1, :])
f.scatter(X_init[0, :], X_init[1, :], color='green')
plt.show()
difference_x = []
for i in range(len(all_X) - 2):
    difference_x.append(np.linalg.norm(all_X[i] - X_converge, 1) + np.linalg.norm(all_X_sub[i] - X_sub_converge, 1))

difference_x = np.array(difference_x)
print(difference_x)
f, ax = plt.subplots()
ax.set_yscale("log")
ax.set_ylim(1e-4, 2e1)
ax.plot(range(len(difference_x)), difference_x)
plt.show()
