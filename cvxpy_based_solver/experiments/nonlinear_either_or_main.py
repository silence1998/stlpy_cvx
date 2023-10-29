from time import time
import numpy as np

from Discretization import FirstOrderHoldFixTime
from SCproblem import FixTimeSCProblem
from SCproblem import FixTimeSubDynamicSCProblem
from utils import format_line, save_arrays

# from Models.stlpy_unicycle import Unicycle
from Models.nonlinear_either_or import DoubleIntegral

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import stlpy.STL

from SCvx_solver import SCvxSolverFixTime
t_total_1 = time()
K = 26
iterations = 50

# Weight constants
w_nu = 1e5  # virtual control
# initial trust region radius
tr_radius = 50
# trust region variables

sigma = 2.5

m = DoubleIntegral(K, sigma)
#solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius, w_nu, rho_1, rho_1, alpha, beta)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)
X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
print('Solve time: ', solver.solving_time, 's')
print('Compile time: ', solver.compile_time, 's')
print('final robustness:%f', X_robust[-1, 0])
X_converge = X
X_sub_converge = X_sub
U_converge = U
f = plt.gca()
f.set_aspect('equal')
m.add_to_plot(f)
f.scatter(X_converge[0, :], X_converge[1, :], zorder=10)
f.scatter(X_init[0, :], X_init[1, :], color='green')
plt.show()

# difference_x = []
# for i in range(len(all_X) - 2):
#     difference_x.append(np.linalg.norm(all_X[i] - X_converge, 1) + np.linalg.norm(all_X_sub[i] - X_sub_converge, 1))
#
# difference_x = np.array(difference_x)
# print(difference_x)
# f, ax = plt.subplots()
# ax.set_yscale("log")
# ax.set_ylim(1e-4, 2e1)
# ax.plot(range(len(difference_x)), difference_x)
# plt.show()
