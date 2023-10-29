import os
import math
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
from IPython.display import clear_output
import numpy as np
# import classes and functions for reading xml file and visualizing commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad_dc.feasibility.vehicle_dynamics import KinematicSingleTrackDynamics
from commonroad.common.solution import Solution, PlanningProblemSolution, \
    VehicleModel, VehicleType, CostFunction
from commonroad_stl import *
from Models.single_track import SingleTrack
from SCvx_solver import SCvxSolverFixTime
from visualization import plotTrajectory, plotVelocityAcceleration

# generate path of the file to be read
path_file = "./scenario/DEU_Gar-1_1_T-1.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_file).open()

# plot the scenario for 40 time steps, here each time step corresponds to 0.1 second

lanelets = scenario._lanelet_network._lanelets
boundary_left = lanelets[47240].left_vertices
self_boundary_right = lanelets[47240].right_vertices
boundary_right = lanelets[47238].right_vertices
# boundary_right = lanelets[47240].right_vertices
z_right = np.polyfit(boundary_right[:, 0], boundary_right[:, 1], 3)
self_z_right = np.polyfit(self_boundary_right[:, 0], self_boundary_right[:, 1], 3)
z_lef = np.polyfit(boundary_left[:, 0], boundary_left[:, 1], 3)
right_formula = upon_poly_formula(z_right, 0, 1, 5, deviation=[0, -2*Width/3])
lef_formula = upon_poly_formula(z_lef, 0, 1, 5, deviation=[0, 2*Width/3])
lef_formula = lef_formula.negation()
road = right_formula & lef_formula


dynamic_model = KinematicSingleTrackDynamics(VehicleType.BMW_320i)
x_init = getInitialState(planning_problem_set._planning_problem_dict, dynamic_model)
K, x_final = getFinalState(planning_problem_set._planning_problem_dict, dynamic_model)
iterations = 20
tr_radius = 20  ##### 40 for K=50
sigma = (K - 1) * 0.1

spec_1 = getDynamicObstaclesSTL(scenario.dynamic_obstacles, K, dynamic_model)
spec_2 = getInSameLaneAndFrontSTL(x_init, scenario.dynamic_obstacles, K, z_lef, self_z_right, dynamic_model)
spec = (road).always(0, K) & spec_1.always(0, K) & spec_2.always(0, K)
#spec = (road).always(0, K) & spec_2.always(0, K)

m = SingleTrack(K,
                sigma,
                spec,
                max_k=10,
                smin_C=0.1,
                x_init=x_init,
                x_final=(2*x_init+3*x_final)/5, center_var_number=3)

# m.settingStateBoundary(x_min=np.array([-np.inf, -np.inf, -np.inf, 0.0, -np.inf]),
#                        x_max=np.array([np.inf, np.inf, np.inf, 50.0, np.inf]))
m.settingWeights(u_weight=0, velocity_weight=0)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
X_converge = X
X_sub_converge = X_sub
U_converge = U

plotTrajectory(X_converge, dynamic_model, K, scenario, planning_problem_set,
               [z_right, boundary_right, z_lef, boundary_left])
plotVelocityAcceleration(X_converge, U_converge)


