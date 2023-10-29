import os
import math
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
from IPython.display import clear_output
import numpy as np
import sympy as sp
# import classes and functions for reading xml file and visualizing commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad_dc.feasibility.vehicle_dynamics import KinematicSingleTrackDynamics
from commonroad.common.solution import Solution, PlanningProblemSolution, \
    VehicleModel, VehicleType, CostFunction
from commonroad_stl import *
from visualization import plotTrajectory, plotVelocityAcceleration


# generate path of the file to be read
path_file = "./scenario/ZAM_Tutorial-1_2_T-2.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_file).open()


lanelets = scenario._lanelet_network._lanelets



from Models.single_track import SingleTrack
from SCvx_solver import SCvxSolverFixTime

dynamic_model = KinematicSingleTrackDynamics(VehicleType.BMW_320i)
x_init = getInitialState(planning_problem_set._planning_problem_dict, dynamic_model)
K = 30
x_final = x_init+np.array([80, 0, 0, 0, 0])
iterations = 20
tr_radius = 20  ##### 40 for K=50
sigma = (K - 1) * 0.1
road, z_right, boundary_right, z_left, boundary_left = getBoundarySTL(lanelets)
spec_1 = getStaticObstaclesSTL(scenario.static_obstacles, dynamic_model)
spec_2 = getDynamicObstaclesSTL(scenario.dynamic_obstacles, K, dynamic_model)
spec = road.always(0, K) & spec_1.always(0, K) & spec_2.always(0, K)
# spec = spec_2.eventually(0, K)
m = SingleTrack(K,
                sigma,
                spec,
                max_k=5,
                smin_C=0.1,
                x_init=x_init,
                x_final=x_final, center_var_number=3)

m.settingWeights(u_weight=0.1, velocity_weight=0)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
X_converge = X
X_sub_converge = X_sub
U_converge = U

plotTrajectory(X_converge, dynamic_model, K, scenario, planning_problem_set,
               [z_right, boundary_right, z_left, boundary_left])
plotVelocityAcceleration(X_converge, U_converge)

    
