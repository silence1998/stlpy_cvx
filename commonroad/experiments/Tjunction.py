import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.animation as animation

plt.rcParams.update({'figure.max_open_warning': 0})
from IPython.display import clear_output
import numpy as np
import sympy as sp
# import classes and functions for reading xml file and visualizing commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad_dc.feasibility.vehicle_dynamics import KinematicSingleTrackDynamics
from commonroad.common.solution import Solution, PlanningProblemSolution, \
    VehicleModel, VehicleType, CostFunction

from commonroad_stl import *
from Models.single_track import SingleTrack
from SCvx_solver import SCvxSolverFixTime
from visualization import plotTrajectory, plotVelocityAcceleration

# generate path of the file to be read
path_file = "./scenario/ZAM_Tjunction-1_288_T-1.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_file).open()



def getGoalSTL(planning_problem_dict: PlanningProblemSet, K, dynamic_model):
    ####TODO what does the number of planning problem means
    goal_formulas = []
    goal_patch = []
    for i, problem in enumerate(planning_problem_dict):
        initial_state = planning_problem_dict[problem].initial_state
        goal = planning_problem_dict[problem]._goal_region
        shapes = goal.state_list[0].position.shapes
        for j in range(1):
            position = goal.state_list[0].position.shapes[j].center
            tt = inside_polygon_deviation_formula(shapes[j].vertices[14:18, :], 0, 1, 4, 5, dynamic_model.parameters.b)
            goal_formulas.append(tt)
            goal_patch.append(shapes[j].vertices[14:18, :])
            tmp_pass = np.vstack((shapes[j].vertices[0:3, :], shapes[j].vertices[-2, :]))
            tt = inside_polygon_deviation_formula(tmp_pass, 0, 1, 4, 5, dynamic_model.parameters.b)
            goal_formulas.append(tt)
            goal_patch.append(tmp_pass)
            for ii in range(2, 14):
                tt = inside_polygon_formula(np.vstack((shapes[j].vertices[ii:ii+2, :], shapes[j].vertices[-ii-1:-ii+1, :])), 0, 1, 5)
                goal_formulas.append(tt)
    goal_stl = goal_formulas[0]
    for i in range(1, len(goal_formulas)):
        goal_stl = goal_stl | goal_formulas[i]
    return goal_stl, goal_patch


(x1, y1) = (18.7, 9.5)
(x2, y2) = (2.0, 0.0)


dynamic_model = KinematicSingleTrackDynamics(VehicleType.BMW_320i)
x_init = getInitialState(planning_problem_set._planning_problem_dict, dynamic_model)
K = 147
x_final = x_init
#x_final = np.array([x1, y1, 0, x_init[3], np.pi/2 + np.pi/12])
x_final = np.array([x2, y2, 0, x_init[3], 0])
iterations = 20
tr_radius = 20  ##### 40 for K=50
sigma = (K - 1) * 0.1

spec_2 = getDynamicObstaclesSTL(scenario.dynamic_obstacles, K, dynamic_model)
spec_1, goal_patch = getGoalSTL(planning_problem_set._planning_problem_dict, K, dynamic_model)
spec = spec_2.always(0, K) & spec_1.eventually(0, K)
# spec = spec_2.eventually(0, K)
m = SingleTrack(K,
                sigma,
                spec,
                max_k=20,
                smin_C=0.1,
                x_init=x_init,
                x_final=x_final, center_var_number=3)

m.settingWeights(u_weight=0.1, velocity_weight=0)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
X_converge = X
X_sub_converge = X_sub
U_converge = U

#np.save("X_converge_Tjunction.npy", X_converge)

plotTrajectory(X_converge, dynamic_model, K, scenario, planning_problem_set)
plotVelocityAcceleration(X_converge, U_converge)

result, solution = checkTrajectory(X_converge, dynamic_model, planning_problem_set, scenario)

print(result)