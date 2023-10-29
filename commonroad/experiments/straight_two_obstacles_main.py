import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

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
from visualization import plotTrajectory, plotVelocityAcceleration


# generate path of the file to be read
path_file = "./scenario/ZAM_Urban-3_3_Repair copy.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_file).open()


lanelets = scenario._lanelet_network._lanelets



from Models.single_track import SingleTrack
from SCvx_solver import SCvxSolverFixTime

dynamic_model = KinematicSingleTrackDynamics(VehicleType.BMW_320i)
x_init = getInitialState(planning_problem_set._planning_problem_dict, dynamic_model)
K, x_final = getFinalState(planning_problem_set._planning_problem_dict, dynamic_model)
K = 29
iterations = 20
tr_radius = 20  ##### 40 for K=50
sigma = (K - 1) * 0.1
road, z_right, boundary_right, z_left, boundary_left = getBoundarySTL(lanelets)
spec_1 = getStaticObstaclesSTL(scenario.static_obstacles, dynamic_model)
spec_2 = getGoalSTL(planning_problem_set._planning_problem_dict, dynamic_model)
spec = road.always(0, K) & spec_1.always(0, K) & spec_2.eventually(0, K)
# spec = spec_2.eventually(0, K)
m = SingleTrack(K,
                sigma,
                spec,
                max_k=5,
                smin_C=0.1,
                x_init=x_init,
                x_final=x_final)


m.settingWeights(u_weight=1e-5, velocity_weight=0)
solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
X_converge = X
X_sub_converge = X_sub
U_converge = U

plotVelocityAcceleration(X_converge, U_converge)

plt.figure(figsize=(25, 10))

renderer = MPRenderer()

# uncomment the following line to visualize with animation
clear_output(wait=True)

scenario.draw(renderer)

# plot the planning problem set

planning_problem_set.draw(renderer)

renderer.render()

yvals_r = np.polyval(z_right, boundary_right[:,0]) 
plt.plot(boundary_right[:,0], yvals_r, 'r', label='fit', zorder=6000)
yvals_l = np.polyval(z_left, boundary_left[:,0]) 
plt.plot(boundary_left[:,0], yvals_l, 'r', label='fit', zorder=6000)


X_center = getCenterTrajectory(X_converge, dynamic_model)
plot_x_init = getCenterTrajectory(X_init, dynamic_model)
plt.scatter(X_center[0, :], X_center[1, :], zorder=10000)
plt.scatter(plot_x_init[0, :], plot_x_init[1, :], color='green', zorder=10000)

radius_car = math.sqrt(Length**2/(6**2)+Width**2/4)
deviation_car = 2*math.sqrt(radius_car**2-Width**2/4)

for i in range(len(X_converge[0, :])):
    angel = X_converge[4, i]
    x = X_converge[
        0, i] - (Length / 2 - dynamic_model.parameters.b) * math.cos(angel) + Width / 2 * math.sin(angel)
    y = X_converge[
        1, i] - (Length / 2 - dynamic_model.parameters.b) * math.sin(angel) - Width / 2 * math.cos(angel)
    renderer.ax.add_patch(
        Rectangle((x, y),
                  Length,
                  Width,
                  angel / math.pi * 180,
                  color='green',
                  alpha=0.3,
                  zorder=8000))
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()

#print(checkTrajectory(X_converge, dynamic_model, planning_problem_set, scenario))