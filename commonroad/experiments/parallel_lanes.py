import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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
from commonroad.common.solution import CommonRoadSolutionWriter
from commonroad_stl import *
from visualization import plotTrajectory, plotVelocityAcceleration


dir_output = "/home/chenmo/solutions"
filePath = "./scenario/parallel_lanes/"
for _, _, filenamelist in os.walk(filePath):
    print(filenamelist)

success_times = 0
solve_time_list = []
compile_time_list = []

for i_file in range(len(filenamelist)):
    try:
        # generate path of the file to be read
        path_file = filePath + filenamelist[i_file]
        # read in the scenario and planning problem set
        scenario, planning_problem_set = CommonRoadFileReader(path_file).open()


        lanelets = scenario._lanelet_network._lanelets
        


        from Models.single_track import SingleTrack
        from SCvx_solver import SCvxSolverFixTime

        dynamic_model = KinematicSingleTrackDynamics(VehicleType.BMW_320i)
        x_init = getInitialState(planning_problem_set._planning_problem_dict, dynamic_model)
        K, x_final = getFinalState(planning_problem_set._planning_problem_dict, dynamic_model)
        K = 31
        final_position_max, final_position_min = getFinalStateLimit(planning_problem_set._planning_problem_dict, dynamic_model)
        iterations = 20
        tr_radius = 20  ##### 40 for K=50
        sigma = (K - 1) * 0.1
        road, z_right, boundary_right, z_left, boundary_left = getBoundarySTL(lanelets)
        spec_1 = getStaticObstaclesSTL(scenario.static_obstacles, dynamic_model)
        spec_2 = getDynamicObstaclesSTL(scenario.dynamic_obstacles, K, dynamic_model)
        spec_3 = getGoalSTL(planning_problem_set._planning_problem_dict, dynamic_model)
        spec = road.always(0, K)
        if spec_1 is not None:
            spec = spec & spec_1.always(0, K)
        if spec_2 is not None:
            spec = spec & spec_2.always(0, K)
        if spec_3 is not None:
            spec = spec & spec_3.eventually(0, K)
        # spec = spec_2.eventually(0, K)
        m = SingleTrack(K,
                        sigma,
                        spec,
                        max_k=5,
                        smin_C=0.1,
                        x_init=x_init,
                        x_final=x_final, center_var_number=3)

        m.settingFinalPositionLimit(final_position_max, final_position_min)
        #m.settingFixFinalState()
        #m.settingFinalAngel([x_final[4]-0.1,x_final[4]+0.1])
        #m.settingStateBoundary(x_min=np.array([-50.0, -50.0, -5.0, -5.0]), x_max=np.array([200.0, 200.0, 5.0, 5.0]))
        #m.settingControlBoundary(u_min=np.array([-5, -5]), u_max=np.array([5, 5]))
        m.settingWeights(u_weight=0.1, velocity_weight=0)
        solver = SCvxSolverFixTime(m, K, iterations, sigma, tr_radius)

        X, X_sub, X_robust, U, X_init, all_X, all_X_sub = solver.solve()
        X_converge = X
        X_sub_converge = X_sub
        U_converge = U
        solve_time_list.append(solver.solving_time)
        compile_time_list.append(solver.compile_time)

        plotTrajectory(X_converge, dynamic_model, K, scenario, planning_problem_set,
               [z_right, boundary_right, z_left, boundary_left], X_init=X_init)

        result, solution = checkTrajectory(X_converge, dynamic_model, planning_problem_set, scenario)
        print(result)
        if result[0] == True:
            success_times += 1
            # create directory if not exists
            if not os.path.exists(os.path.dirname(dir_output)):
                os.makedirs(dir_output, exist_ok=True)
            # write solution to a CommonRoad XML file
            csw = CommonRoadSolutionWriter(solution)
            csw.write_to_file(output_path=dir_output, overwrite=True)
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print('\n'+message)
        continue
    
print('success_rate=', success_times/len(filenamelist))
print('mean solve time=', sum(solve_time_list)/len(solve_time_list))
print('mean compile time=', sum(compile_time_list)/len(compile_time_list))

