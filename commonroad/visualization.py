import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from commonroad.common.solution import Solution, PlanningProblemSolution, \
    VehicleModel, VehicleType, CostFunction
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_crime.utility.visualization import draw_state_list, TUMcolor, draw_dyn_vehicle_shape
from commonroad_stl import getCenterTrajectory
from IPython.display import clear_output

def plotTrajectory(X_converge, dynamic_model, K, scenario, planning_problem_set, boundary=None, X_init=None):
    Length = 4.508
    Width = 1.610
    f = plt.figure()
    renderer = MPRenderer()

    # uncomment the following line to visualize with animation
    clear_output(wait=True)

    # plot the scenario for each time step
    renderer.draw_params.time_begin = 0
    scenario.draw(renderer)

    # plot the planning problem set
    planning_problem_set.draw(renderer)

    renderer.render()

    if boundary is not None:
        z_right = boundary[0]
        boundary_right = boundary[1]
        z_lef = boundary[2]
        boundary_left = boundary[3]
        yvals_r = np.polyval(z_right, boundary_right[:,0]) 
        plt.plot(boundary_right[:,0], yvals_r, 'r', label='fit', zorder=6000)
        yvals_l = np.polyval(z_lef, boundary_left[:,0]) 
        plt.plot(boundary_left[:,0], yvals_l, 'r', label='fit', zorder=6000)

    if X_init is not None:
        plot_x_init = getCenterTrajectory(X_init, dynamic_model)
        plt.scatter(plot_x_init[0, :], plot_x_init[1, :], color=TUMcolor.TUMgreen, zorder=10000)

    state_list = []
    for i in range(X_converge.shape[1]):
        state_list.append(dynamic_model._array_to_state(X_converge[:, i], i))


    for obs in scenario.dynamic_obstacles:
        draw_state_list(renderer, obs.prediction.trajectory.state_list[0:K],
                        color=TUMcolor.TUMdarkred, linewidth=2)
        length = obs.obstacle_shape.length
        width = obs.obstacle_shape.width
        for i in range(0, K):
            if i == 0:
                state = obs.initial_state
            else:
                state = obs.prediction.trajectory.state_list[i - 1]
            angle = state.orientation
            x = state.position[0] -\
                 (length / 2) * math.cos(angle) + width / 2 * math.sin(angle)
            y = state.position[1] -\
                 (length / 2) * math.sin(angle) - width / 2 * math.cos(angle)
            renderer.ax.add_patch(
                Rectangle((x, y),
                        length,
                        width,
                        angle / math.pi * 180,
                        color=TUMcolor.TUMdarkred,
                        alpha=0.3,
                        zorder=7000))
        # for ts in range(1, K):
        #     draw_dyn_vehicle_shape(renderer, obs, ts, color=TUMcolor.TUMblue)
    draw_state_list(renderer, state_list, color=TUMcolor.TUMblue, linewidth=3)
    for i in range(0, K):
        angle = X_converge[4, i]
        x = X_converge[
            0, i] - (Length / 2 - dynamic_model.parameters.b) * math.cos(angle) + Width / 2 * math.sin(angle)
        y = X_converge[
            1, i] - (Length / 2 - dynamic_model.parameters.b) * math.sin(angle) - Width / 2 * math.cos(angle)
        renderer.ax.add_patch(
            Rectangle((x, y),
                    Length,
                    Width,
                    angle / math.pi * 180,
                    color='green',
                    alpha=0.3,
                    zorder=8000))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()


def plotVelocityAcceleration(X_converge, U_converge, delta_t=0.1):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 21}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    fig, axs = plt.subplots(4)
    K = X_converge.shape[1]
    x = np.linspace(0, K-1, K)*delta_t
    axs[0].plot(x, X_converge[3], 'tab:blue')
    axs[0].set_xlabel(r'$t[s]$')
    axs[0].set_ylabel(r'$v[m/s]$')
    axs[1].plot(x, X_converge[4], 'tab:orange')
    axs[1].set_xlabel(r'$t[s]$')
    axs[1].set_ylabel(r'$\Psi[rad]$')
    axs[2].plot(x, U_converge[0], 'tab:green')
    axs[2].set_xlabel(r'$t[s]$')
    axs[2].set_ylabel(r'$v_{\delta}[rad/s]$')
    axs[3].plot(x, U_converge[1], 'tab:red')
    axs[3].set_xlabel(r'$t[s]$')
    axs[3].set_ylabel(r'$a_{long}[m/s^2]$')
    plt.show()
