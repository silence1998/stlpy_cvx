import matplotlib.pyplot as plt
import numpy as np

time_step = np.load("time_step.npy")
number_of_variable = np.load("number_of_variable.npy")
total_solve_time_list = np.load("total_solve_time_list.npy")
total_compile_time_list = np.load("total_compile_time_list.npy")
step_solve_time_list = np.load("step_solve_time_list.npy")
step_compile_time_list = np.load("step_compile_time_list.npy")

plt.plot(time_step, step_solve_time_list)
plt.xlabel('time step')
plt.ylabel('average solve time')
plt.show()

plt.plot(time_step, step_compile_time_list)
plt.xlabel('time step')
plt.ylabel('average compile time')
plt.show()

plt.plot(number_of_variable, total_solve_time_list)
plt.xlabel('number of variable')
plt.ylabel('average solve time')
plt.show()

plt.plot(number_of_variable, total_compile_time_list)
plt.xlabel('number of variable')
plt.ylabel('average compile time')
plt.show()

