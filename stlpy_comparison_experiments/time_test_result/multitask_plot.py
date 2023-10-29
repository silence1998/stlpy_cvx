import matplotlib.pyplot as plt
import numpy as np

time_step_1 = np.load("time_test_result/MICP_multitask/time_step.npy")
number_of_variable_1 = np.load("time_test_result/MICP_multitask/number_of_variable.npy")
total_solve_time_list_1 = np.load("time_test_result/MICP_multitask/total_solve_time_list.npy")

time_step_2 = np.load("time_test_result/SCvx_Multitask/time_step.npy")
total_solve_time_list_2 = np.load("time_test_result/SCvx_Multitask/total_solve_time_list.npy")
total_compile_time_list_2 = np.load("time_test_result/SCvx_Multitask/total_compile_time_list.npy")

time_step_3 = np.load("time_test_result/SOS1_multitask/time_step.npy")
total_solve_time_list_3 = np.load("time_test_result/SOS1_multitask/total_solve_time_list.npy")

f, ax = plt.subplots()
plt.plot(time_step_1, total_solve_time_list_1, label='MICP solve time')
plt.plot(time_step_2[1:22] - 1, total_solve_time_list_2[1:22], label='SCvx solve time')
plt.plot(time_step_2[1:22] - 1, total_compile_time_list_2[1:22], label='SCvx compile time')
plt.plot(time_step_3, total_solve_time_list_3, label='SOS1 solve time')
plt.title('Multitask', fontsize=16)
plt.legend()
plt.xlabel('time horizon', fontsize=16)
plt.ylabel('solve time/compile time [s]', fontsize=16)
#ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
