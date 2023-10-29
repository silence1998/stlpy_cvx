import matplotlib.pyplot as plt
import numpy as np

time_step_1 = np.load("either_or/time_step.npy")
number_of_variable_1 = np.load("either_or/number_of_variable.npy")
total_solve_time_list_1 = np.load("either_or/total_solve_time_list.npy")
total_compile_time_list_1 = np.load("either_or/total_compile_time_list.npy")
step_solve_time_list_1 = np.load("either_or/step_solve_time_list.npy")
step_compile_time_list_1 = np.load("either_or/step_compile_time_list.npy")

time_step_2 = np.load("multitask/time_step.npy")
number_of_variable_2 = np.load("multitask/number_of_variable.npy")
total_solve_time_list_2 = np.load("multitask/total_solve_time_list.npy")
total_compile_time_list_2 = np.load("multitask/total_compile_time_list.npy")
step_solve_time_list_2 = np.load("multitask/step_solve_time_list.npy")
step_compile_time_list_2 = np.load("multitask/step_compile_time_list.npy")

f, ax = plt.subplots()
plt.plot(number_of_variable_1, total_solve_time_list_1, label='either_or')
plt.plot(number_of_variable_2, total_solve_time_list_2, label='multitask')
plt.legend()
plt.xlabel('number of variable')
plt.ylabel('total solve time')
#ax.set_xscale("log")
#ax.set_yscale("log")
plt.show()

f, ax = plt.subplots()
plt.plot(number_of_variable_1, total_compile_time_list_1, label='either_or')
plt.plot(number_of_variable_2, total_compile_time_list_2, label='multitask')
plt.legend()
plt.xlabel('number of variable')
plt.ylabel('total compile time')
#ax.set_xscale("log")
#ax.set_yscale("log")
plt.show()

x_n = np.hstack((number_of_variable_1, number_of_variable_2))
y_n = np.hstack((total_solve_time_list_1, total_solve_time_list_2))

A = np.vstack([x_n, np.ones(len(x_n))]).T
result_1 = np.linalg.lstsq(A, y_n, rcond=None)
m, c = result_1[0]
print("1: sigma=", result_1[1]/len(x_n))
yfit = m*x_n + c
dyfit = np.sqrt(result_1[1]/len(x_n))
_ = plt.plot(x_n, y_n, 'o', label='Original data', markersize=10)
_ = plt.plot(x_n, m*x_n + c, 'r', label='Fitted line')
_ = plt.fill_between(x_n, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.6)
plt.xlabel('number of variable')
plt.ylabel('total solve time')
_ = plt.legend()
plt.show()

A = np.vstack([np.power(x_n, 3.5), np.ones(len(x_n))]).T
result_2 = np.linalg.lstsq(A, y_n, rcond=None)
m, c = result_2[0]
print("2: sigma=", result_2[1]/len(x_n))
yfit = m*np.power(x_n, 3.5) + c
dyfit = np.sqrt(result_2[1]/len(x_n))
_ = plt.plot(x_n, y_n, 'o', label='Original data', markersize=10)
_ = plt.plot(x_n, yfit, 'r', label='Fitted line')
_ = plt.fill_between(x_n, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.6)
plt.xlabel('number of variable')
plt.ylabel('total solve time')
_ = plt.legend()
plt.show()



