import os
import sys

# 数据目录：默认读取仓库中已提交的 results/ 数据。
# 若要画自己运行 sweep/计时脚本后得到的新数据，把 outputs 目录作为第一个参数传入，例如：
#   python plots/time_test/plot_multitask.py outputs
# （参数先按当前目录解析，不存在时再按本子项目根目录解析。）
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if len(sys.argv) > 1:
    _ROOT = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.join(_PROJ_DIR, sys.argv[1])
else:
    _ROOT = os.path.join(_PROJ_DIR, "results")
DATA_DIR = os.path.join(_ROOT, "time_test", "multitask")



import matplotlib.pyplot as plt
import numpy as np

time_step = np.load(os.path.join(DATA_DIR, "time_step.npy"))
number_of_variable = np.load(os.path.join(DATA_DIR, "number_of_variable.npy"))
total_solve_time_list = np.load(os.path.join(DATA_DIR, "total_solve_time_list.npy"))
total_compile_time_list = np.load(os.path.join(DATA_DIR, "total_compile_time_list.npy"))
step_solve_time_list = np.load(os.path.join(DATA_DIR, "step_solve_time_list.npy"))
step_compile_time_list = np.load(os.path.join(DATA_DIR, "step_compile_time_list.npy"))

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