import os
import sys

# 数据目录：默认读取仓库中已提交的 results/ 数据。
# 若要画自己运行 sweep/计时脚本后得到的新数据，把 outputs 目录作为第一个参数传入，例如：
#   python plots/plot_time_multitask.py outputs
# （参数先按当前目录解析，不存在时再按本子项目根目录解析。）
# 注意：本图需要 <根>/time_test/ 下的 MICP_*、SOS1_*、SCvx_* 子目录。stl_time_test_*.py 会写到
#   outputs/time_test/stlpy_either_or|stlpy_multitask/，请按所用求解器改名为 MICP_* 或 SOS1_*；
#   SCvx_* 数据来自 cvxpy_based_solver/outputs/time_test/（需复制过来）。
_PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if len(sys.argv) > 1:
    _ROOT = sys.argv[1] if os.path.isdir(sys.argv[1]) else os.path.join(_PROJ_DIR, sys.argv[1])
else:
    _ROOT = os.path.join(_PROJ_DIR, "results")
DATA_DIR = os.path.join(_ROOT, "time_test")

import matplotlib.pyplot as plt
import numpy as np

time_step_1 = np.load(os.path.join(DATA_DIR, "MICP_multitask/time_step.npy"))
number_of_variable_1 = np.load(os.path.join(DATA_DIR, "MICP_multitask/number_of_variable.npy"))
total_solve_time_list_1 = np.load(os.path.join(DATA_DIR, "MICP_multitask/total_solve_time_list.npy"))

time_step_2 = np.load(os.path.join(DATA_DIR, "SCvx_Multitask/time_step.npy"))
total_solve_time_list_2 = np.load(os.path.join(DATA_DIR, "SCvx_Multitask/total_solve_time_list.npy"))
total_compile_time_list_2 = np.load(os.path.join(DATA_DIR, "SCvx_Multitask/total_compile_time_list.npy"))

time_step_3 = np.load(os.path.join(DATA_DIR, "SOS1_multitask/time_step.npy"))
total_solve_time_list_3 = np.load(os.path.join(DATA_DIR, "SOS1_multitask/total_solve_time_list.npy"))

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
