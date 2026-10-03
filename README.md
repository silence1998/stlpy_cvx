# scvx-stl / stlpy_cvx

**English summary.** This repository plans trajectories under **Signal Temporal Logic (STL)** specifications with
**Successive Convexification (SCvx)**. STL formulas are built with [stlpy](https://github.com/vincekurtz/stlpy) and turned into extra dynamics: smoothed min/max "sub-states" plus a robustness state. Each SCvx iteration discretises the dynamics with first-order hold, linearises them and solves a convex subproblem with **cvxpy + ECOS** inside a trust region.
`cvxpy_based_solver/` holds the solver and the stlpy benchmarks, and runs on Windows/Linux/macOS.
`stlpy_comparison_experiments/` holds baselines with Drake, which needs Linux/macOS/WSL.
`commonroad/` applies the method to CommonRoad scenarios, which needs `commonroad-drivability-checker` (Linux/macOS).
Setup: `conda env create -f environment.yml`, then `conda activate stlpy_cvx`. The discretisation and SCproblem code is adapted from [EmbersArc/SCvx](https://github.com/EmbersArc/SCvx).

---

## 1. 项目简介

本项目用 **SCvx（Successive Convexification，逐次凸化）** 求解带 **STL（Signal Temporal Logic，信号时序逻辑）** 约束的轨迹规划问题：

- 用 [stlpy](https://github.com/vincekurtz/stlpy) 构造 STL 公式，再通过平滑 min/max 子状态（sub-states）和鲁棒度状态，把公式变成附加的动力学约束；
- 每次迭代用一阶保持（FOH）离散化并线性化，得到一个凸子问题，交给 **cvxpy + ECOS** 求解；
- 用信赖域（trust region）参数 `rho_1, rho_2, alpha, beta` 和虚拟控制权重 `w_nu` 控制收敛。

论文/报告与答辩幻灯片见 `docs/`。

三个子项目相互独立，都没有 setup.py / pyproject，不是可安装的包，各自靠 `PYTHONPATH` 导入：

| 子项目 | 内容 | Windows 原生 |
|---|---|---|
| `cvxpy_based_solver/` | SCvx 求解器 + stlpy 基准（either-or、multitask、单车模型、unicycle 等）、参数扫描、耗时测试 | ✅ 可以运行 |
| `stlpy_comparison_experiments/` | 与 stlpy 自带求解器（Drake MICP / SOS1 / 平滑 NLP）的对比 | ❌ 需要 pydrake |
| `commonroad/` | CommonRoad 场景上的 SCvx（运动学单车模型） | ❌ 需要 commonroad_dc |

## 2. 目录结构

```
stlpy_cvx/
├── README.md               本文件（README.old.md 为原英文 README 的备份）
├── environment.yml         conda 环境（实测可用）
├── docs/                   SA.pdf（报告）、SA_presentation_Chen_Mo.pptx（幻灯片）
├── cvxpy_based_solver/
│   ├── SCvx_solver.py      SCvxSolverFixTime：SCvx 主循环（solver='ECOS'）
│   ├── utils.py
│   ├── Discretization/     FOH 离散化（来自 EmbersArc/SCvx）
│   ├── SCproblem/          cvxpy 参数化子问题（基于 EmbersArc/SCvx 修改）
│   ├── Models/             各系统 + STL 模型（主力：double_integral_with_sub_state.py）
│   ├── experiments/        可直接运行的实验：单次实验、*_loop.py（50 个随机种子）、time_test_*.py（耗时测试）
│   ├── sweeps/             参数扫描（热力图数据）：{either_or,multitask,no_sub_states}/sweep_{alpha_beta,rho_1_rho_2,w_nu_trust_region}.py
│   ├── plots/              画图脚本：heatmap/<任务>/plot_<参数>[_trajectory].py，time_test/plot_*.py
│   ├── results/            已提交的结果数据（只读）：heatmap/、solution/、time_test/
│   └── outputs/            （运行时自动创建）sweep 与耗时测试的新输出，结构与 results/ 相同
├── stlpy_comparison_experiments/
│   ├── *.py                对比实验（Drake），mylinear.py / myScipyGradientSolver.py 等辅助模块
│   ├── simple_moving_obstacle/   stlpy 用于动态障碍物的尝试
│   ├── plots/              plot_time_either_or.py、plot_time_multitask.py
│   ├── results/time_test/  MICP_* / SOS1_* / SCvx_* 耗时数据
│   └── outputs/            （运行时自动创建）stl_time_test_*.py 的输出
└── commonroad/
    ├── commonroad_stl.py、visualization.py、SCvx_solver.py、Models/ …（与 cvxpy_based_solver 结构相似）
    ├── experiments/        CommonRoad 实验
    ├── scenario/           CommonRoad 场景 xml
    └── solutions/          CommonRoad 解文件
```

> `outputs/` 不在版本控制中（仓库没有 .gitignore，运行后会以未跟踪文件出现），请不要把它提交上去；确认要更新结果时再手动复制到 `results/`。

## 3. 安装环境

```powershell
cd E:\code\stlpy_cvx
conda env create -f environment.yml      # 创建 stlpy_cvx 环境：Python 3.11，conda-forge + pip(stlpy 0.3.0)
conda activate stlpy_cvx
```

⚠️ **一定要先 `conda activate stlpy_cvx` 再运行。** 本机 PATH 中有 base Anaconda 的 `Library\bin`（含 MKL DLL）。如果不激活环境、直接调用 `...\envs\stlpy_cvx\python.exe`，`numpy.linalg` 会直接崩溃（退出码 0xC06D007F）。激活后环境自己的 DLL 目录会排在 PATH 最前面，就能正常运行。

- cvxpy ≥ 1.6 不再自带 ECOS，而代码中写死了 `solver='ECOS'`，所以 `environment.yml` 单独装了 `ecos`。
- 运行 `cvxpy_based_solver/` 不需要任何商业求解器许可证。

## 4. 运行（cvxpy_based_solver）

```powershell
conda activate stlpy_cvx
cd E:\code\stlpy_cvx\cvxpy_based_solver
$env:PYTHONPATH = "$PWD;$PWD\experiments"   # 让脚本找到 Models/SCproblem/SCvx_solver/base_task 等
$env:MPLBACKEND = "Agg"                       # 可选：无界面运行，不弹出图窗
```

### 4.1 单次实验（`experiments/`）

参考耗时为 2026-10 在本机实测：

| 命令 | 内容 | 耗时 |
|---|---|---|
| `python experiments\either_or_main.py` | either-or 任务（双积分器） | ~12 s |
| `python experiments\either_or_main_single_track.py` | either-or，单车模型 | ~5 s |
| `python experiments\multitask.py` | 多目标任务 | ~22 s |
| `python experiments\nonlinear_either_or_main.py` | 非线性谓词 either-or | ~8 s |
| `python experiments\nonlinear_reach_avoid.py` | 非线性 reach-avoid | ~3 s |
| `python experiments\two_obstacles_main.py` | 两个障碍物 | ~5 s |
| `python experiments\fixedtime_doubleintegral.py` | 固定终端时间双积分器（无子状态） | ~2 s |
| `python experiments\freetime_doubleintegral.py` | 自由终端时间双积分器 | ~2 s |
| `python experiments\freetime_unicycle.py` | 自由终端时间 unicycle | ~5 s |

### 4.2 循环与耗时测试（耗时长）

| 命令 | 内容 | 耗时 | 输出 |
|---|---|---|---|
| `python experiments\nonlinear_multitask_loop.py` | 50 个随机种子，统计成功率 | ~9 min | 只打印 |
| `python experiments\multitask_loop.py` | 50 个随机种子 | > 15 min | 只打印 |
| `python experiments\time_test_either_or.py` | K = 16…81 逐个计时 | 数十分钟以上 | `outputs/time_test/either_or/*.npy` |
| `python experiments\time_test_multitask.py` | K = 15…150 逐个计时 | 数小时 | `outputs/time_test/multitask/*.npy`（每 5 个 K 保存一次） |

### 4.3 参数扫描（`sweeps/`，生成热力图数据）

```powershell
python sweeps\either_or\sweep_alpha_beta.py        # 7×7 网格，信赖域缩放 alpha/beta
python sweeps\either_or\sweep_rho_1_rho_2.py       # 8×8 网格，rho_1/rho_2
python sweeps\either_or\sweep_w_nu_trust_region.py # 9×9 网格，w_nu × 初始信赖域半径
# multitask/ 和 no_sub_states/ 下有同名的三个脚本
```

每个网格点都要跑一次完整的 SCvx，单个脚本需要几分钟到几十分钟（multitask 最慢）。结果写到 `outputs/heatmap/<任务>/<参数>/`，文件名与 `results/` 中一致，不会覆盖已提交的数据。

### 4.4 画图（`plots/`）

默认读取 `results/` 中已提交的数据，可以在任意目录运行：

```powershell
python plots\heatmap\either_or\plot_alpha_beta.py
python plots\heatmap\either_or\plot_alpha_beta_trajectory.py
python plots\heatmap\multitask\plot_w_nu_trust_region.py
python plots\time_test\plot_either_or_vs_multitask.py
python plots\time_test\plot_either_or.py
python plots\time_test\plot_multitask.py
```

如果要画自己用 sweep 或耗时测试新跑出的数据，把 `outputs` 作为参数传入（相对路径先按当前目录解析，找不到再按 `cvxpy_based_solver/` 解析）：

```powershell
python plots\heatmap\either_or\plot_alpha_beta.py outputs
```

**LaTeX 说明：** `plots/heatmap/either_or|multitask/plot_{alpha_beta,rho_1_rho_2,w_nu_trust_region}.py` 设置了 `plt.rcParams['text.usetex'] = True`，需要本机装 LaTeX（MiKTeX 或 TeX Live，且 `latex` 在 PATH 上），否则会报
`RuntimeError: Failed to process string with tex because latex could not be found`。
不想装 LaTeX 时，把该行改成 `False` 就能出图（坐标轴上的 `$\alpha$` 等由 matplotlib mathtext 渲染）。其余画图脚本不需要 LaTeX。

## 5. 无法在 Windows 原生运行的部分

- **`stlpy_comparison_experiments/`**：需要 `pydrake`。Drake 官方只提供 Linux 和 macOS 版本（pip 上的 `drake` 没有 Windows wheel）。Drake 的 MICP 基线还需要 **Gurobi 或 MOSEK**，两者都是商业软件，有免费学术许可证。可以在 WSL2 / Linux 上运行：`pip install drake stlpy`。仅依赖 stlpy 的辅助模块（`mylinear.py`、`myScipyGradientSolver.py`、`simple_moving_obstacle/linear_predicate_with_velocity.py`）可以在 Windows 上导入。`plots/plot_time_*.py` 也能在 Windows 上运行，因为它们只读 `results/time_test/`。
- **`commonroad/`**：需要 `commonroad-drivability-checker`（`commonroad_dc`）。它在 PyPI 上只有源码包，包内含带冒号 `:` 的文件名，Windows 无法解包，而且编译需要 C++ 工具链。`visualization.py` 另外还用到 `commonroad-crime`。`environment.yml` 已装 `commonroad-io`，但单靠它跑不起来。请在 Linux/WSL2 上运行：`pip install commonroad-io commonroad-drivability-checker commonroad-crime`，然后在 `commonroad/` 目录下执行 `python experiments/<脚本>.py`（脚本用 `./scenario/...` 相对路径读取场景）。

## 6. 致谢

`Discretization/` 拷贝自 [EmbersArc/SCvx](https://github.com/EmbersArc/SCvx)，`SCproblem/` 在其基础上修改而来。
STL 公式、基准场景和对比求解器来自 [stlpy](https://github.com/vincekurtz/stlpy)（Vince Kurtz）。CommonRoad 场景来自 [CommonRoad](https://commonroad.in.tum.de/)。
