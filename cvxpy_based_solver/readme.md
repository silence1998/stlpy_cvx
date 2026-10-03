This folder provides experiments about SCvx on stlpy benchmarks.

"experiments" folder provides the running scripts (single runs, *_loop.py, time_test_*.py). remember K=T+1, T is time horizon.

"sweeps" folder provides the parameter sweep scripts that generate the heatmap data; new data is written to "outputs/heatmap/...".

"plots" folder provides the scripts for plotting heatmaps of different parameters and running-time results; by default they read "results/", pass "outputs" as first argument to plot newly generated data.

"results/heatmap" contains the committed heatmap data, "results/solution" the solution trajectories with given parameter settings, and "results/time_test" the running time with different time horizons.

"outputs" (created automatically, not committed) receives the output of sweeps and timing scripts.

"Discretization" folder is copy from https://github.com/EmbersArc/SCvx and used for system dynamic discretization.

"SCproblem" folder is changed based on https://github.com/EmbersArc/SCvx and used for transforming problem into cvxpy form.

"Models" folder provide different problems with STL formulas.
