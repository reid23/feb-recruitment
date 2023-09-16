#%%
from casadi import *
from track import get_test_track, offset_path
import numpy as np
from numpy.linalg import *
from numpy.random import random
import matplotlib.pyplot as plt
from time import perf_counter
path = get_test_track()
path = np.array([path[i] for i in range(path.shape[0]) if i%2==0])
left = offset_path(path, 4)
right = offset_path(path, -4)
left = np.array([left[i] for i in range(path.shape[0]) if i%3==0])
right = np.array([right[i] for i in range(path.shape[0]) if i%3==0])

#%%
vision_range=10

# start with x0 since we want to fix x0
x = [MX.sym('x0', 2)]  # MX.sym(2) for _ in range(path.shape[0])
lm = [] # MX.sym(2) for _ in range(something)
lm_guesses = np.array([]) # maps guess_location: index in lm
x_guesses = [path[0]]
x_cons = []  # (x[i]+odo measurement - x[i+1])**2 = 0
lm_cons = [] # (x[i]+lm measurement - lm[j])**2 = 0

def update_graph(dx, z):
    global lm_guesses, lm, lm_cons
    global x_guesses, x, x_cons
    x_guesses.append(x_guesses[-1]+dx)
    curpos = x_guesses[-1]
    x.append(MX.sym(f'x{len(x)}', 2))
    x_cons.append((x[-2]+DM(dx)-x[-1]))
    if len(lm_guesses)==0:
        lm_guesses = z+curpos
        # print(lm_guesses.shape)
        lm = [MX.sym(f'lm{i}', 2) for i in range(len(z))]
        lm_cons = [x[-1] + DM(z[i]) - lm[i] for i in range(len(z))]
    for i in z:
        idx = np.argmin(dists:=norm(lm_guesses-(i+curpos), axis=1))
        if dists[idx] > 2: 
            idx = len(lm_guesses)
            # print("heeeeereeeee")
            # print(lm_guesses.shape, (i+curpos)[:, np.newaxis].shape)
            lm_guesses = np.concatenate((lm_guesses, (i+curpos)[np.newaxis]), axis=0)
            lm.append(MX.sym(f'lm{idx}', 2))
        # print(dict(idx=idx, lm_len=len(lm)))
        lm_cons.append((x[-1] + DM(i) - lm[idx]))

solver_opts = {
    'ipopt.print_level': 0,
    'ipopt.sb': 'yes',
    'print_time': 0,
    # 'ipopt.linear_solver': 'MA57',
}
def solve_graph():
    global lm_guesses, lm, lm_cons
    global x_guesses, x, x_cons

    # construct optimization objective
    cons = vertcat(*x_cons, *lm_cons, x[0]-DM(path[0]))
    # and formulate our QP problem
    qp = {'x':vertcat(*x, *lm), 'f':cons.T@cons}

    # possible solvers
    # so i tried the quadratic ones because
    # like obviously this is a quadratic programming probem
    # but they're slower??
    # needs more info since my computer's BLAS/LAPACK setup is
    # entirely fucked
    # also that's the reason all the HSL solvers are much slower
    # on my mac for MPC, IPOPT with MA27 was like 10x faster than 
    # with MUMPS, but here, it's much slower. On WSL it's about equal
    # so idek anymore
    # prob best just to leave these here so we can test later

    # also maybe because this is an unconstrained optimization problem,
    # whereas these QP and NLP solvers are meant for constrained problems?
    # ig the whole IP method doesn't make as much sense without the need
    # for constraints. Since you can just optimize.

    # solver = qpsol('solver', 'qpoases', qp, {'printLevel':'none', 'print_time':0, 'print_problem':0})
    # solver = qpsol('solver', 'osqp', qp, {'print_time':0, 'print_problem':0})
    solver = nlpsol('solver', 'ipopt', qp, solver_opts)

    # actually solve the QP problem
    soln = np.array(solver(x0=vertcat(*x_guesses, *lm_guesses))['x'])
    split = len(x_guesses)*2

    s = (len(x_guesses), 2)
    x_guesses = np.array(soln[:split])
    x_guesses.resize(s)

    s = lm_guesses.shape
    lm_guesses = np.array(soln[split:])
    lm_guesses.resize(s)

    # print(x_guesses.shape, lm_guesses.shape)
    return list(x_guesses), lm_guesses

#%%
no_slam_x = [path[0]]
no_slam_lm = []
solve_times = []
for i in range(1, len(path)):
    # get fake odometry measurement
    dx = (path[i]-path[i-1]+(random(2)*0.1)) # add noise with stdev 0.01
    no_slam_x.append(no_slam_x[-1]+dx)
    # get fake observations
    z = np.concatenate(
        ((left[norm(left-path[i], axis=1)<vision_range]),
        (right[norm(right-path[i], axis=1)<vision_range])),
        axis=0
    )-path[i]
    z = z+random(z.shape)*1
    no_slam_lm.append(z+no_slam_x[-1])

    # update graph
    t0 = perf_counter()
    update_graph(dx, z)
    t1 = perf_counter()
    #* comment this line out to only solve at the end
    x_guesses, lm_guesses = solve_graph()
    t2 = perf_counter()
    solve_times.append([t1-t0, t2-t1, len(x_cons)+len(lm_cons)])
    # progress bar
    barlen = 20
    print(f'\r|{"@"*int(np.ceil(barlen*(i+1)/len(path)))}{"."*(barlen-int(np.floor(barlen*(i+1)/len(path))))}| {i+1}/{len(path)}', end='')
print()
toc = perf_counter()
x_guesses, lm_guesses = solve_graph()
tic = perf_counter()
print(f'solve time: {tic-toc}')
# %%
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, gridspec_kw=dict(height_ratios=[3,1]))

# plot true track
for ax in axs[0]:
    ax.plot(*path.T, color='tab:blue', linestyle='dashed')
    ax.plot(*left.T, color='tab:blue', label='track')
    ax.plot(*right.T, color='tab:blue')


axs[0][0].plot(*np.array(x_guesses).T, color='red', label='poses')
axs[0][0].scatter(*lm_guesses.T, color='green', label='landmarks', s=4)
axs[0][0].set_title('Localization with GraphSLAM')

axs[0][1].plot(*np.array(no_slam_x).T, color='red', linestyle='dashed', label='slam input path')
axs[0][1].scatter(*np.concatenate(no_slam_lm, axis=0).T, color='green', s=4)
axs[0][1].set_title('Localization without GraphSLAM')

axs[1][0].plot(np.array(solve_times)[:, 1], label='solve time')
axs[1][0].set_xlabel('Drive Distance (2m)')
axs[1][0].set_ylabel('Time (s)')
axs[1][0].set_title('Solve Time vs. Drive Distance')

axs[1][1].plot(*np.array(solve_times)[:, 2:0:-1].T, label='solve time')
axs[1][1].set_xlabel('number of edges')
axs[1][1].set_ylabel('Time (s)')
axs[1][1].set_title('Solve Time vs. Number of Edges')

fig.set_tight_layout(True)
# fig.suptitle(f"GraphSLAM: re-optimization at each new pose\nLandmark view distance: {vision_range}m, Solver: IPOPT+MUMPS")
fig.suptitle(f"GraphSLAM: single optimization after completed lap.\nLandmark view distance: {vision_range}m, Solver: IPOPT+MUMPS. Solve time: 0.1223s")
plt.show()