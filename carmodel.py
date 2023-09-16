#%%
from casadi import *
import numpy as np
from numpy.linalg import *

tstep = 0.2
thoriz = 1
nsteps = int(thoriz/tstep)

x = SX.sym('x', 4) # [x, y, v, th]
u = SX.sym('u', 2) # [acc, steering]
m = 1
b = 3


xdot = vertcat(
    x[2]*cos(x[3]),
    x[2]*sin(x[3]),
    u[0]/m,
    (x[2]/b)*tan(u[1])
)

ode = {'x': x, 'u':u, 'ode':xdot}
intfunc = integrator('F_i', 'rk', ode, 0, tstep)

from track import get_test_track

path = get_test_track()
path = DM(path)
solver_opts = {
    'ipopt.print_level': 0,
    'ipopt.sb': 'yes',
    'print_time': 0,
    'ipopt.linear_solver': 'MA27',
}

def make_step(soln, t, x0=None, steps=1):
    # create a control value for each time step
    # these are what we'll optimize later
    u = [MX.sym('u' + str(j), 2) for j in range(nsteps)]

    # init some variables
    cost_acc = 0
    g = []
    x = x0 if x0 is not None else DM([0,0,0,0])

    # loop to build the computation graph
    # integrates from t0..tf
    # builds the cost function
    # turns out casadi is like torch tensors
    # where they remember what happened to them
    # so once we integrate with symbolic u's,
    # we can optimize

    # we have to do this every time because I don't think it updates the whole graph if you change x0???
    # something like that
    for j in range(nsteps):
        # actual integration
        # print(type(x))
        res = intfunc(x0=x, u=u[j])
        x = res['xf']

        # update cost
        i = t+j
        # cost_acc += 100*cos(x[1]) + x[3]**2 + (5*x[0])**2
        # cost_acc += 100*cos(x[1]) + (j/10)*x[3]**2 + (5*x[0])**2
        # loc = np.argmin(vertcat(*list(map(norm_2, vertsplit((path-repmat(x[0:2], 1, path.shape[0]).T))))))
        # direction = path[loc]-path[loc+1]
        # print(loc)
        cost_acc += -(x[2][0]*100) + u[j][0]**2

        g += [u[j][0], u[j][1], x[2][0], norm_2(path[i%path.shape[0]]-x[0:2])[0][0], (u[j][1]-u[j-1][1])**2]
    

    # formulate NLP and create the solver with it
    nlp = {'x':vertcat(*u), 'f':cost_acc, 'g':vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', nlp, solver_opts)
    
    # this is where the actual solving takes place
    # everything above this takes about 0.01-0.015s
    # this is what takes all the time
    soln = solver(
        # steps allows us to optimally set the initial guess
        # even if more than one timestep has passed since last iteration
        x0=vertcat(soln[steps:], DM([soln[-1]]*steps)),

        # don't need constraints on u (aka x) because
        # the quadratic term in g does that already
        # so just use g

        # these match g
        lbg=[-30, -pi/4, -10, -1, -100]*nsteps,
        ubg=[10, pi/4, 70, 4, 100]*nsteps
    )['x']
    
    return soln

us = DM([[0.0,0.0]]*nsteps)
x = DM([-20.0,0.0,0.0,0.0])

record = [np.array(x)]
for i in range(int(5/tstep)):
    us = make_step(us, int(i*tstep*5), x0=x)
    x = intfunc(x0=x, u=us[0])['xf']
    record.append(np.array(x))
    print(np.array(x))

record = np.array(record)
#%%
import matplotlib.pyplot as plt
plt.plot(*np.array(path).T)
plt.scatter(*record[:, :2, 0].T, c=record[:, 2, 0], cmap='coolwarm')
plt.plot()
# %%
