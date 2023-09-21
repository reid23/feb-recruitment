import numpy as np
from numpy.linalg import *
from casadi import *

class PolarGraphSLAM:
    def __init__(self, **settings):
        self.x0 = np.array([0,0])


        self.firstupdate = True
        self.landmarkTolerance = 2
        self.solver_opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            # 'ipopt.linear_solver': 'MA77'
        }

        self.Q = lambda n: DM_eye(n)
        self.R = lambda n: DM_eye(n)

        self.__dict__.update(settings) # sketchy but nobody's hacking us

        # these things are after since they shouldn't be settings

        self.x = [MX.sym('x0', 3)]
        self.xhat = [self.x0]
        self.x_edges = []
        self.lm = []
        self.lmhat = []
        self.lm_edges = []

    @np.vectorize
    def cartesian(x):
        return np.array([x[0]*np.cos(x[1]), x[0]*np.sin(x[1])])
    def update_graph(self, dx, z):
        """updates graph given odo and lm measurements

        Args:
            dx (ndarray): vector of shape (3, 1) describing the estimated change in [x, y, theta]^T since the previous update 
            z (ndarray): vector of shape (m, 2) describing locations of m landmarks relative to the car in form [r, theta]
        """
        self.xhat.append(self.xhat[-1]+dx)
        curpos = self.xhat[-1]
        
        self.x.append(MX.sym(f'x{len(self.x)}', 3))
        self.x_edges.append((self.x[-1]+DM(dx)-self.x[-1]))

        z = z + np.array([0, curpos[2]])
        z = PolarGraphSLAM.cartesian(z)
        
        if self.firstupdate:
            self.firstupdate = False
            self.lmhat = z+curpos
            self.lm = [MX.sym(f'lm{i}', 2) for i in range(z.shape[0])]
            self.lm_edges = [self.x[-1] + DM(z[i]) - self.lm[i] for i in range(z.shape[0])]
        for i in z:
            idx = np.argmin(dists:=norm(self.lmhat-(i+curpos), axis=1))
            if dists[idx] > self.landmarkTolerance:
                idx = len(self.lmhat)
                self.lmhat = np.concatenate((self.lmhat, (i+curpos)[np.newaxis]), axis=0)
                self.lm.append(MX.sym(f'lm{idx}', 2))
            self.lm_edges.append((self.x[-1] + DM(i) - self.lm[idx])) # x + z_i = lm_i
    
    def solve_graph(self):
        """solves the graph and updates everything
        """
        x_e = vertcat(*self.x_edges)
        lm_e = vertcat(*self.lm_edges)
        x0_e = self.x[0] - DM(self.x0)
        qp = {
            'x':vertcat(*self.x, *self.lm), 
            'f': (x_e.T@self.Q(x_e.shape[0])@x_e 
                + lm_e.T@self.R(lm_e.shape[0])@lm_e 
                + x0_e.T@x0_e)
        }
        # solver = qpsol('solver', 'qpoases', qp, {'printLevel':'none', 'print_time':0, 'print_problem':0})
        # solver = qpsol('solver', 'osqp', qp, {'print_time':0, 'print_problem':0})
        solver = nlpsol('solver', 'ipopt', qp, self.solver_opts)

        # actually solve the QP problem
        soln = np.array(solver(x0=vertcat(*self.xhat, *self.lmhat))['x'])
        split = len(self.xhat)*2

        s = (len(self.xhat), 2)
        x_guesses = np.array(soln[:split])
        x_guesses.resize(s)

        s = (len(self.lmhat), 2)
        lm_guesses = np.array(soln[split:])
        lm_guesses.resize(s)

        self.xhat[:x_guesses.shape[0]] = list(x_guesses)
        self.lmhat[:lm_guesses.shape[0]] = list(lm_guesses)
        
        return list(x_guesses), list(lm_guesses)