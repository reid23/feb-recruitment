import numpy as np
from numpy.linalg import *
from casadi import *

class PolarGraphSLAM:
    def __init__(self, **settings):
        self.x0 = np.array([0,0,0])


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
        self.dx = []
        self.dxhat = []
        self.x = [MX.sym('x0', 3)] # [x, y, th]
        self.xhat = [self.x0]
        self.x_edges = []
        self.lm = []
        self.lmhat = []
        self.lm_edges = []

        r, t = SX.sym('r'), SX.sym('t')
        self.cartesianCasadi=Function('cart', (r, t), vertcat(r*sin(t), r*cos(t)))

    @np.vectorize
    def cartesian(x):
        return np.array([x[0]*np.cos(x[1]), x[0]*np.sin(x[1])])
    def update_graph(self, dx, z):
        """updates graph given odo and lm measurements

        Args:
            dx (ndarray): vector of shape (3, 1) describing the estimated change in [D x, D theta]^T since the previous update 
            z (ndarray): vector of shape (m, 2) describing locations of m landmarks relative to the car in form [r, theta]
        """
        # taking a silly from FTC
        # a silly from Albert Cai
        # a silly called *circular odometry*
        # we assume we moved in a circle, which will give us better integration
        # this kinda worked for mecanum robots, since it made things differentiable
        # but it works much better for cars because they *do* move in circles
        self.dxhat.append(dx)
        r = dx[0]/dx[1]
        #            [    Dx              Dy       D theta]
        D = np.array([r*cos(dx[1])-r, r*sin(dx[1]), dx[1]])
        self.xhat.append(self.xhat[-1]+D)
        curpos = self.xhat[-1]
        
        self.dx.append(MX.sym(f'dx{len(self.dx)}', 2)) # symbolic dx, dtheta
        r = self.dx[-1][0]/self.dx[-1][1]

        self.x.append(self.x[-1]+vertcat(r*cos(self.dx[-1][1])-r, r*sin(self.dx[-1][1]), self.dx[-1][1])) # symbolic x, y, angle
        self.x_edges.append(self.dx[-1]-DM(dx))
        

        z=DM(z)+repmat(SX([0, 0, self.x[-1][-1]]), z.shape[0], 1)
        z=[self.cartesianCasading(*i) for i in z] # symbolic
        
        zhat = np.copy(z)
        zhat[:, 1] += np.ones(zhat.shape[0])*self.xhat[-1][-1]
        zhat = PolarGraphSLAM.cartesian(zhat)
        
        if self.firstupdate:
            self.firstupdate = False
            self.lmhat = z+curpos[:2] # add x and y
            self.lm = [MX.sym(f'lm{i}', 2) for i in range(z.shape[0])]
            self.lm_edges = [self.x[-1][:2] + z[i] - self.lm[i] for i in range(z.shape[0])]
        for i in range(len(z)):
            idx = np.argmin(dists:=norm(self.lmhat-(zhat[i]+curpos), axis=1))
            if dists[idx] > self.landmarkTolerance:
                idx = len(self.lmhat)
                self.lmhat = np.concatenate((self.lmhat, (zhat[i]+curpos)[np.newaxis]), axis=0)
                self.lm.append(MX.sym(f'lm{idx}', 2))
            self.lm_edges.append((self.x[-1][:2] + z[i] - self.lm[idx])) # x + z_i = lm_i
    
    def solve_graph(self):
        """solves the graph and updates everything
        """
        x_e = vertcat(*self.x_edges)
        lm_e = vertcat(*self.lm_edges)
        x0_e = self.x[0] - DM(self.x0)
        qp = {
            'x':vertcat(self.x[0], *self.dx, *self.lm), 
            'f': (x_e.T@self.Q(x_e.shape[0])@x_e 
                + lm_e.T@self.R(lm_e.shape[0])@lm_e 
                + x0_e.T@x0_e)
        }
        # solver = qpsol('solver', 'qpoases', qp, {'printLevel':'none', 'print_time':0, 'print_problem':0})
        # solver = qpsol('solver', 'osqp', qp, {'print_time':0, 'print_problem':0})
        solver = nlpsol('solver', 'ipopt', qp, self.solver_opts)

        # actually solve the QP problem
        soln = np.array(solver(x0=vertcat(self.x0, *self.dxhat, *self.lmhat))['x'])
        new_xhat = Function('x', [self.x[0]]+self.dx, self.x)(self.x0, *soln[1:len(self.dxhat)+1])
        new_lmhat = soln[len(self.dxhat)+1:]
        
        new_dxhat = np.array(soln[1:len(self.dxhat)+1])
        new_dxhat.resize(len(self.dxhat), 2)
        self.dxhat = list(new_dxhat)

        new_xhat = np.array(new_xhat)
        new_xhat.resize(len(self.xhat), 3)
        self.xhat = list(new_xhat)

        new_lmhat = np.array(new_lmhat)
        new_lmhat.resize(len(self.lmhat), 2)
        self.lmhat = list(new_lmhat)
        return list(new_xhat), list(new_lmhat)
    
if __name__ == '__main__':
    from graphslam_clean import Sim, get_path
    path, _, _ = get_path()
    sim = Sim(path, 4, lambda shape: np.random.random(shape)/10, lambda shape: np.random.random(shape)*2)
    slam = PolarGraphSLAM(x0=np.array([path[0][0], path[0][1], 0]))
    for dx, z in sim:
        slam.update_graph(dx, z)
        slam.solve_graph()
    