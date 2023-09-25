import numpy as np
from numpy.linalg import *
from casadi import *

### STATUS: BORKED
class PolarGraphSLAM:
    def __init__(self, **settings):
        self.x0 = np.array([0,0,0])


        self.firstupdate = True
        self.landmarkTolerance = 2
        self.solver_opts = {
            # 'ipopt.print_level': 0,
            # 'ipopt.sb': 'yes',
            # 'print_time': 0,
            'ipopt.linear_solver': 'MA77'
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

        self.silly = []

        r, t = SX.sym('r'), SX.sym('t')
        self.cartesianCasadi=Function('cart', (r, t), (vertcat(r*cos(t), r*sin(t)),))
        theta = SX.sym('theta')
        self.sinc=Function('sinc', (theta,), (sum([((-1)**n)*(theta**(2*n))/(np.math.factorial(2*n+1)) for n in range(4)]),))
        self.cosc=Function('cosc', (theta,), (-theta/2 + (theta**3)/24 - (theta**5)/720 + (theta**7)/40320,))
    def cartesian(X):
        return np.array([[x[0]*np.cos(x[1]), x[0]*np.sin(x[1])] for x in X])
    def rot(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
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
        # dx[1] += self.x0[-1]
        self.dxhat.append(dx)
        
        # r = dx[0]/dx[1]
        #            [    Dx              Dy       D theta]
        D = np.array([dx[0]*float(self.cosc(dx[1])), 
                      dx[0]*float(self.sinc(dx[1])), 
                      dx[1]])
        # self.silly.append(self.silly[-1]+D[:2])
        D[:2] = PolarGraphSLAM.rot(self.xhat[-1][-1])@D[:2]
        # D = np.array([(dx[0]*(cos(dx[1])-1))/dx[1]])
        self.xhat.append(self.xhat[-1]+D)
        curpos = self.xhat[-1]
        
        self.dx.append(MX.sym(f'dx{len(self.dx)}', 2)) # symbolic dx, dtheta
        # r = self.dx[-1][0]/self.dx[-1][1]

        self.x.append(self.x[-1]+vertcat(self.dx[-1][0]*self.cosc(self.dx[-1][1]), self.dx[-1][0]*self.sinc(self.dx[-1][1]), self.dx[-1][1])) # symbolic x, y, angle
        # self.x.append(self.x[-1]+vertcat((self.dx[-1][0]*(cos(self.dx[-1][1])-1))/self.dx[-1][1], (self.dx[-1][0]*(sin(self.dx[-1][1])-1))/self.dx[-1][1], self.dx[-1][1]))
        self.x_edges.append(self.dx[-1]-DM(dx))
        

        zhat = np.copy(z)
        z=DM(z)+repmat(vertcat(0, self.x[-1][-1]), 1, z.shape[0]).T
        # print(z.shape, len(vertsplit(z)))
        z=[self.cartesianCasadi(*horzsplit(i)) for i in vertsplit(z)] # symbolic
        
        zhat[:, 1] += np.ones(zhat.shape[0])*self.xhat[-1][-1]
        zhat = PolarGraphSLAM.cartesian(zhat)
        
        if self.firstupdate:
            self.firstupdate = False
            self.lmhat = zhat+curpos[:2] # add x and y
            self.lm = [MX.sym(f'lm{i}', 2) for i in range(len(z))]
            self.lm_edges = [self.x[-1][:2] + z[i] - self.lm[i] for i in range(len(z))]
        for i in range(len(z)):
            idx = np.argmin(dists:=norm(self.lmhat-(zhat[i]+curpos[:2]), axis=1))
            if dists[idx] > self.landmarkTolerance:
                idx = len(self.lmhat)
                self.lmhat = np.concatenate((self.lmhat, (zhat[i]+curpos[:2])[np.newaxis]), axis=0)
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



### STATUS: BORKED
class LinearGraphSLAM:
    def __init__(self, **settings):
        self.x0 = np.array([[0,0]])


        self.firstupdate = True
        self.landmarkTolerance = 2

        self.Q = lambda n: DM_eye(n)*5
        self.R = lambda n: DM_eye(n)

        self.__dict__.update(settings) # sketchy but nobody's hacking us

        # these things are after since they shouldn't be settings
        self.nx = 1
        self.nm = 0
        self.A = np.array([[1, 0, 1, 0],[0, 1, 0, 1]])
        self.b = np.copy(self.x0)
        self.xhat = self.x0
        self.lmhat = np.array([])

    def update_graph(self, dx, z):
        """updates graph given odo and lm measurements

        Args:
            dx (ndarray): vector of shape (2, 1) describing the estimated change in car location since the previous update 
            z (ndarray): vector of shape (m, 2) describing locations of m landmarks relative to the car
        """
        self.xhat += dx
        l = self.A.shape[1]
        # print(self.A.shape)
        print(([
            [0]*(self.nx*2 - 2) + [-1, 0, 1] + [0]*(l - self.nx*2 - 1), # add equation x_k - x_(k-1) = dx_k
            [0]*(self.nx*2 - 1) + [-1, 0, 1] + [0]*(l - self.nx*2 - 2), # add equation y_k - y_(k-1) = dy_k
        ]))
        self.A = np.append(self.A, np.array([
            [0]*(self.nx*2 - 2) + [-1, 0, 1] + [0]*(l - self.nx*2 - 1), # add equation x_k - x_(k-1) = dx_k
            [0]*(self.nx*2 - 1) + [-1, 0, 1] + [0]*(l - self.nx*2 - 2), # add equation y_k - y_(k-1) = dy_k
        ]), axis=0)
        self.b = np.append(self.b, dx) # add the rhs of the above equations

        # now deal with measurements
        self.b = np.append(self.b, z+self.xhat, axis=0)


        if self.firstupdate:
            self.firstupdate = False
            self.lmhat = z+self.xhat

            pre_dims = self.A.shape
            self.A = np.append(self.A, np.zeros((pre_dims[0]), len(z)), axis=1) # append all the landmark columns
            self.A = np.append(self.A, np.zeros((len(z), self.A.shape[1])), axis=0)
            for i in range(len(z)):
                self.A[pre_dims[0]+(i*2),   pre_dims[1]+(i*2)] = 1 # add the 1 for the x coord of the landmark
                self.A[pre_dims[0]+(i*2)+1, pre_dims[1]+(i*2)+1] = 1 # add the 1 for the y coord of the landmark
                self.A[pre_dims[0]+(i*2),   pre_dims[1]-2] = 1 # add the 1 for the x coord of cur pos
                self.A[pre_dims[0]+(i*2)+1, pre_dims[1]-1] = 1 # add the 1 for the y coord of cur pos
            return

        for i in z+self.xhat:
            idx = np.argmin(dists:=norm(self.lmhat-i, axis=1))
            if dists[idx] > self.landmarkTolerance:
                idx = len(self.lmhat)
                self.lmhat = np.concatenate((self.lmhat, i[np.newaxis]), axis=0)
                self.A = np.append(self.A, np.zeros((self.A.shape[0], 2)), axis=1)
                self.A[-2, -2] = 1
                self.A[-1, -1] = 1
            self.A = np.append(self.A, np.zeros((2, self.A.shape[1])), axis=0)
            self.A[-2, self.nx*2]       = 1
            self.A[-1, self.nx*2+1]     = 1
            self.A[-2, self.nx+idx*2]   = 1
            self.A[-1, self.nx+idx*2+1] = 1
        self.nx += 1
        self.nm += len(z)
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

if __name__ == '__main__':
    from graphslam_clean import Sim, get_path
    import matplotlib.pyplot as plt
    path, _, _ = get_path()
    sim = Sim(
        path, 
        4, 
        lambda shape: np.random.random(shape)*0.00001, 
        lambda shape: np.random.random(shape)*0.001,
        pose_divider=2,
        lm_divider=3,
        # mode=Sim.POLAR
    )
    # slam = PolarGraphSLAM(x0=np.array([path[0][0], path[0][1], -np.pi/2]))
    slam = LinearGraphSLAM(x0=np.array([0.0,0.0]))
    count = 0
    start = np.array([0.0,0.0])
    angle = 0
    states = []
    landmarks = []
    for dx, z in sim:
        angle += dx[1]
        #TODO: implement stupid test here for measurements. Isolate errors in this file or Sim
        # Sim.rot(angle)@np.array([[dx[1]], [0]])
        states.append(list(start))
        start += (sim.rot(angle)@np.array([[dx[0]], [0]])).flatten()
        slam.update_graph(dx, z)
        # print(count)
        for i in z:
            landmarks.append(start+(sim.rot(angle+i[1])@np.array([[i[0]],[0]])).flatten())
        count += 1
        # if count > 30: break
    # plt.scatter(*np.array(path).T, c=np.linspace(0, 1, len(states)), cmap='coolwarm')
    # plt.scatter(*np.array(landmarks).T, c=np.linspace(0, 1, len(landmarks)), cmap='coolwarm')
    # plt.scatter(*np.array(states).T, c=np.linspace(0, 1, len(states)), cmap='coolwarm')
    # slam.solve_graph()
    
    fig, ax = plt.subplots()
    ax.scatter(*(np.array(slam.xhat)[:, :2]).T, c=np.linspace(0, 1, len(slam.xhat)), cmap='coolwarm')
    for i in slam.xhat:
        ax.annotate(round(i[-1], 1), i[:2])
    plt.show()