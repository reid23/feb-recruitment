#%%
from casadi import *
from track import get_test_track, offset_path
import numpy as np
from numpy.linalg import *
from numpy.random import random
import matplotlib.pyplot as plt
from time import perf_counter

def get_path():
    path = get_test_track()
    path = np.array([path[i] for i in range(path.shape[0]) if i%3==0])
    # path = path-path[0]
    left = offset_path(path, 4)
    right = offset_path(path, -4)
    left = np.array([left[i] for i in range(path.shape[0]) if i%2==0])
    right = np.array([right[i] for i in range(path.shape[0]) if i%2==0])
    return path, left, right
path, left, right = get_path()

#%%
class Sim:
    CARTESIAN = 1
    POLAR = 2
    def __init__(self, path, width, odo_noise_func, measurement_noise_func, vision_range=10, pose_divider=2, lm_divider=3, shift_to_origin=True, record_no_slam=True, mode=CARTESIAN):
        self.path = path
        self.width = width
        self.pose_divider = pose_divider
        self.lm_divider = lm_divider
        self.shift_to_origin = shift_to_origin
        self.mode=mode
        self.idx = 0
        self.dx_noise = odo_noise_func
        self.z_noise = measurement_noise_func
        self.vision_range = vision_range
        self.record_no_slam = record_no_slam
        self.no_slam_x = [self.path[0]]
        self.no_slam_m = []
        self._update()
    def _update(self):
        """util to run whenever settings are changed
        """
        if self.shift_to_origin:
            self.path -= self.path[0]
        self.x = np.array([self.path[i] for i in range(self.path.shape[0]) if i%self.pose_divider==0])
        self.left = offset_path(self.path, self.width/2)
        self.right = offset_path(self.path, -self.width/2)
        self.left = np.array([self.left[i] for i in range(self.left.shape[0]) if i%self.lm_divider==0])
        self.right = np.array([self.right[i] for i in range(self.right.shape[0]) if i%self.lm_divider==0])
    def rot(self, a):
        return np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
    def get_step(self):
        # get fake observations
        z = np.concatenate(
            ((self.left[norm(self.left-self.path[self.idx], axis=1)<self.vision_range]), # change when creating an actual sim class
            (self.right[norm(self.right-self.path[self.idx], axis=1)<self.vision_range])),
            axis=0
        )-self.path[self.idx]
        # plt.scatter(*(z+self.path[self.idx]).T)
        # plt.show()
        dx = self.path[self.idx]-self.path[self.idx-1]

        if self.record_no_slam:
            dx = self.path[self.idx]-self.path[self.idx-1]
            self.no_slam_x.append(self.no_slam_x[-1]+dx)
            self.no_slam_m.append(z+self.no_slam_x[-1])

        if self.mode == Sim.POLAR:
            dx = self.path[self.idx]-self.path[self.idx-1]
            dx_prev = self.path[self.idx-1]-self.path[self.idx-2]
            dx_out = np.array([norm(dx), (1 if det(np.concatenate((dx_prev[:, np.newaxis], dx[:, np.newaxis]), axis=1))>0 else -1)*np.arccos(dx@dx_prev/(norm(dx)*norm(dx_prev)))]) # convert to polar
            
            z = (self.rot(-(1 if det(np.concatenate((np.array([[1],[0]]), dx[:, np.newaxis]), axis=1))>0 else -1)*np.arccos(dx@np.array([1,0])/norm(dx)))@(z.T)).T
            norm_z = z/norm(z, axis=1, keepdims=True)
            # print((norm_z[:, 1]/norm_z[:, 0])[:, np.newaxis].shape)
            # print(norm(z, axis=1, keepdims=True).shape)
            # print(((np.sign([det(np.concatenate((np.array([[1],[0]]),norm_z[i][:, np.newaxis]), axis=1)) for i in range(norm_z.shape[0])])*np.arccos(norm_z[:, 0]))[:, np.newaxis]).shape)
            z = np.concatenate((norm(z, axis=1, keepdims=True), (np.sign([det(np.concatenate((np.array([[1],[0]]),norm_z[i][:, np.newaxis]), axis=1)) for i in range(norm_z.shape[0])])*np.arccos(norm_z[:, 0]))[:, np.newaxis]), axis=1)
            dx = dx_out
        self.idx += 1
        return dx+self.dx_noise((2,)), z+self.z_noise(z.shape)
    def __iter__(self):
        self.idx=0
        self.no_slam_x = [self.path[0]]
        self.no_slam_m = []
        self._update()
        return self
    def __next__(self):
        if self.idx==len(self.path):
            raise StopIteration
        out = self.get_step()
        return out




class GraphSLAM:
    def __init__(self, **settings):
        self.x0 = np.array([0,0])


        self.firstupdate = True
        self.landmarkTolerance = 2
        self.solver_opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.linear_solver': 'MA57'
        }

        self.Q = lambda n: DM_eye(n)*5
        self.R = lambda n: DM_eye(n)

        self.__dict__.update(settings) # sketchy but nobody's hacking us

        # these things are after since they shouldn't be settings

        self.x = [MX.sym('x0', 2)]
        self.xhat = [self.x0]
        self.x_edges = []
        self.lm = []
        self.lmhat = []
        self.lm_edges = []

    def update_graph(self, dx, z):
        """updates graph given odo and lm measurements

        Args:
            dx (ndarray): vector of shape (2, 1) describing the estimated change in car location since the previous update 
            z (ndarray): vector of shape (m, 2) describing locations of m landmarks relative to the car
        """
        self.xhat.append(self.xhat[-1]+dx)
        curpos = self.xhat[-1]
        
        self.x.append(MX.sym(f'x{len(self.x)}', 2))
        self.x_edges.append((self.x[-2]+DM(dx)-self.x[-1]))

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



#%%
if __name__ == '__main__':
    no_slam_x = [path[0]]
    no_slam_lm = []
    solve_times = []
    vision_range = 10
    slam = GraphSLAM(x0=path[0])
    for i in range(1, len(path)):
        # get fake odometry measurement
        dx = (path[i]-path[i-1]+(random(2)*0.1)) # add noise with stdev 0.01
        no_slam_x.append(no_slam_x[-1]+dx)
        # get fake observations
        z = np.concatenate(
            ((left[norm(left-path[i], axis=1)<vision_range]), # change when creating an actual sim class
            (right[norm(right-path[i], axis=1)<vision_range])),
            axis=0
        )-path[i]
        z = z+random(z.shape)*1
        no_slam_lm.append(z+no_slam_x[-1])

        # update graph
        t0 = perf_counter()
        slam.update_graph(dx, z)
        t1 = perf_counter()
        #* comment this line out to only solve at the end
        x_guesses, lm_guesses = slam.solve_graph()
        t2 = perf_counter()
        solve_times.append([t1-t0, t2-t1, len(slam.x_edges)+len(slam.lm_edges)])
        # progress bar
        barlen = 20
        print(f'\r|{"@"*int(np.ceil(barlen*(i+1)/len(path)))}{"."*(barlen-int(np.floor(barlen*(i+1)/len(path))))}| {i+1}/{len(path)}', end='')
    # %%
    # plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, gridspec_kw=dict(height_ratios=[3,1]))

    # plot true track
    for ax in axs[0]:
        ax.plot(*path.T, color='tab:blue', linestyle='dashed')
        ax.plot(*left.T, color='tab:blue', label='track')
        ax.plot(*right.T, color='tab:blue')


    axs[0][0].plot(*np.array(x_guesses).T, color='red', label='poses')
    axs[0][0].scatter(*np.array(lm_guesses).T, color='green', label='landmarks', s=4)
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