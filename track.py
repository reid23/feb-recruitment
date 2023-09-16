#%%
import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

#%%
pts = np.array([[0,0],[50,0],[50, 60],[-100, 60],[-100, -50],[-50, -50],[-50,0]])
# radius: 10
# track width: 4
# units: m

# lil utility function
def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
def smooth_path(pts, rad, dx=1, curve_est_dist = 15, extra_points=10):
    """enforces a minimum radius along a path. returns the path as a mesh with spacing dx.

    Args:
        pts (ndarray): an matrix in $\mathbb{R}^{n \\times 2}$ of points defining the path
        rad (float): minimum radius of the path
        dx (float, optional): interpolation distance. Defaults to 1.
        curve_est_dist (int, optional): how many points to use to fit a circle to estimate radius of curvature. defaults to 15.
        extra_points (int, optional): how many extra points to add on to the sub-path after finding the curvatures. defaults to 10.
    """

    # step 1: interpolate points to regular intervals
    path = []
    for idx, pt in enumerate(pts):
        interp_count = int(np.ceil(norm(pt-pts[idx-1])/dx))
        # print(np.sqrt((pt-pts[idx-1]).T@(pt-pts[idx-1])), interp_count)
        path.append(
            np.array(
                # affine combination               0..1 so its from one to the other    exclude endpoint bc we concatenate these all together at the end
                [a*pt + (1-a)*pts[idx-1] for a in np.linspace(0, 1, interp_count, endpoint=False)]
            )
        )
    path = np.concatenate(path, axis=0)
    
    # step 2: find places where min rad needs to be enforced
    bad = []
    curve_est_n = int(curve_est_dist/dx)
    for idx in range(len(path)):
        segment = np.array([path[idx-curve_est_n+i] for i in range(curve_est_n)]) # funny listcomp to deal with wrapping

        # check for colinearity - this can mess with the circle fit since radius will appear to be infinite
        # ideally we could check the rank and if it's 3 (ie, the affine span of the segment is the whole plane,
        # plus the 1 from the barycentric coords), we would know we're okay, but since there are 
        # :sparkles: float errors :sparkles:, its better to check if any singular values are too close to zero. 
        # we need to span R^3 since we have barycentric coordinates so 2+1=3
        # we can't use the determinant because this is a nonsquare matrix.
        if abs(svd(np.concatenate((segment.T, np.ones((1,curve_est_n)))))[1][-1]) < 0.001:
            continue
        
        # fit circle to points using standard linear regression
        # derivation is trivial and left as an exercise for the reader
        # (yeah that's right reid in the future, fuck you, i accidentally
        # closed the desmos tab with the reasoning and now i'm coping)
        M = np.concatenate([2*segment, np.ones((curve_est_n, 1))], axis=1)
        res = inv(M.T@M)@M.T@np.array([[-(a.T@a)] for a in segment])
        r = np.sqrt(-res[2]+res[1]**2+res[0]**2)

        if r<rad:
            # bad.append(segment)
            bad.append(np.array([path[idx-curve_est_n+i-extra_points] for i in range(curve_est_n+(2*extra_points))]))
            # bad.append(np.array([path[idx-curve_est_n+i]]))
    
    # concatenate all the bad places together
    bad = np.concatenate(bad, axis=0)

    # remove duplicates. Each time a bad was found, it added the whole segment,
    # so we'll have tons of duplicates.

    # extra bad becaause np.unique messes up the order
    badbad = np.unique(bad, axis=0)
    
    bad = []
    for point in path:
        if sum(to_delete:=np.nonzero(norm(badbad-point, axis=1)<(dx/5))) > 0:
            badbad = np.delete(badbad, to_delete, axis=0)
            bad.append(point)
    bad = np.array(bad)

    # split it into the separate corners:
    # if there's a dx bigger than the dx of the points, we know it's a gap
    bad = np.split(
        bad, 
        np.nonzero(
            norm(
                # we need to prepend the last point
                # so the diffs come out with the same shape
                # as the path
                # it makes sense to add the last point because the
                # racetrack is a loop! 
                np.diff(bad, prepend=bad[-1:], axis=0),
                axis=1
            )>(2*dx)
        )[0],
        axis=0
    )
    # the prepend of bad[-1] means split() tried to split
    # our path at index 0, which makes sense, but it leaves
    # an extra segment of length zero at the front which we
    # need to remove
    bad[-1] = np.concatenate((bad[-1], bad[0]), axis=0)
    bad = bad[1:]

    # for this bit see the derivation on the paper
    for area in bad:
        # find location of the cutout in the path
        cutout_len = area.shape[0]
        cutout_idx = np.argmin(norm(path-area[0], axis=1)) # where both x and y match

        b1, b2 = area[0], area[-1]
        v1, v2 = area[0]-area[1], area[-1]-area[-2]
        v1, v2 = area[0]-area[1], area[-1]-area[-2]
        # convert to column vectors
        b1, b2, v1, v2 = b1[:, np.newaxis], b2[:, np.newaxis], v1[:, np.newaxis], v2[:, np.newaxis]
        # get normalized versions
        v1hat, v2hat = v1/norm(v1), v2/norm(v2)

        # find intersection point, angle there, and distance there from tangent point
        x = b2 - v2*solve(np.concatenate([v1, v2], axis=1), b2-b1)[1] # use [1] (t2) since t1 is negative (bc of the funny way we constructed v1 and v2)
        theta = np.arccos(v1hat.T@v2hat)[0,0] # [0,0] converts to float
        c = rad/np.tan(theta/2)

        # functions for straight bit
        # second has t and (1-t) flipped because we want the
        # points to go from b1->T1 but T2->b2
        # (so they're in order for the track)
        straight1 = lambda t: (1-t)*b1 + (t)*(x+c*v1hat)
        straight2 = lambda t: t*b2 + (1-t)*(x+c*v2hat)

        # find lengths of linear segments
        l1 = norm(b1-(x+c*v1hat))
        l2 = norm(b2-(x+c*v2hat))

        # vectorize and apply to all the t values
        straight1, straight2 = np.vectorize(straight1, signature='(m)->(2,m)'), np.vectorize(straight2, signature='(m)->(2,m)')

        t1 = np.linspace(0, 1, int(np.ceil(l1/dx)), endpoint=True) 
        t2 = np.linspace(0, 1, int(np.ceil(l2/dx)), endpoint=True) 
        straight1, straight2 = straight1(t1), straight2(t2)

        vbarhat = (1/2)*(v1hat+v2hat)
        vbarhat = vbarhat/norm(vbarhat)

        o = x + vbarhat*norm([rad, c])
        sign = -np.sign(det(np.concatenate((v1, v2), axis=1))) # negative bc of the cardinality we assumed during the derivation
        arc = lambda phi: o+rot(phi*sign)@(-rad*vbarhat)
        alpha = (1/2)*(np.pi-theta)
        phi_vals = np.linspace(-alpha, alpha, int(np.ceil(2*alpha*rad/dx)), endpoint=True)
        arc = np.concatenate(list(map(arc, phi_vals)), axis=1)

        
        # now splice!
        snippet = np.concatenate((straight1, arc, straight2), axis=1)
        # cope about if there's a corner covering the start area
        if cutout_idx+cutout_len > len(path): 
            path = np.concatenate((path[cutout_idx+cutout_len-len(path):cutout_idx], snippet.T), axis=0)
        else:
            path = np.concatenate((path[:cutout_idx], snippet.T, path[(cutout_idx+cutout_len):]), axis=0)

    # remove duplicates again, since sometimes the edges of the cut regions overlapped 
    nasty = np.unique(path, axis=0)
    
    newpath = []
    for point in path:
        if np.sum(to_delete:=np.nonzero(norm(nasty-point, axis=1)<(dx/5))) > 0:
            nasty = np.delete(nasty, to_delete, axis=0)
            newpath.append(point)
    return np.array(newpath)

def offset_path(path, offset):
    # use diff to get vectors parallel to the path
    diffs = np.diff(path, prepend=path[-1:], axis=0).T
    # rotate all the vecs ccw by 90 degrees to get vecs orth to path
    shifts = rot(np.pi/2)@diffs
    # scale them properly by `offset`
    shifts = offset*shifts/norm(shifts, axis=0)
    # shift the actual path by these shifts and return it
    return path+shifts.T # need shifts.T to convert back to row vectors

#util for displaying nicely
def loopify(path):
    return np.concatenate((path[-1:], path), axis=0).T

def export_path(path, width):
    df = pd.DataFrame(np.concatenate((path, np.ones(path.shape)*width*0.5), axis=1))
    df.columns = ['x_m', 'y_m', 'w_left', 'w_right']
    print(df.head())
    df.to_csv('path.csv', ',')

def get_test_track():
    return smooth_path(pts, 10)
# %%
if __name__ == '__main__':
    track = ['test_track', 'curved_track', 'sword_track'][0]

    if track != 'test_track':
        with open(track, "r") as f:
            pts = np.array(eval(f.read())) # eval() is okay bc I'm the only one writing test_track or whatever

    if track == 'curved_track':
        path = smooth_path(pts, 20, dx=1, curve_est_dist = 15, extra_points=20)
    elif track == 'sword_track':
        path = smooth_path(pts, 15, dx=1, curve_est_dist = 15, extra_points=15)
    elif track == 'test_track':
        path = smooth_path(pts, 10)

    left = offset_path(path, 2)
    right = offset_path(path, -2)
    plt.plot(*loopify(left), color='tab:blue', label='boundary')
    plt.plot(*loopify(right), color='tab:blue')

    plt.plot(*loopify(pts), color='tab:orange', label='input path')
    plt.plot(*loopify(path), color='tab:blue', linestyle='dashed', label='centerline')

    plt.legend()
    # plt.scatter(*path.T, c=np.linspace(0, 1, len(path)), cmap='coolwarm')
    plt.show()

    # %%
    export_path(path, 4)