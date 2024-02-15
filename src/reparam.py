import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
import scipy.stats
from scipy.special import roots_legendre
import cv2

from utils import set_axes_equal

ROOTS, WEIGHTS = roots_legendre(6)


def arclength(dspline, a, b):
    t = (b - a)/2 * ROOTS + (a + b)/2
    l = (b - a)/2 * \
        np.linalg.norm(dspline(t), axis=-1).dot(WEIGHTS)
    return l 

def reparam(spline, keypt_u):
    INNER_MULT = 1 # multiplier on number control points
    OUTER_MULT = 2 # multiplier to increase number of sampled points
    
    # Integrate curve speed to get segment lengths
    # And lengths between keypoints
    knots, ctrl, k = spline.t, spline.c, spline.k
    dspline = spline.derivative()

    total_l = 0
    for a, b in zip(knots[:-1], knots[1:]):
        total_l += arclength(dspline, a, b)
    
    segment_l = []
    u_idx = 0
    keypt_s = []
    l_traversed = 0
    for a, b in zip(knots[:-1], knots[1:]):
        li = arclength(dspline, a, b) / total_l
        segment_l.append(li)
        while(u_idx < len(keypt_u) and keypt_u[u_idx] <= b + 1e-4):
            keypt_si = l_traversed + arclength(dspline, a, keypt_u[u_idx])/total_l
            keypt_s.append(keypt_si)
            u_idx += 1
        l_traversed += li
    
    """
    ##########################
    The following is redundant
    ##########################
    """
    # Find m equally spaced points
    m = (ctrl.shape[0] - k)*INNER_MULT*OUTER_MULT
    spacing = sum(segment_l)/(m-1)
    t_spaced = [knots[0]]
    l_traversed = 0
    j = 0
    for i in range(1, m-1):
        l_diff = i*spacing - l_traversed
        while j < len(segment_l) and l_diff > segment_l[j]:
            l_traversed += segment_l[j]
            j += 1
            l_diff = i*spacing - l_traversed
        
        def rootfinder(x):
            return l_diff - arclength(dspline, knots[j], x)/total_l
        
        t_bisect = scipy.optimize.bisect(rootfinder, knots[j], knots[j+1])
        t_spaced.append(t_bisect)
    t_spaced.append(knots[-1])

    # New spline parameters
    s_spaced = [i*spacing for i in range(m)]

    # Fit spline to previously collected points
    init_pts = spline(t_spaced)
    init_s = np.array(s_spaced)

    # Construct spline
    tck, *_ = interp.splprep(
        init_pts.T,
        u=init_s,
        task=-1,
        t=knots,
        k=k,
        nest=init_s.shape[0]+k+1
    )
    t = tck[0]
    c = np.array(tck[1]).T
    k = tck[2]
    new_spline = interp.BSpline(t, c, k)

    # old_samples = spline(np.linspace(0, knots[-1], 150))
    # new_samples = new_spline(np.linspace(0, init_s[-1], 150))

    # # plt.imshow(img1, cmap="gray")
    # ax = plt.subplot(projection="3d")
    # ax.plot(old_samples[:, 0], old_samples[:, 1], old_samples[:, 2])
    # ax.plot(new_samples[:, 0], new_samples[:, 1], new_samples[:, 2])
    # ax.scatter(init_pts[:, 0], init_pts[:, 1], init_pts[:, 2],\
    #         c=init_s, cmap="hot")
    # # plt.axis("equal")
    # set_axes_equal(ax)
    # plt.show()
    # validate_reparam(new_spline)


    return new_spline, knots, keypt_s

def validate_reparam(spline):
    # Integrate curve speed to get segment lengths
    knots, ctrl, k = spline.t, spline.c, spline.k
    segment_l = []
    dspline = spline.derivative()
    samples = np.linspace(knots[0], knots[-1], 150)

    for a, b in zip(samples[:-1], samples[1:]):
        li = arclength(dspline, a, b)
        segment_l.append(li)
    
    # Visualize segments, they should have similar length
    plt.scatter(samples[:-1], segment_l)
    plt.show()



if __name__ == "__main__":
    t = np.linspace(-1, 3)
    num_ctrl = 15
    k = 3
    knots = np.linspace(0, 1, num_ctrl+k+1)
    x1 = np.sin(t)
    x2 = np.cos(t)
    x = np.stack((x1, x2))
    tck, *_ = interp.splprep(x, task=-1, t=knots, k=k)
    t = tck[0]
    c = np.array(tck[1]).T
    k = tck[2]
    tck = interp.BSpline(t, c, k)
    # validate_reparam(tck)
    new_spline = reparam(tck)
    validate_reparam(new_spline)