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


    return new_spline, knots, keypt_s