import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
from scipy.sparse import csc_matrix
import cv2
import osqp

from utils import *
from reparam import reparam

CONSTR_WIDTH_2D = 5

def optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, cam2img, P1, P2):
    # Get necessary values
    init_pts, keypoint_idxs = keypoints[order], np.arange(len(order))
    knots, init_u, constr_lower_d, constr_upper_d = optim_init(init_pts, keypoints, keypoint_idxs, order, cam2img)
    keypt_u = init_u[keypoint_idxs]
    k = 3
    num_ctrl = len(knots)-k-1
    num_constr = len(keypt_u)*3
    spline = None

    constr_centers = keypoints[order]
    constr_lower_px = constr_centers[:, 1] - CONSTR_WIDTH_2D
    constr_upper_px = constr_centers[:, 1] + CONSTR_WIDTH_2D
    constr_lower_py = constr_centers[:, 0] - CONSTR_WIDTH_2D
    constr_upper_py = constr_centers[:, 0] + CONSTR_WIDTH_2D

    # Put bounds in a good shape
    # x constr
    constr_lower_px_rshp = np.repeat(
        constr_lower_px, num_ctrl*3
    ).reshape(num_constr//3, num_ctrl*3)
    constr_upper_px_rshp = np.repeat(
        constr_upper_px, num_ctrl*3
    ).reshape(num_constr//3, num_ctrl*3)

    # y constr
    constr_lower_py_rshp = np.repeat(
        constr_lower_py, num_ctrl*3
    ).reshape(num_constr//3, num_ctrl*3)
    constr_upper_py_rshp = np.repeat(
        constr_upper_py, num_ctrl*3
    ).reshape(num_constr//3, num_ctrl*3)
    # knots, num_ctrl, k = spline.t, len(spline.c), spline.k

    
    def QP_step(init_guess, knots, keypt_s):
        # Set up optimization...
        solver = osqp.OSQP()

        # Create objective function
        deriv_coeff = (
            get_deriv_matrix(knots[2:-2], num_ctrl-2, k-2) @
            get_deriv_matrix(knots[1:-1], num_ctrl-1, k-1) @
            get_deriv_matrix(knots, num_ctrl, k)
        )
        weight_coeff = np.diag(
            np.repeat(knots[4:-3] - knots[3:-4], 3)
        )
        loss_coeff = (
            deriv_coeff.T @
            weight_coeff @
            deriv_coeff
        )

        # Create constraints
        spl_bases = np.zeros((num_constr, num_ctrl*3))
        I = np.eye(num_ctrl)
        for i in range(num_ctrl):
            basis = interp.BSpline(knots, I[i], k)
            basis_eval = basis(keypt_s)
            spl_bases[::3, 3*i] = basis_eval
            spl_bases[1::3, 3*i+1] = basis_eval
            spl_bases[2::3, 3*i+2] = basis_eval
            
        cam2img_rep = np.zeros((num_constr, num_constr))
        for i in range(0, num_constr, 3):
            cam2img_rep[i:i+3, i:i+3] = cam2img
        spl_eval_matrix = cam2img_rep @ spl_bases
        
        I_constr = np.eye(num_constr)
        eval_select_x = I_constr[::3] @ spl_eval_matrix
        eval_select_y = I_constr[1::3] @ spl_eval_matrix
        eval_select_z = I_constr[2::3] @ spl_eval_matrix
        
        x_lower = constr_lower_px_rshp * eval_select_z - eval_select_x
        x_upper = eval_select_x - constr_upper_px_rshp * eval_select_z
        y_lower = constr_lower_py_rshp * eval_select_z - eval_select_y
        y_upper = eval_select_y - constr_upper_py_rshp * eval_select_z
        z_lower = eval_select_z
        z_upper = eval_select_z

        constr_A = np.concatenate((x_lower, x_upper, y_lower, y_upper, z_lower), axis=0)
        constr_l = np.ones(num_constr//3*5) * (-np.inf)
        constr_l[-num_constr//3:] = constr_lower_d
        constr_u = np.zeros_like(constr_l)
        constr_u[-num_constr//3:] = constr_upper_d
        solver.setup(P=csc_matrix(loss_coeff), q=np.zeros(num_ctrl*3), A=csc_matrix(constr_A), l=constr_l, u=constr_u, verbose=False)
        if init_guess is not None:
            solver.warm_start(x=init_guess)
        result = solver.solve()
        return result.x

    for i in range(5):
        if i == 0:
            knots, keypt_s = knots, keypt_u
            init_guess = None
        else:
            new_spline, knots, keypt_s = reparam(spline, keypt_u)
            init_guess = new_spline.c.flatten()
        
        qp_out = QP_step(init_guess, knots, keypt_s)
        new_ctrl = qp_out.reshape(num_ctrl, 3)
        new_spline = interp.BSpline(knots, new_ctrl, k)

        spline, keypt_u = new_spline, keypt_s
    
    # Assign reliability values w/ a gaussian
    bounds = constr_upper_d-constr_lower_d
    cutoff = 3
    sigma = 5
    clipped_bounds = np.clip(bounds, a_min=cutoff, a_max=None)
    reliability_bounds = gaussian(clipped_bounds, cutoff, sigma) / (gaussian(cutoff, cutoff, sigma) + 1e-3)
    keypt_s[0], keypt_s[-1] = 0.0, 1.0
    reliability = interp.interp1d(keypt_s, reliability_bounds)
    
    return spline, reliability

def optim_init(init_pts, keypoints, keypoint_idxs, order, cam2img):
    # Construct bounds
    lower = np.zeros((len(order), 3))
    upper = np.zeros((len(order), 3))
    fit_rad = keypoints.shape[0] // 10
    bound_rads = []
    bound_thresh = 1e-5
    lowest_nonzero = np.inf
    key_line_pts = np.zeros(len(order))
    for key_ord, key_id in enumerate(order):
        # Fit line to range of points
        start = max(key_ord-fit_rad, 0)
        end = min(key_ord+fit_rad, len(order)-1)
        if start == 0:
            end = min(end+fit_rad//2, len(order)-1)
        elif end == len(order)-1:
            start = max(start-fit_rad//2, 0)
        x = np.arange(keypoint_idxs[start], keypoint_idxs[end]+1)
        data = init_pts[x, 2]
        slope, intercept, *_ = scipy.stats.linregress(x, data)

        # Construct bound radius from current point
        line_pts = slope * x + intercept
        curr_line_pt = slope * keypoint_idxs[key_ord] + intercept
        key_line_pts[key_ord] = curr_line_pt
        line_std = np.mean((data - line_pts)**2) ** (1/2)
        bound_rad = np.abs(keypoints[key_id, 2] - curr_line_pt) * 1.5
        bound_rad = max(bound_rad, line_std)
        bound_rads.append(bound_rad)
        if bound_rad > bound_thresh:
            lowest_nonzero = min(lowest_nonzero, bound_rad)
    
    # Set bounds, accounting for minimum radius threshold
    for key_ord, (key_id, bound_rad) in enumerate(zip(order, bound_rads)):
        if bound_rad <= bound_thresh:
            bound_rad = lowest_nonzero
        lower[key_ord] = keypoints[key_id] - np.array([[0, 0, bound_rad]])
        upper[key_ord] = keypoints[key_id] + np.array([[0, 0, bound_rad]])

    # Get endpoint trends to get endpoint constraints
    # TODO is second iteration of line fitting necessary?
    endpts = []
    slopes = []
    endpt_ids = [order[0], order[-1]]
    endpt_ords = [0, len(order)-1]
    for key_ord, key_id in zip(endpt_ords, endpt_ids):
        # Fit line to range of points
        start = max(key_ord-fit_rad, 0)
        end = min(key_ord+fit_rad, len(order)-1)
        if start == 0:
            end = min(end+fit_rad//2, len(order)-1)
        elif end == len(order)-1:
            start = max(start-fit_rad//2, 0)
        x = np.arange(keypoint_idxs[start], keypoint_idxs[end]+1)
        data = init_pts[x, 2]
        slope, intercept, *_ = scipy.stats.linregress(x, data)
        if key_ord == 0:
            endpts.append(intercept)
            slopes.append(slope)
        elif key_ord == len(order) - 1:
            endpts.append(slope*x[-1] + intercept)
            slopes.append(slope)
    init_pts[0, 2] = endpts[0]
    init_pts[-1, 2] = endpts[-1]

    # Bring points into camera coords
    init_pts = change_coords(init_pts, cam2img)

    # initialize 3D spline
    k = 3
    num_ctrl = 20

    dists = np.linalg.norm(init_pts[1:] - init_pts[:-1], axis=1)
    dists /= np.sum(dists)
    u = np.zeros(init_pts.shape[0])
    u[1:] = np.cumsum(dists)
    u[-1] = 1

    knots = np.concatenate(
        (np.repeat(0, k),
        np.linspace(0, u[-1], num_ctrl),
        np.repeat(u[-1], k))
    )

    return knots, u, lower[:, 2], upper[:, 2]

def get_deriv_matrix(knots, num_ctrl, k):
    mat = np.zeros((3*(num_ctrl-1), 3*num_ctrl))
    for i in range(num_ctrl-1):
        coeff = k / (knots[i+k+1] - knots[i+1])
        
        mat[3*i, 3*i] = -coeff
        mat[3*i+1, 3*i+1] = -coeff
        mat[3*i+2, 3*i+2] = -coeff

        mat[3*i, 3*(i+1)] = coeff
        mat[3*i+1, 3*(i+1)+1] = coeff
        mat[3*i+2, 3*(i+1)+2] = coeff
    
    return mat