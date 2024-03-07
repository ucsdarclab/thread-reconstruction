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
    segpix1 = np.argwhere(mask1>0)
    # init_pts, keypoint_idxs = augment_keypoints(img1, segpix1, img_3D, keypoints, grow_paths, order)
    init_pts, keypoint_idxs = keypoints[order], np.arange(len(order))
    spline, init_u, low_constr, high_constr = optim_init(init_pts, keypoints, keypoint_idxs, order, cam2img)
    keypt_u = init_u[keypoint_idxs]
    k = 3
    num_ctrl = len(spline.t)-k-1
    num_constr = len(keypt_u)*3

    constr_centers = keypoints[order]
    constr_lower_px = constr_centers[:, 1] - CONSTR_WIDTH_2D
    constr_upper_px = constr_centers[:, 1] + CONSTR_WIDTH_2D
    constr_lower_py = constr_centers[:, 0] - CONSTR_WIDTH_2D
    constr_upper_py = constr_centers[:, 0] + CONSTR_WIDTH_2D
    constr_lower_d = low_constr(keypt_u)
    constr_upper_d = high_constr(keypt_u)

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
            knots, keypt_s = spline.t, keypt_u
            init_guess = None
        else:
            new_spline, knots, keypt_s = reparam(spline, keypt_u)
            
            param_change = np.abs(np.array(keypt_s) - np.array(keypt_u))
            print(np.mean(param_change), np.std(param_change))

            init_guess = new_spline.c.flatten()
        
        qp_out = QP_step(init_guess, knots, keypt_s)
        new_ctrl = qp_out.reshape(num_ctrl, 3)
        new_spline = interp.BSpline(knots, new_ctrl, k)

        old_samples = spline(np.linspace(0, keypt_u[-1], 150))
        new_samples = new_spline(np.linspace(0, knots[-1], 150))

        num_eval_pts = 200
        left, left_max = reprojection_error(new_spline, mask1, P1, num_eval_pts)
        right, right_max = reprojection_error(new_spline, mask2, P2, num_eval_pts)
        # print("Reprojection error: mean left %f, max left %f, mean right %f, max right %f" \
        #     % (left, left_max, right, right_max))

        # plt.imshow(img1, cmap="gray")
        ax = plt.subplot(projection="3d")
        ax.plot(old_samples[:, 0], old_samples[:, 1], old_samples[:, 2])
        ax.plot(new_samples[:, 0], new_samples[:, 1], new_samples[:, 2])
        ax.scatter(new_ctrl[..., 0], new_ctrl[..., 1], new_ctrl[..., 2])
        ax.plot(new_spline(keypt_s)[:, 0], new_spline(keypt_s)[:, 1], constr_lower_d, c="turquoise")
        ax.plot(new_spline(keypt_s)[:, 0], new_spline(keypt_s)[:, 1], constr_upper_d, c="turquoise")
        # ax.scatter(init_pts[:, 0], init_pts[:, 1], init_pts[:, 2],\
        #         c=init_s, cmap="hot")
        # plt.axis("equal")
        set_axes_equal(ax)
        plt.show()

        spline, keypt_u = new_spline, keypt_s
    return spline
    

def augment_keypoints(img1, segpix1, img_3D, keypoints, grow_paths, order):
    # Gather more points between keypoints to get better data for curve initialization
    init_pts = []
    size_thresh = segpix1.shape[0] // 100
    ang_thresh = np.pi/5
    interval_floor = size_thresh // 2
    keypoint_idxs = []
    for key_ord, key_id in enumerate(order[:-1]):
        # Find segmented points between keypoints
        keypoint_idxs.append(len(init_pts))
        init_pts.append(keypoints[key_id])
        curr_growth = grow_paths[key_id]
        next_id = order[key_ord+1]
        next_growth = grow_paths[next_id]
        btwn_pts = curr_growth.intersection(next_growth)

        # gather extra points if keypoint distance is large
        if len(btwn_pts) > size_thresh:
            btwn_pts = np.array(list(btwn_pts))
            btwn_depths = img_3D[btwn_pts[:, 0], btwn_pts[:, 1], 2]
            num_samples = btwn_pts.shape[0] // size_thresh
            # remove outliers
            quartiles = np.percentile(btwn_depths, [25, 75])
            iqr = quartiles[1] - quartiles[0]
            low_clip = quartiles[0]-1.5*iqr < btwn_depths
            up_clip = btwn_depths < quartiles[1]+1.5*iqr
            mask = np.logical_and(low_clip, up_clip)
            mask_idxs = np.squeeze(np.argwhere(mask))
            if mask_idxs.shape[0] < num_samples:
                continue
            filtered_pix = btwn_pts[mask_idxs]
            filtered_depths = btwn_depths[mask_idxs]
            filtered_pts = np.concatenate((filtered_pix, np.expand_dims(filtered_depths, 1)), axis=1)
            
            # project filtered points onto 2D line between keypoints
            p1 = keypoints[key_id, :2]
            p2 = keypoints[next_id, :2]
            p1p2 = p2 - p1
            p1pt = filtered_pix - np.expand_dims(p1, 0)
            p2pt = filtered_pix - np.expand_dims(p2, 0)
            proj1 = np.dot(p1pt, p1p2) / np.linalg.norm(p1p2)
            proj2 = np.dot(p2pt, -1*p1p2) / np.linalg.norm(p1p2)

            # Use angle to prune away more points
            ang1 = np.arccos(proj1 / (np.linalg.norm(p1pt, axis=1)+1e-7))
            ang2 = np.arccos(proj2 / (np.linalg.norm(p2pt, axis=1)+1e-7))
            mask1 = ang1 < ang_thresh
            mask2 = ang2 < ang_thresh
            mask = np.logical_and(mask1, mask2)
            mask_idxs = np.atleast_1d(np.squeeze(np.argwhere(mask)))
            if mask_idxs.shape[0] < num_samples:
                continue
            filtered_pix = filtered_pix[mask_idxs]
            filtered_depths = filtered_depths[mask_idxs]
            filtered_pts = filtered_pts[mask_idxs]
            proj = proj1[mask_idxs]

            # Choose evenly spaced points, based on projections
            pt2ord = np.argsort(proj)
            floor = interval_floor if interval_floor<np.max(proj) else max(np.min(proj), 0)
            intervals = np.linspace(interval_floor, np.max(proj), num_samples)
            int_idx = 0
            for pt_idx in pt2ord:
                if int_idx >= num_samples or \
                    (filtered_pix[pt_idx] == keypoints[next_id, :2]).all():
                    break
                if proj[pt_idx] >= intervals[int_idx]:
                    init_pts.append(filtered_pts[pt_idx])
                    int_idx += 1
    keypoint_idxs.append(len(init_pts))
    init_pts.append(keypoints[order[-1]])
    init_pts = np.array(init_pts)

    return init_pts, keypoint_idxs

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

    # initialize 3D spline
    k = 3
    num_ctrl = 20

    dists = np.linalg.norm(init_pts[1:, :2] - init_pts[:-1, :2], axis=1)
    dists /= np.sum(dists)
    u = np.zeros(init_pts.shape[0])
    u[1:] = np.cumsum(dists)
    u[-1] = 1

    key_weight = init_pts.shape[0] / keypoints.shape[0]
    w = np.ones_like(u)
    w[keypoint_idxs] = key_weight
    knots = np.concatenate(
        (np.repeat(0, k),
        np.linspace(0, u[-1], num_ctrl),
        np.repeat(u[-1], k))
    )

    # Set depths to follow centerline between boundaries
    low_constr = interp.interp1d(u[keypoint_idxs], lower[:, 2])
    high_constr = interp.interp1d(u[keypoint_idxs], upper[:, 2])
    center = (low_constr(u) + high_constr(u))/2
    init_pts[:, 2] = center

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

    # Fit spline to initial points
    tck, u, *_ = interp.splprep(init_pts.T, w=w, u=u, k=k, task=-1, t=knots)
    t = tck[0]
    c = np.array(tck[1]).T
    k = tck[2]
    spline = interp.BSpline(t, c, k)
    return spline, u, low_constr, high_constr

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