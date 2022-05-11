import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import numpy as np
from DSA_copy.nurbs_dsa.nurbs_eval import BasisFunc
from pixel_ordering import order_pixels
from stereo_matching import stereo_match
from tqdm import tqdm

TESTING = True
"""
Much of this code is based off of this repo:
https://github.com/idealab-isu/DSA
"""
def fit_2D_curves():
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(img_dir + "thread_1_right_rembg.png")
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ordered_pixels_l, ordered_pixels_r = order_pixels()

    # initialize shared constants between images
    eval_num = 150

    u = torch.linspace(1e-5, 1.0-1e-5, steps=eval_num, dtype=torch.float32)
    u = u.unsqueeze(0)
    p = 3
    n = 30
    knot_int_u = torch.ones(n+p+1-2*p-1).unsqueeze(0)
    knot_rep_p_0 = torch.zeros(1,p+1)
    knot_rep_p_1 = torch.zeros(1,p)
    knot_u = torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1)
    weights = torch.ones((1,n))
    
    init_control_pts = [None, None]
    initial_curve = [None, None]
    final_curve = [None, None]

    num_iter = 30
    learn_partition = num_iter

    # fit curves to both left and right orderings
    for which_curve, (ordered_pixels, img) in enumerate(zip([ordered_pixels_l, ordered_pixels_r], [img_l, img_r])):
        ordered_pixels = torch.tensor(ordered_pixels, dtype=torch.float32)
        indicies = [(i * ordered_pixels.size(1)) // eval_num for i in range(eval_num)]

        # Get pixel ordering for ground truth
        target = torch.stack(
            (
                torch.tensor([ordered_pixels[1, i] for i in indicies]),
                torch.tensor([ordered_pixels[0, i] for i in indicies])
            )
        )
        
        # initialize control points from ground truth
        indicies = [(i * target.size(1)) // n for i in range(n)]
        control_pts = torch.stack(
            (
                torch.tensor([target[0, i] for i in indicies]),
                torch.tensor([target[1, i] for i in indicies])
            )
        )
        if TESTING:
            init_control_pts[which_curve] = control_pts.clone()
        control_pts.requires_grad = True


        """
        TODO Uncomment if DSA used on knot vector
        Make sure to update opts as necessary as well
        """
        # knot_int_u = torch.ones(n+p+1-2*p-1).unsqueeze(0)
        # knot_int_u.requires_grad = True
        # weights = torch.ones((1,n), requires_grad=True)
        # opt_2 = torch.optim.SGD(iter([knot_int_u]), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_2, milestones=[15, 50, 200], gamma=0.01)        
        
        opt_1 = torch.optim.LBFGS(iter([control_pts]), lr=0.4, max_iter=3)
        
        # pbar = tqdm(range(num_iter))
        for j in range(num_iter):#pbar:
            def closure():
                opt_1.zero_grad()
                # opt_2.zero_grad()

                # knot_u = torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1)

                U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
                U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

                uspan_uv = torch.stack([torch.min(
                    torch.where(
                        (u - U[s,p:-p].unsqueeze(1))>1e-8,
                        u - U[s,p:-p].unsqueeze(1),
                        (u - U[s,p:-p].unsqueeze(1))*0.0 + 1
                    ),
                    0,
                    keepdim=False
                )[1]+p for s in range(U.size(0))])

                Nu_uv = BasisFunc.apply(u, U, uspan_uv, p).squeeze()

                ctrl_pts = torch.cat((control_pts, weights), dim=0).permute(1, 0)

                pts = torch.stack([ctrl_pts[uspan_uv[0,:] - p + l, :] for l in range(p + 1)])

                weighted_pts = torch.stack([
                    pts[:, :, i] * pts[:, :, pts.size(2) - 1]
                for i in range(pts.size(2) - 1)]).permute(1, 2, 0)

                curve_num = torch.stack([
                    torch.sum(
                        Nu_uv * weighted_pts[:, :, i],
                        dim=0
                    )
                for i in range(pts.size(2) - 1)])

                curve_denom = torch.sum(Nu_uv * pts[:, :, pts.size(2) - 1], dim=0)

                curve = torch.stack([
                    curve_num[i,:] / curve_denom
                for i in range(pts.size(2) - 1)])
                with torch.no_grad():
                    if TESTING and initial_curve[which_curve] == None:
                        initial_curve[which_curve] = curve.clone()
                    else:
                        final_curve[which_curve] = curve

                loss = ((target-curve)**2).mean()
                loss.backward(retain_graph=True)
                return loss

            if (j % 100) < learn_partition:
                loss = opt_1.step(closure)
            else:
                # loss = opt_2.step(closure)
                pass
            
            # with torch.no_grad():
            #     weights = weights.clamp(1e-8)
            #     knot_int_u = knot_int_u.clamp(1e-8)
    
        if TESTING:
            with torch.no_grad():
                fig = plt.figure(figsize=(8, 4.8))
                fig.clf()
                pre = fig.add_subplot(121)
                pre.imshow(img, cmap="gray")
                # pre.set_xlim(0, 600)
                # pre.set_ylim(0, 600)
                pre.set_title("Initial spline")
                post = fig.add_subplot(122)
                post.imshow(img, cmap="gray")
                # post.set_xlim(0, 600)
                # post.set_ylim(0, 600)
                post.set_title("Final spline")

                # pre.plot(target[1], target[0], color="b")

                pre.scatter(init_control_pts[which_curve][1], init_control_pts[which_curve][0], color="r", s=2)
                pre.plot(initial_curve[which_curve][1], initial_curve[which_curve][0], color="g")

                # post.plot(target[1], target[0], color="b")

                post.scatter(control_pts[1], control_pts[0], color="r", s=2)
                post.plot(final_curve[which_curve][1], final_curve[which_curve][0], color="g")

                plt.show()
                # For debugging
                # test_tar = target.numpy()
                # test_ctrl = control_pts.numpy()
                # sdFASDFASDF = 1
    
    spacings = [
        (final_curve[0][:, 1:] - final_curve[0][:, :-1]).detach().numpy(),
        (final_curve[1][:, 1:] - final_curve[1][:, :-1]).detach().numpy()
    ]
    spacings = [
        np.linalg.norm(spacings[0], axis=0),
        np.linalg.norm(spacings[1], axis=0)
    ]
    print([np.min(spacings[0]), np.min(spacings[1])])
    print([np.max(spacings[0]), np.max(spacings[1])])
    return final_curve[0].detach(), final_curve[1].detach()

def fit_3D_curve():
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(img_dir + "thread_1_right_rembg.png")
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    P1, P2, points_3D = stereo_match()
    P1 = torch.tensor(P1, dtype=torch.float32)
    P2 = torch.tensor(P2, dtype=torch.float32)
    # in x, y, z order
    points_3D = torch.tensor(points_3D).permute(1, 0)

    # Initialize useful evaluation constants
    init_control_pts = None
    init_ctrl_l = None
    init_ctrl_r = None
    final_ctrl_l = None
    final_ctrl_r = None
    initial_curve = None
    init_curve_l = None
    init_curve_r = None
    final_curve = None
    final_curve_l = None
    final_curve_r = None

    eval_3D = 600
    eval_2D = 150
    
    # Calculate spline generation constants
    p = 3
    n = 40

    u = torch.linspace(1e-5, 1.0-1e-5, steps=eval_3D, dtype=torch.float32)
    u = u.unsqueeze(0)

    knot_int_u = torch.ones(n+p+1-2*p-1).unsqueeze(0)
    knot_rep_p_0 = torch.zeros(1,p+1)
    knot_rep_p_1 = torch.zeros(1,p)
    knot_u = torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1)

    U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
    U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    uspan_uv = torch.stack([torch.min(
        torch.where(
            (u - U[s,p:-p].unsqueeze(1))>1e-8,
            u - U[s,p:-p].unsqueeze(1),
            (u - U[s,p:-p].unsqueeze(1))*0.0 + 1
        ),
        0,
        keepdim=False
    )[1]+p for s in range(U.size(0))])

    Nu_uv = BasisFunc.apply(u, U, uspan_uv, p).squeeze()

    # Initialize control points
    indicies = [(i * points_3D.size(1)) // n for i in range(n)]
    control_pts = torch.stack(
        (
            torch.tensor([points_3D[0, i] for i in indicies]),
            torch.tensor([points_3D[1, i] for i in indicies]),
            torch.tensor([points_3D[2, i] for i in indicies])
        )
    )
    if TESTING:
        init_control_pts = control_pts.clone()
        aug_ctrl_pts = torch.cat(
            (init_control_pts, torch.ones((1, init_control_pts.size(1)))), dim=0
        ).permute(1, 0).unsqueeze(-1)
        #x, y, z
        init_ctrl_l = torch.matmul(P1, aug_ctrl_pts)
        for i in range(init_ctrl_l.size(0)):
            init_ctrl_l[i] /= init_ctrl_l[i, 2, 0].clone()
        init_ctrl_r = torch.matmul(P2, aug_ctrl_pts)
        for i in range(init_ctrl_r.size(0)):
            init_ctrl_r[i] /= init_ctrl_r[i, 2, 0].clone()
    control_pts.requires_grad = True

    # Initialize ground truth
    ord1, ord2 = order_pixels()

    ord1 = torch.tensor(ord1, dtype=torch.float32)
    ord2 = torch.tensor(ord2, dtype=torch.float32)
    idx1 = [(i * ord1.size(1)) // eval_2D for i in range(eval_2D)]
    idx2 = [(i * ord2.size(1)) // eval_2D for i in range(eval_2D)]

    #y, x
    target1 = torch.stack(
        (
            torch.tensor([ord1[1, i] for i in idx1]),
            torch.tensor([ord1[0, i] for i in idx1])
        )
    ).permute(1, 0)
    target2 = torch.stack(
        (
            torch.tensor([ord2[1, i] for i in idx2]),
            torch.tensor([ord2[0, i] for i in idx2])
        )
    ).permute(1, 0)

    # spacings = [
    #     target1[:, 1:] - target1[:, :-1],
    #     target2[:, 1:] - target2[:, :-1]
    # ]
    # spacings = [
    #     np.linalg.norm(spacings[0], axis=0),
    #     np.linalg.norm(spacings[1], axis=0)
    # ]
    # print([np.min(spacings[0]), np.min(spacings[1])])
    # print([np.max(spacings[0]), np.max(spacings[1])])

    # Peform grad descent
    num_iter = 5
    opt = torch.optim.LBFGS(iter([control_pts]), lr=0.4, max_iter=3)

    
    for j in tqdm(range(num_iter)):#pbar:
        def closure():
            nonlocal initial_curve
            nonlocal init_curve_l
            nonlocal init_curve_r
            nonlocal final_curve
            nonlocal final_curve_l
            nonlocal final_curve_r
            nonlocal final_ctrl_l
            nonlocal final_ctrl_r
            opt.zero_grad()

            ctrl_pts = control_pts.permute(1, 0)
            pts = torch.stack([ctrl_pts[uspan_uv[0,:] - p + l, :] for l in range(p + 1)])

            curve = torch.stack([
                torch.sum(
                    Nu_uv * pts[:, :, i],
                    dim=0
                )
            for i in range(pts.size(2))])

            with torch.no_grad():
                if TESTING and initial_curve == None:
                    initial_curve = curve.clone()
                else:
                    final_curve = curve

            #x, y, z 
            curve = torch.cat((curve, torch.ones((1, curve.size(1)))), dim=0)
            #x, y, z
            proj1 = torch.matmul(P1, curve.permute(1, 0).unsqueeze(-1))
            for i in range(proj1.size(0)):
                proj1[i] /= proj1[i, 2, 0].clone()
            proj2 = torch.matmul(P2, curve.permute(1, 0).unsqueeze(-1))
            for i in range(proj2.size(0)):
                proj2[i] /= proj2[i, 2, 0].clone()
            
            # with torch.no_grad():
            #     fig = plt.figure(figsize=(4, 6))
            #     ax1 = fig.add_subplot(3, 2, (1, 4), projection="3d")
            #     ax1.view_init(0, 0)
            #     ax1.plot(curve[1], curve[0], curve[2])

            #     ax2 = fig.add_subplot(325)
            #     ax2.imshow(img_l, cmap="gray")
            #     ax2.plot(proj1[:, 0, 0], proj1[:, 1, 0])

            #     ax3 = fig.add_subplot(326)
            #     ax3.imshow(img_r, cmap="gray")
            #     ax3.plot(proj2[:, 0, 0], proj2[:, 1, 0])
            #     plt.show()
            #     exit(0)
            proj1 = proj1[:, :2, 0]
            proj2 = proj2[:, :2, 0]

            with torch.no_grad():
                if TESTING and init_curve_l == None:
                    init_curve_l = proj1.clone()
                    init_curve_r = proj2.clone()
                else:
                    final_curve_l = proj1
                    final_curve_r = proj2

                    aug_ctrl_pts = torch.cat(
                        (control_pts, torch.ones((1, control_pts.size(1)))), dim=0
                    ).permute(1, 0).unsqueeze(-1)
                    #x, y, z
                    final_ctrl_l = torch.matmul(P1, aug_ctrl_pts)
                    for i in range(final_ctrl_l.size(0)):
                        final_ctrl_l[i] /= final_ctrl_l[i, 2, 0].clone()
                    final_ctrl_r = torch.matmul(P2, aug_ctrl_pts)
                    for i in range(final_ctrl_r.size(0)):
                        final_ctrl_r[i] /= final_ctrl_r[i, 2, 0].clone()



            # get dists for left projection
            dists_l = torch.zeros(eval_2D)
            for i, pix1 in enumerate(target1):
                # bound a sliding window of size 4
                min_idx = max(0, i*4 - 5)
                max_idx = min(eval_3D, i*4 + 6) - 3

                min_dist = None
                for idx in range(min_idx, max_idx):
                    curr_dist = 0
                    for off in range(4):
                        curr_dist += torch.linalg.norm(pix1 - proj1[idx+off])
                    curr_dist /= 4

                    if min_dist == None:
                        min_dist = curr_dist
                    else:
                        min_dist = torch.min(min_dist, curr_dist)
                    
                dists_l[i] = min_dist
            
            # get dists for right projection
            dists_r = torch.zeros(eval_2D)
            for i, pix2 in enumerate(target2):
                # bound a sliding window of size 4
                min_idx = max(0, i*4 - 5)
                max_idx = min(eval_3D, i*4 + 6) - 3

                min_dist = None
                for idx in range(min_idx, max_idx):
                    curr_dist = 0
                    for off in range(4):
                        curr_dist += torch.linalg.norm(pix2 - proj2[idx+off])
                    curr_dist /= 4

                    if min_dist == None:
                        min_dist = curr_dist
                    else:
                        min_dist = torch.min(min_dist, curr_dist)
                    
                dists_r[i] = min_dist

            loss = dists_l.mean() + dists_r.mean()

            
            loss.backward(retain_graph=True)
            return loss

        loss = opt.step(closure)

    if TESTING:
        with torch.no_grad():
            fig = plt.figure(figsize=(10, 7))
            fig.clf()
            pre = fig.add_subplot(221, projection="3d")
            pre_l = fig.add_subplot(245)
            pre_r = fig.add_subplot(246)
            pre_l.imshow(img_l, cmap="gray")
            pre_r.imshow(img_r, cmap="gray")
            pre.set_title("Initial spline")
            
            post = fig.add_subplot(222, projection="3d")
            post_l = fig.add_subplot(247)
            post_r = fig.add_subplot(248)
            post_l.imshow(img_l, cmap="gray")
            post_r.imshow(img_r, cmap="gray")
            post.set_title("Final spline")

            pre.scatter(init_control_pts[1], init_control_pts[0], init_control_pts[2], color="r", s=2)
            pre.plot(initial_curve[1], initial_curve[0], initial_curve[2], color="g")
            pre_l.scatter(init_ctrl_l[0], init_ctrl_l[1], color="r", s=2)
            pre_l.plot(init_curve_l[0], init_curve_l[1], color="g")
            pre_r.scatter(init_ctrl_r[0], init_ctrl_r[1], color="r", s=2)
            pre_r.plot(init_curve_r[0], init_curve_r[1], color="g")

            post.scatter(control_pts[1], control_pts[0], control_pts[2], color="r", s=2)
            post.plot(final_curve[1], final_curve[0], final_curve[2], color="g")
            post_l.scatter(final_ctrl_l[0], final_ctrl_l[1], color="r", s=2)
            post_l.plot(final_curve_l[0], final_curve_l[1], color="g")
            post_r.scatter(final_ctrl_r[0], final_ctrl_r[1], color="r", s=2)
            post_r.plot(final_curve_r[0], final_curve_r[1], color="g")

            plt.show()
    

    
    


if __name__ == "__main__":
    fit_3D_curve()