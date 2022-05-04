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
    P1, P2, points_3D = stereo_match()
    # in x, y, z order
    points_3D = torch.tensor(points_3D).permute(1, 0)

    # Initialize useful evaluation constants
    init_control_pts = None
    initial_curve = None
    final_curve = None

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
    control_pts.requires_grad = True

    # Initialize ground truth
    ord1, ord2 = order_pixels()

    ord1 = torch.tensor(ord1, dtype=torch.float32)
    ord2 = torch.tensor(ord2, dtype=torch.float32)
    idx1 = [(i * ord1.size(1)) // eval_2D for i in range(eval_2D)]
    idx2 = [(i * ord2.size(1)) // eval_2D for i in range(eval_2D)]

    target1 = torch.stack(
        (
            torch.tensor([ord1[1, i] for i in idx1]),
            torch.tensor([ord1[0, i] for i in idx1])
        )
    )
    target2 = torch.stack(
        (
            torch.tensor([ord2[1, i] for i in idx2]),
            torch.tensor([ord2[0, i] for i in idx2])
        )
    )

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
    num_iter = 50
    opt = torch.optim.LBFGS(iter([control_pts]), lr=0.4, max_iter=3)

    for j in range(num_iter):#pbar:
        def closure():
            nonlocal initial_curve
            nonlocal final_curve
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

            #y, x, z 
            curve = torch.cat((curve, torch.ones((1, curve.size(1)))), dim=0)
            #x, y, z
            proj1 = torch.matmul(torch.tensor(P1, dtype=torch.float32), curve.permute(1, 0).unsqueeze(-1))
            for i in range(proj1.size(0)):
                proj1[i] /= proj1[i, 2, 0].clone()
            proj2 = torch.matmul(torch.tensor(P2, dtype=torch.float32), curve.permute(1, 0).unsqueeze(-1))
            for i in range(proj2.size(0)):
                proj2[i] /= proj2[i, 2, 0].clone()
            
            with torch.no_grad():
                plt.figure(1)
                ax1 = plt.axes(projection="3d")
                ax1.view_init(0, 0)
                ax1.plot(curve[1], curve[0], curve[2])

                plt.figure(2)
                ax2 = plt.axes()
                ax2.plot(proj1[:, 0, 0], proj1[:, 1, 0])

                plt.figure(3)
                ax3 = plt.axes()
                ax3.plot(proj2[:, 0, 0], proj2[:, 1, 0])
                plt.show()
                exit(0)
            
            loss = ((target-curve)**2).mean()
            loss.backward(retain_graph=True)
            return loss

        closure()
        return
        if (j % 100) < learn_partition:
            loss = opt_1.step(closure)
        else:
            # loss = opt_2.step(closure)
            pass
    

    
    


if __name__ == "__main__":
    fit_3D_curve()