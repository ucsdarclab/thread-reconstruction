import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import numpy as np
from DSA_copy.nurbs_dsa.nurbs_eval import BasisFunc
from pixel_ordering import order_pixels
from tqdm import tqdm

TESTING = False
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
                if TESTING:
                    with torch.no_grad():
                        if initial_curve[which_curve] == None:
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
    return final_curve[0], final_curve[1]

if __name__ == "__main__":
    fit_2D_curves()