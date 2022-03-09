import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import numpy as np
from DSA_copy.nurbs_dsa.nurbs_eval import BasisFunc
from tqdm import tqdm

if __name__ == "__main__":
    img = mpimg.imread("/Users/neelay/ARClabXtra/Sarah_imgs/thread_1_left_rembg.png")
    plt.imshow(img)

    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    u = torch.linspace(1e-5, 1.0-1e-5, steps=64, dtype=torch.float32)
    u = u.unsqueeze(0)

    p = 3
    n = 20

    knot_int_u = torch.ones(n+p+1-2*p-1).unsqueeze(0)
    knot_int_u.requires_grad = True

    control_pts = torch.tensor([
        [259, 337],
        [299, 296],
        [341, 354],
        [336, 421],
        [297, 460],
        [235, 464],
        [186, 444],
        [146, 403],
        [119, 348],
        [100, 303],
        [88, 245],
        [108, 212],
        [149, 191],
        [188, 180],
        [231, 168],
        [292, 157],
        [354, 136],
        [394, 123],
        [391, 136],
        [372, 133]
    ], dtype=torch.float32, requires_grad=True)

    target_pts = torch.tensor([
        [259, 337],
        ,
        ,
        [299, 296],
        ,
        ,
        [341, 354],
        ,
        ,
        [336, 421],
        ,
        ,
        [297, 460],
        ,
        ,
        [235, 464],
        ,
        ,
        [186, 444],
        ,
        ,
        [146, 403],
        ,
        ,
        [119, 348],
        ,
        ,
        [100, 303],
        ,
        ,
        [88, 245],
        ,
        ,
        [108, 212],
        ,
        ,
        [149, 191],
        ,
        ,
        [188, 180],
        ,
        ,
        [231, 168],
        ,
        ,
        [292, 157],
        ,
        ,
        [354, 136],
        ,
        ,
        [394, 123],
        ,
        ,
        [391, 136],
        ,
        ,
        [372, 133]
    ], dtype=torch.float32)

    weights = torch.ones(n, requires_grad=True)
    initial_curve = None
    final_curve = None
    
    opt_1 = torch.optim.LBFGS(iter([control_pts, weights]), lr=0.5, max_iter=3) #lr = 0.1
    opt_2 = torch.optim.SGD(iter([knot_int_u]), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt_2, milestones=[15, 50, 200], gamma=0.01)

    num_iter = 100 #200
    learn_partition = 20 #30
    pbar = tqdm(range(num_iter))
    for j in pbar:
        knot_rep_p_0 = torch.zeros(1,p+1)
        knot_rep_p_1 = torch.zeros(1,p)

        def closure():
            global initial_curve
            global final_curve
            opt_1.zero_grad()
            opt_2.zero_grad()

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

            ctrl_pts = torch.cat((control_pts, weights.unsqueeze(-1)), dim=-1)


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
                if initial_curve == None:
                    initial_curve = curve.clone()
                else:
                    final_curve = curve

            loss = ((target-curve)**2).mean()
            loss.backward(retain_graph=True)
            return loss

        if (j % 100) < learn_partition:
            loss = opt_1.step(closure)
        else:
            loss = opt_2.step(closure)
    
    plt.show()