import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import numpy as np
from DSA_copy.nurbs_dsa.nurbs_eval import BasisFunc
from pixel_ordering import order_pixels
from tqdm import tqdm

if __name__ == "__main__":
    # img = mpimg.imread("/Users/neelay/ARClabXtra/Sarah_imgs/thread_1_left_rembg.png")
    # plt.imshow(img)

    # img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    # img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    # img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    # control_pts = torch.tensor([
    #     [259, 337],
    #     [299, 296],
    #     [341, 354],
    #     [336, 421],
    #     [297, 460],
    #     [235, 464],
    #     [186, 444],
    #     [146, 403],
    #     [119, 348],
    #     [100, 303],
    #     [88, 245],
    #     [108, 212],
    #     [149, 191],
    #     [188, 180],
    #     [231, 168],
    #     [292, 157],
    #     [354, 136],
    #     [394, 123],
    #     [391, 136],
    #     [372, 133]
    # ], dtype=torch.float32, requires_grad=True)

    # target_pts = torch.tensor([
    #     [259, 337],
    #     [260, 312],
    #     [279, 296],
    #     [299, 296],
    #     [318, 310],
    #     [331, 329],
    #     [341, 354],
    #     [343, 377],
    #     [341, 395],
    #     [336, 421],
    #     [328, 437],
    #     [315, 450],
    #     [297, 460],
    #     [278, 465],
    #     [258, 467],
    #     [235, 464],
    #     [220, 461],
    #     [202, 453],
    #     [186, 444],
    #     [172, 434],
    #     [159, 419],
    #     [146, 403],
    #     [136, 390],
    #     [128, 373],
    #     [119, 348],
    #     [115, 334],
    #     [107, 319],
    #     [100, 303],
    #     [94, 287],
    #     [90, 263],
    #     [88, 245],
    #     [91, 232],
    #     [98, 222],
    #     [108, 212],
    #     [124, 205],
    #     [137, 197],
    #     [149, 191],
    #     [165, 185],
    #     [176, 183],
    #     [188, 180],
    #     [204, 176],
    #     [219, 171],
    #     [231, 168],
    #     [254, 165],
    #     [275, 161],
    #     [292, 157],
    #     [312, 150],
    #     [333, 143],
    #     [354, 136],
    #     [370, 131],
    #     [384, 127],
    #     [394, 123],
    #     [407, 125],
    #     [403, 132],
    #     [391, 136],
    #     [384, 138],
    #     #,
    #     [372, 133]
    # ], dtype=torch.float32)
    target_pts = torch.tensor(order_pixels(), dtype=torch.float32)
    target_pts = torch.stack((target_pts[1], target_pts[0]))
    
    u = torch.linspace(1e-5, 1.0-1e-5, steps=target_pts.size(0), dtype=torch.float32)
    u = u.unsqueeze(0)

    p = 3
    n = 20
    control_pts = torch.stack(
        (
            torch.tensor([target_pts[0, i] for i in range(0, target_pts.size(1), target_pts.size(1) // n)]),
            torch.tensor([target_pts[1, i] for i in range(0, target_pts.size(1), target_pts.size(1) // n)])
        )
    )[:, :-1]
    for_test = control_pts.numpy()
    # control_pts = control_pts.type(torch.float32)
    control_pts.requires_grad = True

    knot_int_u = torch.ones(n+p+1-2*p-1).unsqueeze(0)
    knot_int_u.requires_grad = True

    with torch.no_grad():
        plt.scatter(target_pts[1], target_pts[0])
        plt.scatter(control_pts[1], control_pts[0])
        plt.show()

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

            ctrl_pts = torch.cat((control_pts, weights), dim=-1)


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