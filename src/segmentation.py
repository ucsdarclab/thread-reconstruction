import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as manimation
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import skimage.morphology as skimage_morphology

def segmentation(img):
    # Convolve with gaussian second derivative
    pass

def get_raw_centerline(img_ref):
    ## https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line

    # img_height = img_ref.shape[0]
    # img_width = img_ref.shape[1]
    img_ref = img_ref.copy()
    img_ref = np.where(img_ref<255//2, 1, 0)

    skeleton = skimage_morphology.skeletonize(img_ref)
    img_raw_skeleton = np.argwhere(skeleton == 1)

    # creating a nearest neighbour graph to connect each of the nodes to its 2 nearest neighbors
    # neigh = NearestNeighbors(n_neighbors=2, radius=0.4)

    clf = NearestNeighbors(n_neighbors=2).fit(img_raw_skeleton)
    G = clf.kneighbors_graph()

    # then use networkx to construct a graph from this sparse matrix
    T = nx.from_scipy_sparse_matrix(G)

    # find shortest path from source
    # minimizes the distances between the connections (optimization problem):
    min_dists = []
    min_idxs = []
    opt_skeletons_ordered = []
    opt_skeletons_sets = []


    for i in range(img_raw_skeleton.shape[0]):
        path = list(nx.dfs_preorder_nodes(T, i))
        ordered = img_raw_skeleton[path]  # ordered nodes

        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if len(opt_skeletons_ordered) == 0:
            min_dists.append(cost)
            min_idxs.append(i)
            opt_skeletons_ordered.append(ordered)
        exists = False
        for j in range(len(opt_skeletons_ordered)):
            intersections = [elem in opt_skeletons_ordered[j] for elem in ordered]
            if sum(intersections) > 0:
                exists = True
                if cost < min_dists[j]:
                    min_dists[j] = cost
                    min_idxs[j] = i
                    opt_skeletons_ordered[j] = ordered
                    # opt_skeletons_sets[j] = skel_set
                break
        if not exists:
            min_dists.append(cost)
            min_idxs.append(i)
            opt_skeletons_ordered.append(ordered)
            # opt_skeletons_sets.append(skel_set)

    ### this can gurantee the starting point of the skeleton is always from top-->bottom
    # if opt_skeleton_ordered[0, 0] > 50:
    #     opt_skeleton_ordered = np.flip(opt_skeleton_ordered, 0)

    ### this will flip x/y coordinates in order to fit bezier_proj_img
    # opt_skeleton_ordered = np.stack((opt_skeleton_ordered[:, 1], opt_skeleton_ordered[:, 0]), axis=1)

    return opt_skeletons_ordered

if __name__ == "__main__":
    folder_num = 2
    file_num = 40
    file1 = "../Suture_Thread_06_16/thread_%d_seg_new/left_recif_%d.png" % (folder_num, file_num)
    file2 = "../Suture_Thread_06_16/thread_%d_seg_new/right_recif_%d.png" % (folder_num, file_num)
    calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = np.where(img1 > 0, 0, 255)
    img2 = np.where(img2 > 0, 0, 255)

    # plt.figure(1)
    # plt.imshow(img1, cmap="gray")
    # plt.figure(2)
    # plt.imshow(img2, cmap="gray")
    # plt.show()

    centerlines1 = get_raw_centerline(img1)
    # centerline1[:, 0], centerline1[:, 1] = centerline1[:, 1].copy(), centerline1[:, 0].copy()
    centerlines2 = get_raw_centerline(img2)
    # centerline2[:, 0], centerline2[:, 1] = centerline2[:, 1].copy(), centerline2[:, 0].copy()

    plt.figure(1)
    plt.imshow(img1, cmap="gray")
    for centerline1 in centerlines1:
        plt.plot(centerline1[:, 1], centerline1[:, 0])
    plt.figure(2)
    plt.imshow(img2, cmap="gray")
    for centerline2 in centerlines2:
        plt.plot(centerline2[:, 1], centerline2[:, 0])
    plt.show()

    # print("creating video -- this can take a few minutes")
    # FFMpegWriter = manimation.writers["ffmpeg"]
    # fps = 10
    # metadata = dict(title="Frame Change", artist="NeelayJ", comment="Result of subtracting consecutive frames")
    # writer = FFMpegWriter(fps=fps, metadata=metadata)
    # fig = plt.figure()
    # plt.figure(fig)
    # with writer.saving(fig, "change1.mp4", dpi=200):
    #     for file_num in range(1259):
    #         file1 = "../Suture_Thread_06_16/thread_%d/left_recif_%d.jpg" % (folder_num, file_num)
    #         file2 = "../Suture_Thread_06_16/thread_%d/right_recif_%d.jpg" % (folder_num, file_num)
    #         calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
    #         img1 = cv2.imread(file1)
    #         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    #         img2 = cv2.imread(file2)
    #         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    #         img1 = np.float32(img1)
    #         img2 = np.float32(img2)

    #         img1_d = img1 - img1_z
    #         img2_d = img2 - img2_z

    #         img1_d = np.abs(np.int32(img1_d))
    #         img2_d = np.abs(np.int32(img2_d))

    #         plt.clf()
    #         plt.imshow(img1_d)
    #         plt.title("Frame %d" %(file_num,))
    #         # plt.show()
    #         writer.grab_frame()

    #         img1_z = img1
    #         img2_z = img2
    # plt.close(fig)