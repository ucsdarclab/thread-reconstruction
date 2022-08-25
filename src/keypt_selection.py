import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
from matplotlib import collections
from matplotlib import colors as mcolors
import numpy as np
import cv2
from stereo_matching import stereo_match
import sys
import copy
import time

def keypt_selection(img1, img2, calib):
    pix_thresh = 240
    segpix1 = np.argwhere(img1<=pix_thresh)
    segpix2 = np.argwhere(img2<=pix_thresh)
    disp_cv, img_3D, Q = stereo_match(img1, img2, calib)
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Get reliabilities
    reliab = np.zeros(segpix1.shape[0])
    # TODO change with image size
    max_disp = 80
    rad = 2
    c_data = 5
    c_slope = 8
    c_shift = 0.8
    ignore_rad = 4
    disp_thresh = segpix1.shape[0]//400
    # TODO deal with out-of-bounds conditions
    depth_calc = np.ones((4, segpix1.shape[0]))
    for i in range(segpix1.shape[0]):
        pix = segpix1[i]
        chunk = img1[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
        seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(segpix1[i], 0) - rad

        energy = np.array([
            np.sum(
                (img1[seg[:,0], seg[:,1]] - img2[seg[:,0], seg[:,1] - off])**2
            ) for off in range(max_disp)
        ])
        
        best = np.min(energy)
        disp = np.argmin(energy)
        energy2 = np.delete(energy, slice(disp-ignore_rad, disp+ignore_rad+1))
        next_best = np.min(energy2)
        disp2 = np.argmin(energy2)
        
        x = (next_best - best)/((best + 1e-7)*c_data)
        reliab[i] = 1/(1+np.exp(np.clip(-1*c_slope*(x-c_shift), -87, None)))

        # if np.abs(disp - disp_cv[pix[0], pix[1]]) > disp_thresh:
        #     reliab[i] /= 2
        depth_calc[:, i] = np.array([pix[0], pix[1], disp, 1])
    depth_calc = np.matmul(Q, depth_calc.copy())
    depth_calc /= depth_calc[3].copy() + 1e-7
    img_3D[segpix1[:, 0], segpix1[:, 1], 2] = depth_calc[2]
    
    # prune unreliable points
    reliab_thresh = 0.9
    relidx = np.argwhere(reliab>reliab_thresh)
    relpts = segpix1[relidx[:, 0]]
    relmap = np.zeros_like(img1)
    relmap[relpts[:, 0], relpts[:, 1]] = reliab[relidx[:, 0]]

    # ax = plt.axes(projection='3d')
    # ax.view_init(0, 0)
    # ax.set_zlim(0, 1000)
    # ax.scatter(
    #     segpix1[:, 0],
    #     segpix1[:, 1],
    #     img_3D[segpix1[:, 0], segpix1[:, 1], 2],
    #     s=1, c="r")
    # ax.scatter(
    #     relpts[:, 0],
    #     relpts[:, 1],
    #     img_3D[relpts[:, 0], relpts[:, 1], 2],
    #     c="b")
    # plt.show()
    # return

    #Find Clusters
    clusters = []
    vlist = [pt for pt in relpts.copy()]
    vmap = np.ones_like(img1)
    vmap[relpts[:, 0], relpts[:, 1]] = 0
    escape = False
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                           [1, 1], [-1, -1], [-1, 1], [1, -1],
                           [2, 0], [-2, 0], [0, 2], [0, -2]])
    max_size = segpix1.shape[0] // 100
    min_size = segpix1.shape[0] // 400
    while len(vlist) > 0:
        cluster = []
        # choose source from remaining keypoint candidates
        source = vlist.pop(0)
        while vmap[source[0], source[1]] == 1:
            if len(vlist) == 0:
                escape = True
                break
            source = vlist.pop(0)
        if escape:
            break
        
        # perform BFS until max cluster size
        frontier = [source]
        vmap[source[0], source[1]] = 1
        while len(frontier) > 0 and len(cluster) <= max_size:
            curr = frontier.pop(0)
            cluster.append(curr)
            for d in DIRECTIONS:
                neigh = curr + d
                # make sure not out of bounds
                if neigh[0] < 0 or neigh[0] >= vmap.shape[0] or \
                    neigh[1] < 0 or neigh[1] >= vmap.shape[1]:
                    continue
                if vmap[neigh[0], neigh[1]] == 1:
                    continue
                frontier.append(neigh)
                vmap[neigh[0], neigh[1]] = 1
        # put frontier back into unvisited if max cluster size reached
        while len(frontier) > 0:
            curr = frontier.pop(0)
            vmap[curr[0], curr[1]] = 0
        # Ignore clusters that are too small
        if len(cluster) >= min_size:
            clusters.append(cluster)
   

    cluster_means = np.zeros((len(clusters), 3))
    cluster_stds = np.zeros((len(clusters), 3))
    cluster_sizes = np.zeros((len(clusters),))
    cluster_rels = []
    cluster_map = np.zeros_like(img1)
    all_clustered = []
    for i, cluster in enumerate(clusters):
        all_clustered += cluster
        cluster = np.array(cluster)
        cluster_means[i, :2] = np.mean(cluster, axis=0)
        cluster_means[i, 2] = np.mean(img_3D[cluster[:, 0], cluster[:, 1], 2])
        cluster_stds[i, :2] = np.std(cluster, axis=0)
        cluster_stds[i, 2] = np.std(img_3D[cluster[:, 0], cluster[:, 1], 2])
        cluster_map[cluster[:, 0], cluster[:, 1]] = i+1
        cluster_sizes[i] = len(cluster)
        cluster_rels.append(relmap[cluster[:, 0], cluster[:, 1]])
    all_clustered = np.array(all_clustered)
    all_std = np.std(img_3D[all_clustered[:, 0], all_clustered[:, 1], 2])

    # ax = plt.axes(projection='3d')
    # ax.view_init(0, 0)
    # ax.set_xlim(0, 480)
    # ax.set_ylim(0, 640)
    # ax.set_zlim(0, 20)
    # ax.scatter(
    #     segpix1[:, 0],
    #     segpix1[:, 1],
    #     img_3D[segpix1[:, 0], segpix1[:, 1], 2],
    #     s=1, c="r")
    # ax.scatter(
    #     cluster_means[:, 0],
    #     cluster_means[:, 1],
    #     cluster_means[:, 2],
    #     c="b")
    # plt.show()
    # assert False

    # "Solidify" clusters for ordering
    solid_clusters = copy.deepcopy(clusters)
    solid_map = cluster_map.copy()
    c_rad = 1
    for pix in segpix1:
        if cluster_map[pix[0], pix[1]] != 0:
            continue
        roi = cluster_map[pix[0]-c_rad:pix[0]+c_rad+1, pix[1]-c_rad:pix[1]+c_rad+1]
        near = np.argwhere(roi>0) + np.expand_dims(pix, 0) - c_rad
        if near.shape[0] > 0:
            c_id = int(cluster_map[near[0, 0], near[0, 1]])-1
            solid_clusters[c_id].append(pix)
            solid_map[pix[0], pix[1]] = c_id+1
    
    # Cluster ordering
    adjacents = [set() for _ in solid_clusters]
    grow_paths = [set() for _ in solid_clusters]
    for c_id, cluster in enumerate(solid_clusters):
        # perform DFS to find adjacent clusters
        frontier = copy.deepcopy(cluster)
        while len(frontier) > 0:
            curr = frontier.pop()
            for d in DIRECTIONS[:4]:
                neigh = curr + d
                # Check OOB condition
                if neigh[0] < 0 or neigh[0] >= solid_map.shape[0] or \
                    neigh[1] < 0 or neigh[1] >= solid_map.shape[1]:
                    continue
                t_neigh = tuple(neigh)
                neigh_clust = solid_map[t_neigh]
                # if neighbor is segmented, unvisited, and outside cluster
                if img1[neigh[0], neigh[1]] <= pix_thresh and \
                    t_neigh not in grow_paths[c_id] and \
                    neigh_clust != c_id+1:
                    # note down when adjacent keypoints are found
                    if neigh_clust != 0 :
                        adjacents[c_id].add(neigh_clust-1)
                    else:
                        frontier.append(neigh)
                        grow_paths[c_id].add(t_neigh)
    
    return img_3D, solid_clusters, solid_map, cluster_means, grow_paths, adjacents
    
    # Extract curve segments, avoiding intersections
    segments = []
    # prematurely visit intersection keypoints
    visited = [0 if len(neighs) <= 2 else 1 for neighs in adjacents]
    outer_frontier = [c_id for c_id, neighs in enumerate(adjacents) if len(neighs) <= 2]
    while True:
        # Choose an unvisited source with only 1 unvisited neighbor
        source = None
        while len(outer_frontier) > 0:
            c_id = outer_frontier.pop()
            paths = 0
            for n_id in adjacents[c_id]:
                n_id = int(n_id)
                if visited[n_id] != 1:
                    paths += 1
            if paths == 1:
                source = c_id
                break
        # exit loop when no more curve segments can grow
        if source == None:
            break
        
        # graph search to get thread
        frontier = [source]
        visited[c_id] = 1
        segment = []#cluster_means[source]]
        while len(frontier) > 0:
            assert len(frontier) == 1, "Curve is not linked list"
            curr = frontier.pop()
            segment.append(curr)#cluster_means[curr])
            for neigh in adjacents[curr]:
                neigh = int(neigh)
                if visited[neigh] != 1:
                    frontier.append(neigh)
                    visited[neigh] = 1
        segments.append(segment)
        
    # TODO
    "Join segments to form single ordering"
    return img_3D, cluster_means, grow_paths, segments[-1] 
    # plt.imshow(img1, cmap="gray")
    # plt.scatter(cluster_means[:, 1], cluster_means[:, 0])
    # for segment in segments:
    #     segment = cluster_means[np.array(segment)]
    #     plt.scatter(segment[:, 1], segment[:, 0],\
    #         c=np.arange(0, segment.shape[0]), cmap="hot")
    # plt.show()
        
    "Misc visualization code"
    # print([len(adj) for adj in adjacents])
    # lines = []
    # for c_id, adj in enumerate(adjacents):
    #     curr = cluster_means[c_id, :2].copy()
    #     curr[0], curr[1] = curr[1], curr[0]
    #     for n_id in adj:
    #         neigh = cluster_means[int(n_id), :2].copy()
    #         neigh[0], neigh[1] = neigh[1], neigh[0]
    #         lines.append([curr, neigh])
    # lines = np.array(lines)
    # lc = collections.LineCollection(lines, color="orange")
    # fig, ax = plt.subplots()
    # ax.imshow(img1, cmap="gray")
    # ax.scatter(cluster_means[:, 1], cluster_means[:, 0], s=15, c="r")
    # ax.add_collection(lc)
    # plt.show()
    # return
    
    # plt.figure(1)
    # plt.imshow(img1, cmap="gray")
    # for cluster in clusters:
    #     cluster = np.array(cluster)
    #     plt.scatter(cluster[:, 1], cluster[:, 0])
    # plt.figure(2)
    # plt.imshow(img1, cmap="gray")
    # for cluster in solid_clusters:
    #     cluster = np.array(cluster)
    #     plt.scatter(cluster[:, 1], cluster[:, 0])
    # plt.show()
    # return



    # Calculate "modified variance"
    # mvar = np.zeros(cluspts.shape[0])
    # rad = 2
    # for i, pix in enumerate(cluspts):
    #     chunk = img1[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
    #     seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(segpix1[i], 0) - rad
    #     data = img_3D[seg[:, 0], seg[:, 1]]
    #     # get rid of inf points
    #     data = np.delete(data, np.argwhere(np.isinf(data[:, 2])), axis=0)
    #     center = np.mean(data, axis=0)
    #     data_cen = data - center
    #     _, _, v = np.linalg.svd(data_cen)
        
    #     lsrl = v[0]
    #     x1 = np.expand_dims(center, 0)
    #     x2 = x1 + np.expand_dims(lsrl, 0)
    #     x1x0 = data - x1
    #     x2x0 = data - x2
    #     x2x1 = np.linalg.norm(lsrl)
    #     dists = np.linalg.norm(
    #         np.cross(x1x0, x2x0), axis=1
    #     ) / x2x1
    #     mvar[i] = np.median(dists)
    
    # ax = plt.axes(projection='3d')
    # ax.view_init(0, 0)
    # ax.set_zlim(0, 1000)
    # ax.scatter(
    #     cluspts[:, 0],
    #     cluspts[:, 1],
    #     img_3D[cluspts[:, 0], cluspts[:, 1], 2],
    #     c=mvar)
    # plt.show()



if __name__ == "__main__":
    fileb = "../Blender_imgs/blend_thread_1.jpg"
    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend1_calibration.yaml"
    imgb = cv2.imread(fileb)
    imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    img1 = imgb[:, :640]
    img2 = imgb[:, 640:]
    img1 = np.where(img1>=200, 255, img1)
    img2 = np.where(img2>=200, 255, img2)
    
    # file1 = "../Sarah_imgs/thread_1_left_final.jpg"#sys.argv[1]
    # file2 = "../Sarah_imgs/thread_1_right_final.jpg"#sys.argv[2]
    # calib = "/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml"
    # img1 = cv2.imread(file1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread(file2)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    keypt_selection(img1, img2, calib)