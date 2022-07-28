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

def prob_cloud(img1, img2):
    pix_thresh = 250
    segpix1 = np.argwhere(img1<=pix_thresh)
    segpix2 = np.argwhere(img2<=pix_thresh)
    img_3D = stereo_match(img1, img2)
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Get reliabilities
    # TODO get max_disp from img instead of hardcoding
    reliab = np.zeros(segpix1.shape[0])
    max_disp = 40
    rad = 2
    c_data = 5
    c_slope = 8
    c_shift = 0.8
    # TODO deal with out-of-bounds conditions
    for i in range(segpix1.shape[0]):
        pix = segpix1[i]#.copy()#(segpix1[i][0], segpix1[i][1])
        chunk = img1[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
        seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(segpix1[i], 0) - rad

        energy = np.array([
            np.sum(
                (img1[seg[:,0], seg[:,1]] - img2[seg[:,0], seg[:,1] - off])**2
            ) for off in range(max_disp)
        ])
        
        best = np.min(energy)
        disp = np.argmin(energy)
        energy2 = np.delete(energy, slice(disp-1, disp+2))
        next_best = np.min(energy2)
        disp2 = np.argmin(energy2)
        
        x = (next_best - best)/((best + 1e-7)*c_data)
        reliab[i] = 1/(1+np.exp(np.clip(-1*c_slope*(x-c_shift), -87, None)))

        # pixels should also match from right to left
        # pix[1] = pix[1] - disp
        # chunk = img2[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
        # seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(pix, 0) - rad
        # energy = np.array([
        #     np.sum(
        #         (img2[seg[:,0], seg[:,1]] - img1[seg[:,0], seg[:,1] + off])**2
        #     ) for off in range(max_disp)
        # ])
        # if not (disp <= np.argmin(energy) <= disp):
        #     reliab[i] = 0

    
    # prune unreliable points
    reliab_thresh = 0.9
    relidx = np.argwhere(reliab>reliab_thresh)
    relpts = segpix1[relidx[:, 0]]

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

    # Make keypoints more sparse
    # relmap = np.zeros_like(img1)
    # relmap[relpts[:, 0], relpts[:, 1]] = 1
    # morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # clusmap = cv2.erode(relmap, morph)
    # cluspts = np.argwhere(clusmap==1)

    #Find Clusters
    clusters = []
    vlist = [pt for pt in relpts.copy()]
    vmap = np.ones_like(img1)
    vmap[relpts[:, 0], relpts[:, 1]] = 0
    escape = False
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                           [2, 0], [-2, 0], [0, 2], [0, -2],
                           [1, 1], [-1, -1], [-1, 1], [1, -1]])
    # TODO make these relative to thread size?
    max_size = 20
    min_size = 4
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
   

    # Get statistics for each cluster
    cluster_means = np.zeros((len(clusters), 3))
    cluster_vars = np.zeros((len(clusters), 3))
    cluster_map = np.zeros_like(img1)
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        cluster_means[i, :2] = np.mean(cluster, axis=0)
        cluster_means[i, 2] = np.mean(img_3D[cluster[:, 0], cluster[:, 1], 2])
        cluster_vars[i, :2] = np.var(cluster, axis=0)
        cluster_vars[i, 2] = np.var(img_3D[cluster[:, 0], cluster[:, 1], 2])
        cluster_map[cluster[:, 0], cluster[:, 1]] = i+1
    
    # ax = plt.axes(projection='3d')
    # ax.view_init(0, 0)
    # ax.set_zlim(0, 1000)
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
    # return

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
        segment = [cluster_means[source]]
        while len(frontier) > 0:
            assert len(frontier) == 1, "Curve is not linked list"
            curr = frontier.pop()
            segment.append(cluster_means[curr])
            for neigh in adjacents[curr]:
                neigh = int(neigh)
                if visited[neigh] != 1:
                    frontier.append(neigh)
                    visited[neigh] = 1
        segments.append(segment)
        
    plt.imshow(img1, cmap="gray")
    plt.scatter(cluster_means[:, 1], cluster_means[:, 0])
    for segment in segments:
        segment = np.array(segment)
        plt.scatter(segment[:, 1], segment[:, 0],\
            c=np.arange(0, segment.shape[0]), cmap="hot")
    plt.show()
        
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
    mvar = np.zeros(cluspts.shape[0])
    rad = 2
    for i, pix in enumerate(cluspts):
        chunk = img1[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
        seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(segpix1[i], 0) - rad
        data = img_3D[seg[:, 0], seg[:, 1]]
        # get rid of inf points
        data = np.delete(data, np.argwhere(np.isinf(data[:, 2])), axis=0)
        center = np.mean(data, axis=0)
        data_cen = data - center
        _, _, v = np.linalg.svd(data_cen)
        
        lsrl = v[0]
        x1 = np.expand_dims(center, 0)
        x2 = x1 + np.expand_dims(lsrl, 0)
        x1x0 = data - x1
        x2x0 = data - x2
        x2x1 = np.linalg.norm(lsrl)
        dists = np.linalg.norm(
            np.cross(x1x0, x2x0), axis=1
        ) / x2x1
        mvar[i] = np.median(dists)
    
    ax = plt.axes(projection='3d')
    ax.view_init(0, 0)
    ax.set_zlim(0, 1000)
    ax.scatter(
        cluspts[:, 0],
        cluspts[:, 1],
        img_3D[cluspts[:, 0], cluspts[:, 1], 2],
        c=mvar)
    plt.show()


    "First Draft"
    # TODO delete
    # img_3D = stereo_match(img1, img2)
    # thresh = 226
    # seg_pix = np.argwhere(img1 <= thresh)

    # rng = np.random.default_rng()
    # sample = seg_pix[rng.choice(seg_pix.shape[0], seg_pix.shape[0]//10)]
    # # test = np.ones_like(img1) * 255
    # # test[sample[:, 0], sample[:, 1]] = 0
    # # plt.imshow(test)
    # # plt.show()
    # rad = 6
    # # TODO try sampling
    # # means = np.zeros_like(img1)
    # # varians = np.zeros_like(means)
    # cloud = np.zeros((5*seg_pix.shape[0], 3))
    # varians = np.zeros_like(img1)
    # for i, pix in enumerate(seg_pix):
    #     r0 = max(0, pix[0]-rad)
    #     r1 = min(img1.shape[0]-1, pix[0]+rad+1)
    #     c0 = max(0, pix[1]-rad)
    #     c1 = min(img1.shape[1]-1, pix[1]+rad+1)
    #     roi = img1[r0:r1, c0:c1]
    #     roi_seg = np.argwhere(roi <= thresh) + np.expand_dims(pix, 0) - rad
    #     roi_depths = img_3D[roi_seg[:, 0], roi_seg[:, 1], 2]
    #     roi_depths = roi_depths[roi_depths != np.inf]
    #     roi_depths = roi_depths[roi_depths > 0]
    #     mean = np.mean(roi_depths)
    #     std = np.std(roi_depths)
    #     varians[pix[0], pix[1]] = std**2
    #     cloud[5*i] = np.array([pix[0], pix[1], mean-2*std])
    #     cloud[5*i+1] = np.array([pix[0], pix[1], mean-std])
    #     cloud[5*i+2] = np.array([pix[0], pix[1], mean])
    #     cloud[5*i+3] = np.array([pix[0], pix[1], mean+std])
    #     cloud[5*i+4] = np.array([pix[0], pix[1], mean+2*std])
    # # var_var = np.zeros((seg_pix.shape[0]*5))
    # # for i, pix in enumerate(seg_pix):
    # #     roi = img1[pix[0]-1:pix[0]+2, pix[1]-1:pix[1]+2]
    # #     roi_seg = np.argwhere(roi <= thresh) + np.expand_dims(pix, 0) - 1
    # #     roi_vars = varians[roi_seg[:, 0], roi_seg[:, 1]]
    # #     var_var[5*i:5*i+5] = 0 if varians[pix[0], pix[1]]<1e-7 else \
    # #         np.log(np.var(roi_vars) / varians[pix[0], pix[1]]+1)
    # # ax = plt.axes(projection="3d")
    # # ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2)#, c=var_var)
    # # ax.scatter(
    # #     seg_pix[:, 0],
    # #     seg_pix[:, 1],
    # #     img_3D[seg_pix[:, 0], seg_pix[:, 1], 2],
    # #     s=2, c="r")
    # # ax.set_zlim(0, 1000)
    # # plt.show()
    # return cloud


if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_1_left_final.jpg"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_final.jpg"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    prob_cloud(img1, img2)