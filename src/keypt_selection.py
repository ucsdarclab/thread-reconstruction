import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections
from matplotlib import colors as mcolors
import numpy as np
import cv2
import copy


def keypt_selection(img1, img2, Q):
    # Color segment images
    pix_thresh = 250
    segpix1 = np.argwhere(img1<=pix_thresh)
    segpix2 = np.argwhere(img2<=pix_thresh)
    img_3D = np.zeros((img1.shape[0], img1.shape[1], 3))
    
    reliab = np.zeros(segpix1.shape[0])
    max_disp = 80
    rad = 2
    c_data = 5
    c_slope = 8
    c_shift = 0.8
    ignore_rad = 2
    disp_thresh = segpix1.shape[0]//400
    depth_calc = np.ones((4, segpix1.shape[0]))
    # Stereo match and get reliabilites
    for i in range(segpix1.shape[0]):
        pix = segpix1[i]
        curr_max_disp = min(pix[1], max_disp)
        chunk = img1[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
        seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(segpix1[i], 0) - rad

        energy = np.zeros(curr_max_disp)
        for off in range(curr_max_disp):
            g_l = img1[seg[:,0], seg[:,1]]
            g_r = img2[seg[:,0], seg[:,1] - off]
            # Compare blocks, heavily penalizing fully unsegmented right-image blocks
            if (np.abs(g_r - 255) < 1).all():
                energy[off] = 255**2 * g_r.shape[0]
            else:
                energy[off] = np.sum((g_l - g_r)**2)
        
        best = np.min(energy)
        disp = np.argmin(energy)
        energy2 = np.delete(energy, slice(disp-ignore_rad, disp+ignore_rad+1))
        next_best = np.min(energy2)
        # disp2 = np.argmin(energy2)
        
        x = (next_best - best)/((best + 1e-7)*c_data)
        reliab[i] = 1/(1+np.exp(np.clip(-1*c_slope*(x-c_shift), -87, None)))

        depth_calc[:, i] = np.array([pix[0], pix[1], disp, 1])
    # Reproject to 3D
    depth_calc = np.matmul(Q, depth_calc.copy())
    depth_calc /= depth_calc[3].copy() + 1e-7
    img_3D[segpix1[:, 0], segpix1[:, 1], 2] = depth_calc[2]
    
    # prune unreliable points
    reliab_thresh = 0.9
    relidx = np.argwhere(reliab>reliab_thresh)
    relpts = segpix1[relidx[:, 0]]

    #Find Clusters
    clusters = []
    vlist = [pt for pt in relpts.copy()]
    vmap = np.ones_like(img1)
    vmap[relpts[:, 0], relpts[:, 1]] = 0
    escape = False
    # Neighbors are all pixels of manhattan distance at most 2
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

    # Extract keypoints, create cluster map
    cluster_means = np.zeros((len(clusters), 3))
    cluster_map = np.zeros_like(img1)
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        cluster_means[i, :2] = np.mean(cluster, axis=0)
        cluster_means[i, 2] = np.mean(img_3D[cluster[:, 0], cluster[:, 1], 2])
        cluster_map[cluster[:, 0], cluster[:, 1]] = i+1

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
    
    # Get adjacency information
    adjacents = [set() for _ in solid_clusters]
    grow_paths = [set() for _ in solid_clusters]
    for c_id, cluster in enumerate(solid_clusters):
        # perform DFS to find adjacent clusters
        frontier = copy.deepcopy(cluster)
        while len(frontier) > 0:
            curr = frontier.pop()
            for d in DIRECTIONS[:8]:
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