import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections
from matplotlib import colors as mcolors
import numpy as np
import cv2
from keypt_selection import keypt_selection
import sys
import copy
import time

def keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents):
    # Partition growpaths into individual disjoint parts
    # grow_lists = [np.array(list(path)) for path in grow_paths]
    # visiteds = [[False for i in len(path)] for path in grow_paths]
    grow_parts = [[] for c_id in range(len(grow_paths))]
    part_adjs = [[] for c_id in range(len(grow_paths))]
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    min_size = 2
    for c_id, path in enumerate(grow_paths):
        path_list = list(path)
        visited = [False for i in range(len(path))]
        pix2idx = {pix:i for i, pix in enumerate(path)}
        # visiteds[i][0] = True
        num_visited = 0
        source = 0
        # frontier = [path_list[0]]
        while num_visited < len(visited):
            while visited[source]:
                source += 1
            frontier = [np.array(path_list[source])]
            visited[source] = True
            num_visited += 1
            part = []
            part_adj = set()
            while len(frontier) > 0:
                curr = frontier.pop(0)
                part.append(curr)
                for d in DIRECTIONS:
                    neigh = curr + d
                    t_neigh = tuple(neigh)
                    # Don't include out-of-path neighbors in partition
                    if t_neigh not in path:
                        # Ignore OOB
                        if neigh[0] < 0 or neigh[0] >= cluster_map.shape[0] or \
                            neigh[1] < 0 or neigh[1] >= cluster_map.shape[1]:
                            continue
                        # See if partition is connected to any other clusters
                        neigh_clust = cluster_map[t_neigh]
                        if neigh_clust != 0 and neigh_clust != c_id+1:
                            part_adj.add(neigh_clust-1)
                        continue
                    n_idx = pix2idx[t_neigh]
                    if visited[n_idx]:
                        continue
                    frontier.append(neigh)
                    visited[n_idx] = True
                    num_visited += 1
            
            if len(part) > min_size:
                grow_parts[c_id].append(part)
                part_adjs[c_id].append(part_adj)
    
    # for c_id, parts in enumerate(grow_parts):
    #     plt.clf()
    #     plt.imshow(img1, cmap="gray")
    #     adj_nums = []
    #     for i, part in enumerate(parts):
    #         part = np.array(part)
    #         plt.scatter(part[:, 1], part[:, 0])
    #         adj_nums.append(len(part_adjs[c_id][i]))
    #         plt.text(part[-1, 1], part[-1, 0], str(adj_nums[-1]))
    #     plt.scatter(keypoints[c_id, 1], keypoints[c_id, 0], c="turquoise")
    #     if 0 in adj_nums:
    #         plt.show()

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
        segment = []#keypoints[source]]
        while len(frontier) > 0:
            assert len(frontier) == 1, "Curve is not linked list"
            curr = frontier.pop()
            segment.append(curr)#keypoints[curr])
            for neigh in adjacents[curr]:
                neigh = int(neigh)
                if visited[neigh] != 1:
                    frontier.append(neigh)
                    visited[neigh] = 1
        segments.append(segment)

    # Extend out endpoints
    for segment in segments:
        for side, endpt in enumerate([segment[0], segment[-1]]):
            p_adjs = part_adjs[endpt]
            for i, adjs in enumerate(p_adjs):
                # a grow part with 0 adjacents is an endpoint part
                if len(adjs) == 0:
                    # Choose pixel in part that's farthest from keypoint as endpoint
                    part = np.array(grow_parts[endpt][i])
                    dists = np.linalg.norm(keypoints[endpt:endpt+1, :2] - part, axis=1)
                    new_end = part[np.argmax(dists)]
                    # Set depth to be the same as the current pixel
                    new_end = np.array([new_end[0], new_end[1], keypoints[endpt, 2]])
                    # Add to segment and keypoint set
                    if side == 0:
                        segment.insert(0, keypoints.shape[0])
                    else:
                        segment.append(keypoints.shape[0])
                    keypoints = np.concatenate((keypoints, np.expand_dims(new_end, 0)), axis=0)
                    grow_paths.append({tuple(pix) for pix in part})

    # TODO
    "Join segments to form single ordering"
    return img_3D, keypoints, grow_paths, segments[-1]

if __name__ == "__main__":
    # file1 = "../Sarah_imgs/thread_1_left_final.jpg"#sys.argv[1]
    # file2 = "../Sarah_imgs/thread_1_right_final.jpg"#sys.argv[2]
    # img1 = cv2.imread(file1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread(file2)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # calib = "/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml"
    # img_3D, keypoints, grow_paths, order = keypt_selection(img1, img2)
    fileb = "../Blender_imgs/blend_thread_1.jpg"
    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend1_calibration.yaml"
    imgb = cv2.imread(fileb)
    imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    img1 = imgb[:, :640]
    img2 = imgb[:, 640:]
    img1 = np.where(img1>=200, 255, img1)
    img2 = np.where(img2>=200, 255, img2)
    # plt.figure(1)
    # plt.imshow(img1, cmap="gray")
    # plt.figure(2)
    # plt.imshow(img2, cmap="gray")
    # plt.show()
    # assert False
    # test()
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = \
        keypt_selection(img1, img2, calib)

    keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)