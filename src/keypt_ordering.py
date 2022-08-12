import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections
from matplotlib import colors as mcolors
import numpy as np
import cv2
from prob_cloud import prob_cloud
import sys
import copy
import time

def keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents):
    # Partition growpaths into individual disjoint parts
    # grow_lists = [np.array(list(path)) for path in grow_paths]
    # visiteds = [[False for i in len(path)] for path in grow_paths]
    grow_part = [[] for c_id in range(len(grow_paths))]
    part_adjs = [[] for c_id in range(len(grow_paths))]
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    for c_id, path in enumerate(grow_paths):
        path_list = list(path)
        visited = [False for i in len(path)]
        pix2idx = {pix:i for i, pix in enumerate(path)}
        # visiteds[i][0] = True
        num_visited = 0
        source = 0
        # frontier = [path_list[0]]
        while num_visited < len(visited):
            while visited[source]:
                source += 1
            frontier = [path_list[source]]
            visited[source] = True
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



if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_3_left_final.jpg"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_3_right_final.jpg"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    calib = "/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml"
    # img_3D, keypoints, grow_paths, order = prob_cloud(img1, img2)
    # fileb = "../Blender_imgs/blend_thread_1.jpg"
    # calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend1_calibration.yaml"
    # imgb = cv2.imread(fileb)
    # imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    # img1 = imgb[:, :640]
    # img2 = imgb[:, 640:]
    # img1 = np.where(img1>=200, 255, img1)
    # img2 = np.where(img2>=200, 255, img2)
    # plt.figure(1)
    # plt.imshow(img1, cmap="gray")
    # plt.figure(2)
    # plt.imshow(img2, cmap="gray")
    # plt.show()
    # assert False
    # test()
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = \
        prob_cloud(img1, img2, calib)

    keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)