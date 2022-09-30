import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections
from matplotlib import colors as mcolors
import numpy as np
import cv2

def keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents):
    # Partition growpaths into individual disjoint parts using BFS
    # TODO simplify? This is only useful for endpoints at the moment
    grow_parts = [[] for c_id in range(len(grow_paths))]
    part_adjs = [[] for c_id in range(len(grow_paths))]
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                            [1, 1], [-1, -1], [-1, 1], [1, -1]])
    min_size = 2
    for c_id, path in enumerate(grow_paths):
        path_list = list(path)
        visited = [False for i in range(len(path))]
        pix2idx = {pix:i for i, pix in enumerate(path)}
        num_visited = 0
        source = 0
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

    # Extract curve segments, avoiding intersections
    # TODO Only find one segment, so using segment list is unnecessary
    segments = []
    visited = [0 for c_id in range(len(keypoints))]
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
        # TODO clean up use of frontier
        frontier = [source]
        visited[source] = 1
        segment = []
        while len(frontier) > 0:
            curr = frontier.pop()
            segment.append(curr)
            min_dist = np.inf
            min_neigh = None
            for neigh in adjacents[curr]:
                neigh = int(neigh)
                dist = np.linalg.norm((keypoints[neigh] - keypoints[curr])[:2])
                if visited[neigh] != 1 and dist < min_dist:
                    min_dist = dist
                    min_neigh = neigh
            if min_neigh is not None:
                visited[min_neigh] = 1
                frontier.append(min_neigh)
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

    return img_3D, keypoints, grow_paths, segments[0]