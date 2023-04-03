import bpy
import numpy as np
from mathutils import Vector

curve = bpy.context.active_object
pts = curve.data.splines[0].points
to_world = curve.matrix_world
coords = np.array([to_world @ Vector(pt.co[:3]) for pt in list(pts)])

camera = bpy.data.objects["Camera"]
location = np.array(camera.location)
orientation = np.array(list(camera.rotation_axis_angle))
theta = orientation[0]
omega_x = np.array([
    [0, -1*orientation[3], orientation[2]],
    [orientation[3], 0, -1*orientation[1]],
    [-1*orientation[2], orientation[1], 0]
])

# rotate camera by 180 deg around x first
omega_init = np.array([
    [0, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])
R_init = np.eye(3) + 2*np.matmul(omega_init, omega_init)

# get camera transformation matrix from there
R = np.eye(3) + np.sin(theta)*omega_x + (1 - np.cos(theta))*np.matmul(omega_x, omega_x)
R = np.matmul(R, R_init)
T = np.zeros((4, 4))
T[:3, :3] = R
T[:3, 3] = location
T[3, 3] = 1
T_inv = np.linalg.inv(T)


aug_coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)
camera_coords = np.matmul(T_inv, aug_coords.T).T
np.save("/Users/neelay/ARClabXtra/Blender_imgs/blend3/blend3_4.npy", camera_coords[:, :3])