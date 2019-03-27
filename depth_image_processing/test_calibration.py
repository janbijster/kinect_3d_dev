import transforms3d
import numpy as np
import utils
import time
import open3d
import functools
import colorsys
import scipy.ndimage


# params
session_name = 'v1test1'
snapshot_number = 6
num_sensors = 2
depth_images_folder = '../data/depth_scans'
calibration_matrices_folder = '../data/calibration'

filter = [
    [-0.4, 0.4],
    [-0.65, 2],
    [-0.7, 1.5]
]
filter= [None, None, None]

# script
calibration_matrices_filename = '{}/{}.npz'.format(calibration_matrices_folder, session_name)
transformations = np.load(calibration_matrices_filename)

scans = []
model = None
for sensor_index in range(num_sensors):
    depth_image_filename = '{}/{}/{:04d}_{:02d}.npy'.format(depth_images_folder, session_name, snapshot_number, sensor_index)
    
    depth_image = np.load(depth_image_filename)
    pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter, estimate_normals=False)
    if pointcloud is None:
        exit('No points in pointcloud left after filtering, try increasing the crop size.')
    pointcloud.paint_uniform_color(utils.item_in_range_color(sensor_index, num_sensors, True))
    transformation = transformations[transformations.files[sensor_index]]

    # pointcloud.transform(transformation)
    points = np.asarray(pointcloud.points)
    points_transformed = np.matmul(
        np.hstack((points, np.ones((points.shape[0], 1)))),
        transformation.transpose()
        )[:, :3]
    pointcloud.points = open3d.Vector3dVector(points_transformed)

    scans.append(pointcloud)
    model = utils.merge_pointclouds(model, pointcloud)

#model.paint_uniform_color((0.8, 0.9, 0.7))
open3d.draw_geometries([model])