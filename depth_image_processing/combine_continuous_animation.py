import transforms3d
import numpy as np
import utils
import time
import open3d
import functools
import colorsys
import os


# params
session_name = 'v1test2'
snapshot_number = 0
num_sensors = 2
depth_images_folder = '../data/depth_scans'
calibration_matrices_folder = '../data/calibration'

filter = [
    [-0.4, 0.4],
    [-0.65, 2],
    [-0.7, 1.5]
]
# filter=[None, None, None]

# computed params and global variables 
calibration_matrices_filename = '{}/{}.npz'.format(calibration_matrices_folder, session_name)
transformations = np.load(calibration_matrices_filename)

snapshots_processed = set([])
voxels_combined = None

all_snapshots = []

# funcs
def check_for_new_images():
    global snapshots_processed, depth_images_folder, session_name, num_sensors

    available_run_images = os.listdir('{}/{}'.format(depth_images_folder, session_name))
    snapshots = {}
    for image in available_run_images:
        snapshot_index = int(image[:4])
        if snapshot_index not in snapshots:
            snapshots[snapshot_index] = 1
        else:
            snapshots[snapshot_index] += 1
        if snapshots[snapshot_index] == num_sensors:
            # snapshot is ready to process
            if snapshot_index not in snapshots_processed:
                # snapshot is not yet processed
                return snapshot_index

    return -1

def process_new_snapshot(snapshot_index):
    global num_sensors, depth_images_folder, session_name, snapshots_processed
    
    snapshots_processed.add(snapshot_index)
    # load depth scans and combine to snapshot model:
    snapshot_pointcloud = None
    for sensor_index in range(num_sensors):
        depth_image_filename = '{}/{}/{:04d}_{:02d}.npy'.format(depth_images_folder, session_name, snapshot_index, sensor_index)
        print('Opening image {}...'.format(depth_image_filename))
        depth_image = np.load(depth_image_filename)
        pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter)
        if pointcloud is None:
            return None
        pointcloud.paint_uniform_color(utils.item_in_range_color(snapshot_index, 10, True))
        transformation = transformations[transformations.files[sensor_index]]
        pointcloud.transform(transformation)
        snapshot_pointcloud = utils.merge_pointclouds(snapshot_pointcloud, pointcloud)

    return snapshot_pointcloud


def update(vis=None):
    global all_snapshots
    snapshot_index = check_for_new_images()
    if snapshot_index == -1:
        return False
    snapshot_pointcloud = process_new_snapshot(snapshot_index)
    if snapshot_pointcloud is None:
        return False
    all_snapshots.append(snapshot_pointcloud)
    if vis is not None:
        vis.add_geometry(snapshot_pointcloud)
        vis.update_geometry()
        vis.update_renderer()
    return False


# Main script:
all_snapshots = []
update()
open3d.draw_geometries_with_animation_callback(all_snapshots, update)