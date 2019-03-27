import warnings
warnings.simplefilter("ignore")

import transforms3d
import numpy as np
import utils
import time
import open3d
import functools
import colorsys
import os
import packing_utils
import transform_utils
from math import inf


#todo:
#for downsampled pcds use only the points not the pcd

# params
session_name = 'v1test1'
snapshot_number = 0
num_sensors = 2
depth_images_folder = '../data/depth_scans'
calibration_matrices_folder = '../data/calibration'
min_packing_iterations = 1000 # even if a new snapshot is available, first perform this many packing iterations on the current one
initial_offset = 2

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
old_snapshot_downsampled = None
new_snapshot_downsampled = None

packing_iterations_performed = 0
packing_cost = inf

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

    global all_snapshots, packing_cost, packing_iterations_performed, min_packing_iterations, old_snapshot_downsampled, new_snapshot_downsampled

    snapshot_index = check_for_new_images()
    if snapshot_index == -1 or (len(all_snapshots) > 1 and packing_iterations_performed < min_packing_iterations):
        # no new snapshots to combine, perform packing:
        if len(all_snapshots) > 1:
            transformation, packing_cost, new_snapshot_downsampled = packing_utils.iteration(old_snapshot_downsampled, new_snapshot_downsampled, packing_cost)
            packing_iterations_performed += 1
            
            if transformation is not None:
                all_snapshots[-1].transform(transformation)
            else:
                print('.', end='')
                return False
        else:
            print('Not enough snapshots for packing')
            return False
    else:
        # process new snapshot
        snapshot_pointcloud = process_new_snapshot(snapshot_index)

        if snapshot_pointcloud is None:
            print('Got empty snapshot.')
            return False
        
        print('\n\n New snapshot: {} \n'.format(len(all_snapshots) + 1))

        packing_iterations_performed = 0
        packing_cost = inf
        
        # apply initial random translation
        random_translation = transform_utils.get_random_translation(initial_offset)
        snapshot_pointcloud.transform(random_translation)

        all_snapshots.append(snapshot_pointcloud)
        
        # downsampled pointclouds for packing behind the screen:
        # merge previous new with the rest
        if new_snapshot_downsampled is not None:
            if old_snapshot_downsampled is not None:
                old_snapshot_downsampled = np.vstack((old_snapshot_downsampled, new_snapshot_downsampled))
            else:
                old_snapshot_downsampled = new_snapshot_downsampled
        # get new downsampled snapshot
        new_snapshot_downsampled = np.asarray(open3d.voxel_down_sample(snapshot_pointcloud, voxel_size = 0.05).points)

        if vis is not None:
            vis.add_geometry(snapshot_pointcloud)
    if vis is not None:
        vis.update_geometry()
        vis.update_renderer()

    print(':', end='')
    return False


# Main script:
all_snapshots = []
update()
open3d.draw_geometries_with_animation_callback(all_snapshots, update)