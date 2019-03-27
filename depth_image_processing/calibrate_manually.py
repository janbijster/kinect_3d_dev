import transforms3d
import numpy as np
import utils
import time
import open3d
import functools
import colorsys
import transform_utils


# params
session_name = 'v1test1'
snapshot_number = 6
num_sensors = 2
depth_images_folder = '../data/depth_scans'
calibration_matrices_folder = '../data/calibration'

filter = [
    [-10, 10],
    [-10, 10],
    [-10, 10]
]
filter = [None, None, None]

# computed and global variables
current_model = 0
transformations = [np.eye(4) for i in range(num_sensors)]
calibration_matrices_filename = '{}/{}.npz'.format(calibration_matrices_folder, session_name)

# script
def model_color(i):
    global num_sensors, current_model
    return utils.item_in_range_color(i, num_sensors, i == current_model)

pointclouds = []
for sensor_index in range(num_sensors):
    depth_image_filename = '{}/{}/{:04d}_{:02d}.npy'.format(depth_images_folder, session_name, snapshot_number, sensor_index)
    
    depth_image = np.load(depth_image_filename)
    pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter)
    if pointcloud is None:
        exit('No points in pointcloud left after filtering, try increasing the crop size.')
    pointcloud.paint_uniform_color(model_color(sensor_index))
    pointclouds.append(pointcloud)


# callbacks
def change_current_model(vis, change_to):
    global current_model 
    current_model = change_to
    for i in range(num_sensors):
        pointclouds[i].paint_uniform_color(model_color(i))
    vis.update_geometry()
    vis.update_renderer()
    print('Current model is now {}.'.format(current_model))
    return False

def callback_transform_model(vis, transformation_matrix):
    global current_model
    pointclouds[current_model].transform(transformation_matrix)
    transformations[current_model] = np.matmul(transformation_matrix, transformations[current_model])
    vis.update_geometry()
    vis.update_renderer()
    return False

def callback_save(vis):
    print('saving...')
    np.savez(calibration_matrices_filename, *transformations)
    return False

key_to_callback = {
    ord('U'): callback_save
}
for i in range(num_sensors):
    key_to_callback[ord(str(i+1))] = functools.partial(change_current_model, change_to=i)
for key, value in transform_utils.get_transform_keys().items():
    key_to_callback[ord(key)] = functools.partial(callback_transform_model, transformation_matrix=value)

print('\n\nKeys:')
print('1, 2, etc: select scan')
print('u: save transformations')
for key, value in transform_utils.get_transform_names().items():
    print('{}: {}'.format(key.lower(), value[:-7].replace('_', ' ')))

open3d.draw_geometries_with_key_callbacks(pointclouds, key_to_callback)