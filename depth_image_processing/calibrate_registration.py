import numpy as np
import utils
import fast_global_registration as fgr
import icp_registration as icp
import time
import open3d

# params
session_name = 'cali'
snapshot_number = 0
num_sensors = 2
depth_images_folder = '../data/depth_scans'

filter = [
    [0, 4],
    [-1, 1],
    [-1, 1.5]
]

# script

pointclouds = []
for sensor_index in range(num_sensors):
    depth_image_filename = '{}/{}/{:04d}_{:02d}.npy'.format(depth_images_folder, session_name, snapshot_number, sensor_index)
    
    depth_image = np.load(depth_image_filename)
    pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter)
    if pointcloud is None:
        exit('No points in pointcloud left after filtering, try increasing the crop size.')
    pointclouds.append(pointcloud)


voxel_size = 0.01
source, target, source_down, target_down, source_fpfh, target_fpfh = fgr.prepare_dataset(
    voxel_size,
    pointclouds[0],
    pointclouds[1]
)

start = time.time()
result_fast = fgr.execute_fast_global_registration(source_down, target_down,
        source_fpfh, target_fpfh, voxel_size)
print("Fast global registration took %.3f sec.\n" % (time.time() - start))
fgr.draw_registration_result(source_down, target_down,
        result_fast.transformation)

# start accurate registration
threshold = 0.5
trans_init = result_fast.transformation
print("Initial alignment")
evaluation = open3d.evaluate_registration(source, target,
        threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = open3d.registration_icp(source, target, threshold, trans_init,
        open3d.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
print("")
fgr.draw_registration_result(source, target, reg_p2p.transformation)

print("Apply point-to-plane ICP")
reg_p2l = open3d.registration_icp(source, target, threshold, trans_init,
        open3d.TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
print("")
fgr.draw_registration_result(source, target, reg_p2l.transformation)