
import open3d
import numpy as np
import copy

import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    open3d.estimate_normals(pcd_down, open3d.KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.compute_fpfh_feature(pcd_down,
            open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, source, target):
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down,
        source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = open3d.registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            open3d.FastGlobalRegistrationOption(
            maximum_correspondence_distance = distance_threshold))
    return result

if __name__ == "__main__":
    exit()
#     voxel_size = 0.05 # means 5cm for the dataset
#     source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
#         voxel_size,
#         source_pointcloud,
#         target_pointcloud
#     )

#     start = time.time()
#     result_fast = execute_fast_global_registration(source_down, target_down,
#             source_fpfh, target_fpfh, voxel_size)
#     print("Fast global registration took %.3f sec.\n" % (time.time() - start))
#     draw_registration_result(source_down, target_down,
#             result_fast.transformation)