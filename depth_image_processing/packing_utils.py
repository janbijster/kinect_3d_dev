# packing utils

# To exclude ceratin transformations, comment them out in the transforms_for_packing list in transform_utils.py

import transforms3d
import numpy as np
import transform_utils
import scipy.spatial.distance
import random
import open3d


# params
standard_collision_weight = 0.2
standard_collision_distance = 0.02
standard_distance_weight = 1.
amplification_growth = 0.05

# internal counters
consecutive_times_not_improved = 0
last_iteration_improvement = True
current_amplification = 1

def reset_internal_counters():
    global consecutive_times_not_improved, last_iteration_improvement, current_amplification
    consecutive_times_not_improved = 0
    last_iteration_improvement = True
    current_amplification = 1

def center_of_mass(points):
    return points.mean(axis=0)

def com_distance(points1, points2):
    return np.linalg.norm(center_of_mass(points1) - center_of_mass(points2))

def min_point_distance_cost(points1, points2):
    # cityblock is slightly faster than euclidean
    dists = scipy.spatial.distance.cdist(points1, points2, metric='cityblock')
    num_collisions = (dists < standard_collision_distance).sum()
    print(num_collisions, end='')
    return num_collisions
    min_dist = dists.min()
    if min_dist > standard_collision_distance:
        return 0.
    return (standard_collision_distance - min_dist) / standard_collision_distance

def cost(points1, points2):
    return standard_distance_weight * com_distance(points1, points2) + standard_collision_weight * min_point_distance_cost(points1, points2)


def iteration(points1, points2, last_cost, reset=False):
    global consecutive_times_not_improved, last_iteration_improvement, current_amplification, initial_amplification
    if reset:
        reset_internal_counters
    # Note: pcd2 is modified in place
    # 1. Pick random transformation
    transformation = random.sample(transform_utils.get_transforms(), 1)[0]

    points2_transformed = points2
    for i in range(int(current_amplification)):
        points2_transformed = transform_utils.apply_transformation(points2_transformed, transformation)

    # 2. check if transformation resulted in lower cost function
    new_cost = cost(points1, points2_transformed)
    if new_cost < last_cost:
        # 2.a Yes.
        reset_internal_counters()
        return transformation, new_cost, points2_transformed
    else:
        # 2.b No: Do nothing
        if not last_iteration_improvement:
            consecutive_times_not_improved += 1
            current_amplification += amplification_growth
            if int(current_amplification) > int(current_amplification - amplification_growth):
                print('<{}>'.format(int(current_amplification)), end='')
        last_iteration_improvement = False

        return None, last_cost, points2


