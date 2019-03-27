# Generates matrices for transforming polar to cartesian coordinates.
# Only need to run this is if your paramteres differ from the ones below.

import math
import numpy as np

# parameters (For Kinect Xbox 360)
param_file = 'utils_data/pol2car.npz'

# field of view
fov_x = math.radians(57)
fov_y = math.radians(43)

# raw depth to distance in m.
r_scale = 1. / 65.
r_offset = 3. / 65.
depth_after_scale = 1

# angle bounds (= depth image resolution)
raw_x_max = 640
raw_y_max = 480

# script
params = {}

# computed parameters
raw_x_center = raw_x_max / 2
raw_y_center = raw_y_max / 2
mat_shape = (raw_x_max, raw_y_max)

# values for transforming from raw_z to r (radial distance in meters from the sensor)
params['raw_z_to_r'] = np.array([r_scale, r_offset])
params['depth_after_scale'] = np.array([depth_after_scale])

# raw x and y coordinates
raw_x, raw_y = np.meshgrid(
    np.arange(raw_x_max),
    np.arange(raw_y_max),
    indexing='ij')

# phi (azimuthal angle)
phi = (raw_x - raw_x_center) / raw_x_max * fov_x
# theta (polar angle)
theta = (raw_y - raw_y_center) / raw_y_max * fov_y

# distance to real coordinates multiplication matrices (element wise)
params['r_to_x'] = np.cos(theta) * np.sin(phi)
params['r_to_y'] = np.cos(theta) * np.cos(phi)
params['r_to_z'] = np.sin(theta)

# matrices for not transforming
#params['r_to_x'] = phi
params['r_to_y'] = 1
#params['r_to_z'] = theta


# save all parameters and variables
np.savez(param_file, **params)