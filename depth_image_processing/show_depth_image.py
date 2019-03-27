import matplotlib.pyplot as plt
import numpy as np
import utils


# params
depth_image_filename = '../data/depth_scans/v1test2/0006_01.npy'

filter = [
    [-1, 1],
    [-0.65, 4],
    [-1, 2]
]

# script
depth_image = np.load(depth_image_filename)

print('raw image:')
plt.matshow(depth_image)
plt.show()

depth_image_preprocessed = utils.preprocess_image(depth_image)

print('image preprocessed:')
plt.matshow(depth_image_preprocessed)
plt.show()

pointcloud = utils.depth_image_to_pointcloud(depth_image, filter=filter)

print('image with actual depth in m:')
utils.show_pointcloud(pointcloud)

#pointcloud_mat = utils.depth_image_to_scientific_coordinates(depth_image_preprocessed, filter=filter)
#utils.show_pointcloud(pointcloud_mat)