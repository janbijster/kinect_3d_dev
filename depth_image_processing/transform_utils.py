import transforms3d
import numpy as np

rotate_interval = 0.02
translate_interval = 0.01
zoom_interval = 0.1

# transformation matrices
rotate_identity = transforms3d.euler.euler2mat(0, 0, 0)
rotate_x_min = transforms3d.euler.euler2mat(-rotate_interval, 0, 0)
rotate_y_min = transforms3d.euler.euler2mat(0, -rotate_interval, 0)
rotate_z_min = transforms3d.euler.euler2mat(0, 0, -rotate_interval)
rotate_x_plus = transforms3d.euler.euler2mat(rotate_interval, 0, 0)
rotate_y_plus = transforms3d.euler.euler2mat(0, rotate_interval, 0)
rotate_z_plus = transforms3d.euler.euler2mat(0, 0, rotate_interval)

translate_identity = [0, 0, 0]
translate_x_min = [-translate_interval, 0, 0]
translate_y_min = [0, -translate_interval, 0]
translate_z_min = [0, 0, -translate_interval]
translate_x_plus = [translate_interval, 0, 0]
translate_y_plus = [0, translate_interval, 0]
translate_z_plus = [0, 0, translate_interval]

zoom_identity = [1, 1, 1]
zoom_x_min = [1./(1+zoom_interval), 1, 1]
zoom_y_min = [1, 1./(1+zoom_interval), 1]
zoom_z_min = [1, 1, 1./(1+zoom_interval)]
zoom_x_plus = [1+zoom_interval, 1, 1]
zoom_y_plus = [1, 1+zoom_interval, 1]
zoom_z_plus = [1, 1, 1+zoom_interval]

translate_x_min_matrix = transforms3d.affines.compose(translate_x_min, rotate_identity, zoom_identity)
translate_y_min_matrix = transforms3d.affines.compose(translate_y_min, rotate_identity, zoom_identity)
translate_z_min_matrix = transforms3d.affines.compose(translate_z_min, rotate_identity, zoom_identity)

translate_x_plus_matrix = transforms3d.affines.compose(translate_x_plus, rotate_identity, zoom_identity)
translate_y_plus_matrix = transforms3d.affines.compose(translate_y_plus, rotate_identity, zoom_identity)
translate_z_plus_matrix = transforms3d.affines.compose(translate_z_plus, rotate_identity, zoom_identity)

rotate_x_min_matrix = transforms3d.affines.compose(translate_identity, rotate_x_min, zoom_identity)
rotate_y_min_matrix = transforms3d.affines.compose(translate_identity, rotate_y_min, zoom_identity)
rotate_z_min_matrix = transforms3d.affines.compose(translate_identity, rotate_z_min, zoom_identity)

rotate_x_plus_matrix = transforms3d.affines.compose(translate_identity, rotate_x_plus, zoom_identity)
rotate_y_plus_matrix = transforms3d.affines.compose(translate_identity, rotate_y_plus, zoom_identity)
rotate_z_plus_matrix = transforms3d.affines.compose(translate_identity, rotate_z_plus, zoom_identity)

zoom_x_min_matrix = transforms3d.affines.compose(translate_identity, rotate_identity, zoom_x_min)
zoom_y_min_matrix = transforms3d.affines.compose(translate_identity, rotate_identity, zoom_y_min)
zoom_z_min_matrix = transforms3d.affines.compose(translate_identity, rotate_identity, zoom_z_min)

zoom_x_plus_matrix = transforms3d.affines.compose(translate_identity, rotate_identity, zoom_x_plus)
zoom_y_plus_matrix = transforms3d.affines.compose(translate_identity, rotate_identity, zoom_y_plus)
zoom_z_plus_matrix = transforms3d.affines.compose(translate_identity, rotate_identity, zoom_z_plus)

transform_keys = {
    'Q': translate_x_min_matrix,
    'W': translate_x_plus_matrix,
    'A': translate_y_min_matrix,
    'S': translate_y_plus_matrix,
    'Z': translate_z_min_matrix,
    'X': translate_z_plus_matrix,
    'E': rotate_x_min_matrix,
    'R': rotate_x_plus_matrix,
    'D': rotate_y_min_matrix,
    'F': rotate_y_plus_matrix,
    'C': rotate_z_min_matrix,
    'V': rotate_z_plus_matrix,
    'T': zoom_x_min_matrix,
    'Y': zoom_x_plus_matrix,
    'G': zoom_y_min_matrix,
    'H': zoom_y_plus_matrix,
    'B': zoom_z_min_matrix,
    'N': zoom_z_plus_matrix
}
transform_names = {
    'Q': 'translate_x_min',
    'W': 'translate_x_plus',
    'A': 'translate_y_min',
    'S': 'translate_y_plus',
    'Z': 'translate_z_min',
    'X': 'translate_z_plus',
    'E': 'rotate_x_min',
    'R': 'rotate_x_plus',
    'D': 'rotate_y_min',
    'F': 'rotate_y_plus',
    'C': 'rotate_z_min',
    'V': 'rotate_z_plus',
    'T': 'zoom_x_min',
    'Y': 'zoom_x_plus',
    'G': 'zoom_y_min',
    'H': 'zoom_y_plus',
    'B': 'zoom_z_min',
    'N': 'zoom_z_plus'
}
transforms = [
    translate_x_min_matrix,
    translate_x_plus_matrix,
    translate_y_min_matrix,
    translate_y_plus_matrix,
    translate_z_min_matrix,
    translate_z_plus_matrix,
    rotate_x_min_matrix,
    rotate_x_plus_matrix,
    rotate_y_min_matrix,
    rotate_y_plus_matrix,
    rotate_z_min_matrix,
    rotate_z_plus_matrix,
    # zoom_x_min_matrix,
    # zoom_x_plus_matrix,
    # zoom_y_min_matrix,
    # zoom_y_plus_matrix,
    # zoom_z_min_matrix,
    # zoom_z_plus_matrix
]

def get_transform_keys():
  return transform_keys

def get_transform_names():
  return transform_names

def get_transforms():
    return transforms

def apply_transformation(points, transformation):
    points_transformed = np.matmul(
        np.hstack((points, np.ones((points.shape[0], 1)))),
        transformation.transpose()
        )[:, :3]
    return points_transformed

def get_random_translation(dist=1, no_z=True):
    translation = np.random.randn(3, 1).flatten()
    if no_z:
        translation[2] = 0
    print(translation)
    translation = (translation / np.linalg.norm(translation) * dist)
    return transforms3d.affines.compose(translation, rotate_identity, zoom_identity)