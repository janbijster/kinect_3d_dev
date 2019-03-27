# utils.py
#
# Functions for generating and showing meshes, voxel matrices or pointclouds from depth images.

import trimesh
import open3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import convolve
from skimage.measure import marching_cubes_lewiner
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import uniform_filter, gaussian_filter
from scipy.ndimage import rotate as sp_rotate
import math
import colorsys


pol2car_param_file = 'utils_data/pol2car.npz'
pol2car_params = np.load(pol2car_param_file)


# Depth image preprocessing
def raw_depth_to_radial_depth(img):
    r_scale, r_offset = pol2car_params['raw_z_to_r']
    return img * r_scale + r_offset

def depth_image_to_scientific_coordinates(img, flatten=True, filter=[None, None, None]):
    r = raw_depth_to_radial_depth(img)
    x = r * pol2car_params['r_to_x']
    y = r * pol2car_params['r_to_y']
    z = r * pol2car_params['r_to_z']
    points = np.dstack((x, y, z)).reshape(-1, 3)
    points = scientific_coordinates_to_standard(points)
    for i in range(3):
        if filter[i] is not None:
            points = points[(points[:, i] > filter[i][0]) & (points[:, i] < filter[i][1]), :]
    if flatten:
        return points
    else:
        return points[0, :], point[1, :], points[2, :]

def scientific_coordinates_to_standard(mat):
    # center y and invert:
    transformed_mat = (mat - np.array([[0, 2, 0]])) * np.array([[1, -1, 1]])
    # swap y and z:
    transformed_mat = transformed_mat[:, [0, 2, 1]]
    return transformed_mat

def preprocess_image(original_img):
    # Step 1: Put the origin on the lower left position of the image:
    img = np.flipud(np.fliplr(original_img))
    # Step 2: Filter (crop) the depth:
    img = filter_depth_image(img)
    return img

def filter_depth_image(img, min_distance=0, max_distance=255, flip_z=False):
    # kinect depth images are 0 for far away:
    img[img == 0] = max_distance
    img[img > max_distance] = max_distance
    img[img < min_distance] = min_distance
    if flip_z:
        # flip depth so that back 'wall' is at 0 and foremost objects are at (max_distance-min_distance)
        img = max_distance - img
    return img

def blur_image(img, blur_size=3):
    return gaussian_filter(img, sigma=blur_size)

def flatten_edges(img, value=0):
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img

# pointclouds

def merge_pointclouds(pcd1, pcd2, keep_colors=True, estimate_normals=True):
    if pcd1 is None and pcd2 is None:
        return None
    if pcd1 is None:
        return pcd2
    if pcd2 is None:
        return pcd1

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points))))
    if keep_colors:
        pcd.colors = open3d.Vector3dVector(np.vstack((np.asarray(pcd1.colors), np.asarray(pcd2.colors))))
    if estimate_normals:
        open3d.estimate_normals(pcd, search_param=open3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def print_pointcloud_mat_stats(mat):
    print('min x: {0:.02f}, max x: {3:.02f}, min y: {1:.02f}, max y: {4:.02f}, min z: {2:.02f}, max z: {5:.02f}'.format(*mat.min(axis=0), *mat.max(axis=0)))

def depth_image_to_pointcloud(img, filter = [None, None, None], estimate_normals=True):
    depth_image_preprocessed = preprocess_image(img)
    mat = depth_image_to_scientific_coordinates(depth_image_preprocessed, flatten=True, filter=filter)
    if mat.shape[0] == 0:
        print('No points left after filtering.')
        return None
    mat = scientific_coordinates_to_standard(mat)
    print_pointcloud_mat_stats(mat)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(mat)
    if estimate_normals:
        open3d.estimate_normals(pcd, search_param=open3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

# Voxel matrix generation

def voxels_to_mesh(voxels):
    vertices, faces, normals, values = marching_cubes_lewiner(voxels)
    mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces, normals=normals)
    mesh.invert()
    return mesh


def depth_img_to_voxels(img, filter = [None, None, None]):
    mat = depth_image_to_scientific_coordinates(img, flatten=True, filter=filter)
    mat = scientific_coordinates_to_standard(mat)

    # Returns voxel matrix based on depth image img.
    # Expects preprocessed depth image.
    width, height = img.shape
    depth = int(depth)
    voxels = np.ones((width, height, depth), dtype=bool)
    for i in range(depth):
        voxels[:, :, i] = voxels[:, :, i] & (img > i)
    return voxels


def carve_voxels(img, voxels):
    width, height, depth = voxels.shape
    if width != img.shape[0] or height != img.shape[1]:
        raise ValueError('Can\'t carve out voxels of shape {}x{}x{} with image with shape {}x{}'.format(
            width, height, depth, img.shape[0], img.shape[1]))
    
    return voxels

def voxels_union(voxels1, voxels2):
    if voxels1 is None:
        return voxels2
    else:
        return voxels1 | voxels2


def voxels_intersection(voxels1, voxels2):
    if voxels1 is None:
        return voxels2
    else:
        return voxels1 & voxels2


def rotate_voxels(voxels, rotation=(0.0, 0.0, 0.0)):
    # rotation contains three euler angles in degrees around the x, y and z axis.
    for i, axes in enumerate([(1, 2), (2, 0), (0, 1)]):
        voxels = sp_rotate(voxels, rotation[i], axes, reshape=False, order=0, mode='nearest')
    return np.round(voxels).astype(bool)

def shift_voxels(voxels, offset, rotate_cropped_part=False):
    new_voxels = np.roll(voxels, offset, (0, 1, 2))

    if not rotate_cropped_part:
        
        if offset[0] > 0:
            new_voxels[:offset[0], :, :] = False
        if offset[0] < 0:
            new_voxels[offset[0]:, :, :] = False
        
        if offset[1] > 0:
            new_voxels[:, :offset[1], :] = False
        if offset[1] < 0:
            new_voxels[:, offset[1]:, :] = False
        
        if offset[2] > 0:
            new_voxels[:, :, :offset[2]] = False
        if offset[2] < 0:
            new_voxels[:, :, offset[2]:] = False

    return new_voxels

def scale_voxels(voxels, scale):
    return np.round(zoom(voxels, scale, order=1, mode='nearest')).astype(bool)

def calibration_axes_voxels(shape, origin=(0.6, 0.75, 0.5)):
    origin_indices = (np.array(origin) * shape).astype(int)
    axes_voxels = np.zeros(shape).astype(bool)
    axes_voxels[:, origin_indices[1], origin_indices[2]] = True
    axes_voxels[origin_indices[0], :, origin_indices[2]] = True
    axes_voxels[origin_indices[0], origin_indices[1], :] = True
    return axes_voxels


# Display and saving

def save_obj(mesh, filename, vertices=None, faces=None):
    lines = []
    if mesh != None:
        vertices = mesh.vertices
        faces = mesh.faces
    
    for vertex in vertices:
        lines.append('v {0:.4f} {1:.4f} {2:.4f}'.format(vertex[0], vertex[1], vertex[2]))
    for face in faces:
        lines.append('f {0:d} {1:d} {2:d}'.format(face[0] + 1, face[1] + 1, face[2] + 1))
    with open(filename, 'w') as out_file:
        out_file.write('\n'.join(lines))

def plot_voxels(voxels, scale=(0.2, 0.2, 0.2), facecolors='#cccccc', edgecolors='#000000'):
    if scale is 1:
        small_voxels = voxels
    else:
        small_voxels = zoom(voxels, scale, mode='nearest', order=0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(small_voxels, facecolors=facecolors, edgecolors=edgecolors)
    plt.show()

def plot_multiple_voxels(voxel_list, scales=None, title_list=None):
    n = len(voxel_list)
    num_rows = math.ceil(math.sqrt(2.0*float(n)/3.0))
    num_cols = math.ceil(float(n)/num_rows)
    fig = plt.figure(figsize=plt.figaspect(2./3.))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i*num_cols + j
            voxels = voxel_list[index]

            if scales is None:
                small_voxels = zoom(voxels, (0.2, 0.2, 0.2), mode='nearest', order=0)
            else:
                small_voxels = zoom(voxels, scales[index], mode='nearest', order=0)
        
            ax = fig.add_subplot(num_rows, num_cols, index+1, projection='3d')
            ax.voxels(small_voxels, facecolors="#cccccc", edgecolors="#333333")
            
            if title_list is not None:
                ax.set_title(title_list[index])
    
    plt.show()

def plot_multiple_meshes(mesh_list, dist=200):
    scene = trimesh.scene.Scene()
    
    n = len(mesh_list)
    num_rows = math.ceil(math.sqrt(2.0*float(n)/3.0))
    num_cols = math.ceil(float(n)/num_rows)

    for yi in range(num_rows):
        for xi in range(num_cols):
            index = yi*num_cols + xi
            mesh = mesh_list[index]
            mesh.apply_translation((index * dist, 0, 0))
            scene.add_geometry(mesh)
    
    scene.show()

def show_pointcloud(pcd):
    open3d.draw_geometries([pcd])

def show_pointcloud_matrix(mat):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(mat)
    open3d.draw_geometries([pcd])

def show_pointcloud_pyplot(mat):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(mat[:,0], mat[:,1], mat[:,2], s=1, c=mat[:,1])
    plt.show()

def item_in_range_color(index, total, is_active=False):
    h = (float(index)/float(total)) % 1
    if is_active:
        l = 0.5
    else:
        l = 0.25
    return colorsys.hls_to_rgb(h, l, 1)