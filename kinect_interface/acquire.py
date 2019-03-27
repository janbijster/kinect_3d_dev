# saves a depth image from the kinect sensor.

# pykinect.nui source at:
# https://github.com/Microsoft/PTVS/blob/master/Python/Product/PyKinect/PyKinect/pykinect/nui/structs.py

from __future__ import print_function
from pykinect import nui
import time
import pygame
import numpy as np
import functools
import sys
import os


# params
DEPTH_WINSIZE = 640, 480
depth_images_folder = '../data/depth_scans'

# vars
saved_images_counter = 0
tmp_s = pygame.Surface(DEPTH_WINSIZE, 0, 16)
kinect_callbacks = []
saved_images_counter = 0

def depth_frame_ready(frame, sensor_index):
    global depth_images_folder, session_name, saved_images_counter, current_count_saved_images

    if sensor_index not in current_count_saved_images:
        frame.image.copy_bits(tmp_s._pixels_address)
        img = (pygame.surfarray.pixels2d(tmp_s) >> 7) & 255
        filename = '{}/{}/{:04d}_{:02d}.npy'.format(depth_images_folder, session_name, saved_images_counter, sensor_index)
        np.save(filename, img)

        print('\nNew depth image from sensor {}, saved as {}'.format(sensor_index, filename))
        current_count_saved_images.append(sensor_index)

def main():
    global session_name, num_kinect_sensors, saved_images_counter, current_count_saved_images
    
    session_name = raw_input('Session name: ')
    dirname = '{}/{}'.format(depth_images_folder, session_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    num_kinect_sensors = int(raw_input('Number of sensors: '))
    current_count_saved_images = range(num_kinect_sensors)

    print('initializing kinects...')
    kinects = []
    for i in range(num_kinect_sensors):
        try:
            kinects.append(nui.Runtime(index=i))
        except WindowsError:
            sys.exit('Couldn\'t connect all sensors, check if all the sensors are connected and try again.')

        print('Initilaized Kinect sensor {0} ({1}), with depth stream {2}'.format(
            i,
            kinects[i].instance_index,
            kinects[i].depth_stream
        ))

    for sensor_index, kinect in enumerate(kinects):
        kinect_callback = functools.partial(depth_frame_ready, sensor_index=sensor_index)
        kinect.depth_frame_ready += kinect_callback
        kinect_callbacks.append(kinect_callback)
        kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.resolution_640x480, nui.ImageType.Depth)

    # Main game loop    
    while True:
        cmd = raw_input('Press enter for new depth images, q to quit.')
        if cmd == 'q':
            sys.exit()
        
        saved_images_counter += 1
        current_count_saved_images = []


if __name__ == '__main__':
    main()