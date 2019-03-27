# Kinect 3d scanning and combining

Collection of scripts to acquire depth images from one or more Kinect v1 (xbox360) sensors and generate 3d models from them.
The repository is divided in two parts, with different system and python requirements:

 1. `kinect_interface/`
 2. `depth_image_processing/`

The scripts in `kinect_interface/` are only for acquiring depth images from the kinect sensor and storing them as .npy files. These scripts use *pykinect* with the *Microsoft Kinect SDK* and these dependencies also dictate the requirements, most importantly a **Windows environment** and **Python 2.7**. For Mac and Linux computers there are other ways of getting depth images from kinect sensors like OpenNI, see for instance the guide at [https://docs.opencv.org/2.4/doc/user_guide/ug_kinect.html].

The scripts in `depth_image_processing/` constitute the main part of this repo. They are platform independent and require python 3.

## 1) `kinect_interface/`

### 1.1) Installation

(Windows only)

 1. Download & install the **Microsoft Kinect SDK v1.8** [https://www.microsoft.com/en-us/download/details.aspx?id=40278]
 2. *Optional but recommended because of the different python version:* create a virtualenv with python 2.7 interpreter:

    `virtualenv --python=<path/to/python2.7/interpreter> <path/to/new/virtualenv/>`

    Example: `virtualenv --python=D:/python27/python.exe kinect_interface/env`

    Then, activate the environment. (Run `<path/to/new/virtualenv>/scripts/activate`)

 3. Move to the `kinect_interface/` folder and install dependencies:
    
    `pip install -r requirements.txt`

### 1.2) Usage

 1. Connect Kinect scanners and position them pointing at the center of the stage/subject.
 2. Run `python acquire.py`, enter a name for the session and the number of connected sensors.


## 2) `depth_image_processing/`

### 2.1) Installation

 1. *Optional:* create a virtualenv with python 3.x interpreter:

    `virtualenv --python=<path/to/python3/interpreter> <path/to/new/virtualenv/>`

    Then, activate the environment. (Run `<path/to/new/virtualenv>/scripts/activate`)

 3. Move to the `kinect_interface/` folder and install dependencies:
    
    `pip install -r requirements.txt`



## 2) `depth_image_processing/`

### 2.1) Installation

1. *Optional but recommended because of the different python version:* create a virtualenv with python 2.7 interpreter:

    `virtualenv --python=<path/to/python3.x/interpreter> <path/to/new/virtualenv/>`

    Example: `virtualenv --python=D:/python37/python.exe depth_image_processing/env`

    Then, activate the environment. (Run `<path/to/new/virtualenv>/scripts/activate`)

 3. Move to the `depth_image_processing/` folder and install dependencies:
    
    `pip install -r requirements.txt`
   
### 2.2) Usage

 1. Run `calibrate_manually.py`:
   - in the parameters in the top of the file you can specify the session name, number of sensors and the snapshot index to use for the calibration.
   - Use the specified keys to position the views, then press u to save.

 2. Run `test_calibration.py` (with the right parameters for session name and number of sensors) to test if the calibration was saved right.

 3. Run `combine_continuous.py` (with the right parameters for session name and number of sensors)