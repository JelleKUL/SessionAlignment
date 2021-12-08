# PythonDataAlignment
Aligning devices using 2D and 3D reference data

## Installation

```console
python -m pip install git+https://github.com/JelleKUL/PythonDataAlignment.git
```

## Main Use
The main goal of this package is to align the XR device and it's recorded to reference data like point clouds or BIM models.

## workflow
1. Gather all the data (read the files and convert them to session objects)
1. Make a sub selection to get only the relevant reference data
1. Compare images to find matches and calculate the transformation
1. Compare Meshes/ point clouds and calculate the transformation
1. Pick the final transformation

## Inputs

### Input data

This package can estimate the device's global position using XR input data:
- 2D images with relative transform data
- 3D meshes
- Estimated GPS Location + accuracy

### Reference Data

The input data will be referenced against existing reference data:
- 2D Geo-referenced images
- 2D Geo-referenced hdri's
- 3D Geo-referenced meshes
- 3D Geo-referenced point clouds

The data is stored using standardized [RDF](https://www.w3.org/RDF/) data to make sure the data can be read from different datasets

## Sub selection of reference data

Comparing the input data against all the existing reference data on a server, would not only take a very long time, but is also not necessary. We can make a sub selection of the reference data using the GPS and its accuracy to check which reference data sets have data that is close enough.

This is done using:
```py
def FindBestSessions(path: str, coordinates: np.array, maxDistance: float):
```
which returns an array of json reference file paths

## 2D Check

The images and all their information are stored in a `ImageTransform`:
```py
class ImageTransform:
    id = ""
    pos = (0,0,0)
    rot = (0,0,0,0)
    fov = 0
    path = ""
    sessionDataPath = ""
    cameraMatrix = [[0,0,0],[0,0,0],[0,0,0]]
```
This is the main datatype to pass around in the different functions

You can compare the input data and the reference data using:
```py
def CompareImageSession(testSessionDataPath, refSessionDataPath):
```
Which returns a ```BestResult```
```py
class BestResult:
        def __init__(self,testImage, refImage, transImage, transMatrix, matchAmount):
            self.testImage = testImage
            self.refImage = refImage
            self.transImage = transImage
            self.transMatrix = transMatrix
            self.matchAmount = matchAmount
```
> todo optimise for searching feature matches, then transformation matrix

### Feature check

all the images are compared against each other using Image features (currently ORB).
Images that have enough matches will used for the next step

### Creating the transformation matrix

The transformation matrix (essential matrix) is calculated using the fundamental matrix and the camera calibration matrix

- https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html#epipolar-geometry
- https://harish-vnkt.github.io/blog/sfm/

#### Calibration matrix

The camera calibration matrix can be calculated using the vertical field of view of the camera using:
```py
def GetCameraMatrix(fov, referenceImagePath):
```
and returns a 3x3 Matrix like this:

```py
cameraMatrix = 
    [[ f_x ,  s  , c_x ],
     [  0  , f_y , c_y ],
     [  0  ,  0  ,  1  ]]
```
where:
- _f<sub>x</sub>_, _f<sub>y</sub>_ are the horizontal and vertical focal length in pixels.
- _s_ the skewness of the camera
- _c<sub>x</sub>_, _c<sub>y</sub>_ the camera center in pixels.

#### Fundamental matrix

The fundamental matrix is calculated using the matched feature points.
OpenCV has existing functions to calculate the fundamental matrix

#### Essential matrix

To calculate the final transformation matrix, the Essential matrix, we need to calibrate the Fundamental matrix using the Camera matrices. 

```py
E = K.T @ F @ K
```

If the 2 images are from different camera's you need to use both Calibration matrices to get the final transformation

## 3D Check

>todo: add 3D bim & mesh checking

## Licensing

The code in this project is licensed under MIT license.
