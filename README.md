# PythonDataAlignment
Aligning devices using 2D and 3D reference data

## Installation

```console
python -m pip install git+https://github.com/JelleKUL/PythonDataAlignment.git
```

## Main Use
The main goal of this package is to align the XR device and it's recorded to reference data like point clouds or BIM models.

## Inputs

### Input data

This package can estimate the device's global position using XR input data:
- 2D images with relative transform data
- 3D meshes
- Estimated GPS Location + accuraccy

### Reference Data

The input data will be referenced against existing reference data:
- 2D Geo-referenced images
- 2D Geo-referenced hdri's
- 3D Geo-referenced meshes
- 3D Geo-referenced point clouds

The data is stored using standardized [RDF](https://www.w3.org/RDF/) data to make sure the data can be read from different datasets

## Sub selection of reference data

Comparing the input data against all the existing reference data on a server, would not only take a very long time, but is also not nescessary. We can make a sub selection of the refernce data using the GPS and its accuracy to check which reference data sets have data that is close enough.

This is done using:
```py
def FindBestSessions(path: str, coordinates: np.array, maxDistance: float):
```
which returns an array of json reference file paths

## 2D Check

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

all the imaegs are compared against eachother using Image features (currently ORB).
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

```ipynb
$$
\left(\begin{array}{cc} 
f_x & s & c_x\\
0  & f_y & c_y \\
0 & 0 & 1
\end{array}\right)
$$ 
where:
- $f_x, f_y$ are the horizontal and vertical focal length in pixels.
- $s$ the skewness of the camera
- $c_x,c_y$ the camera center in pixels.
```
#### Fundamental matrix

the fundamental matrix is calculated using the matched feature points.

## 3D Check

>todo: add 3D bim & mesh checking

## Licensing

The code in this project is licensed under MIT license.
