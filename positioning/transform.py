"""Classes and Functions to manage image transforms and general transformations"""

import numpy as np
import quaternion
import os

IMG_EXTENSION = ".jpg"

class ImageTransform:
    id = ""
    pos = (0,0,0)
    rot = (0,0,0,0)
    fov = 0
    path = "" # the path of the image
    cameraMatrix = [[0,0,0],[0,0,0],[0,0,0]]
    image = None

    def __init__(self, id = None, pos= None, rot= None, fov= None, path= None):
        """the input path is the location of the folder"""
        self.id = id
        self.pos = pos
        self.rot = rot
        self.fov = fov
        self.path = path

    def from_dict(self, dict, path):
        """the input path is the location of the folder"""
        self.id = dict["id"]
        self.pos = dict_to_np_vector3(dict["pos"])
        self.rot = dict_to_quaternion(dict["rot"])
        self.fov = dict["fov"]
        self.path = os.path.join(path, (self.id + IMG_EXTENSION))
        return self
    

class Transform:
    def __init__(self, id, pos, rot, scale):
        self.id = id
        self.pos = pos
        self.rot = rot
        self.scale = scale


def dict_to_quaternion(dict):
    return np.quaternion(dict["w"],dict["x"],dict["y"],dict["z"])

def dict_to_np_vector3(dict):
    return np.array([dict["x"],dict["y"],dict["z"]])

def get_global_position_offset(testImageTransform: ImageTransform, refImageTransform: ImageTransform, refGlobalTransform: np.array, transformationMatrix: np.array):
    testGlobalTransform = 0

    # Put the refImage in global coordinate system using the global transform
    newPos = dict_to_np_vector3(refImageTransform.pos) + dict_to_np_vector3(refGlobalTransform.pos)
    newRot = dict_to_quaternion(refImageTransform.rot) * dict_to_quaternion(refGlobalTransform.rot)
    globalRefImageTransform = Transform(refImageTransform.id,newPos ,newRot,1)

    #transform the new globalrefImage to the testImage
    globalTestImageTransform = Transform(testImageTransform.id, 0,0,1)

    #print("array:" + str(np.array(transformationMatrix)) + ", position:" + str(np.transpose(globalRefImageTransform.pos)))
    globalTestImageTransform.pos =  np.matmul(np.array(transformationMatrix), np.transpose(globalRefImageTransform.pos))


    testGlobalTransform = globalTestImageTransform.pos - dict_to_np_vector3(testImageTransform.pos)

    print("The reference Image Global position: " + str(newPos))
    print("The reference Image Global rotation: " + str(newRot))
    print("The transformationMatrix: \n" + str(transformationMatrix))
    print("The test Image local position: " + str(dict_to_np_vector3(testImageTransform.pos)))
    print("The test Image Global position: " + str(globalTestImageTransform.pos))
    print("The Calculated test Global offset:" + str(testGlobalTransform))


    return testGlobalTransform

def triangulate_session(image1: ImageTransform, image2: ImageTransform, transMatrix1, transMatrix2):
    """Calculates a new transform based on 2 Essential transformations"""

    #calculate the min distance between the 2 resulting directions with a given scale factor
    scale = 1

    pos1 = get_position(scale, image1, )
    distance = 

def get_position(scaleFactor, imageTransform: ImageTransform, translation : np.array):
    """Returns the translation in function of a scale factor"""

    newPosition = imageTransform.pos + scaleFactor * quaternion.as_rotation_matrix(imageTransform.rot) @ translation

    return newPosition

# helper functions (source https://github.com/harish-vnkt/structure-from-motion)
def check_pose(E):
    """Retrieves the rotation and translation components from the essential matrix by decomposing it and verifying the validity of the 4 possible solutions"""

    R1, R2, t1, t2 = get_camera_from_E(E)  # decompose E
    if not check_determinant(R1):
        R1, R2, t1, t2 = get_camera_from_E(-E)  # change sign of E if R1 fails the determinant test

    return R1, R2, t1, t2
        

def get_camera_from_E(E):
    """Calculates rotation and translation component from essential matrix"""

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)

    R1 = u @ W @ vt
    R2 = u @ W_t @ vt
    t1 = u[:, -1].reshape((3, 1))
    t2 = - t1
    return R1, R2, t1, t2

def check_determinant(R):
    """Validates using the determinant of the rotation matrix"""

    if np.linalg.det(R) + 1.0 < 1e-9:
        return False
    else:
        return True