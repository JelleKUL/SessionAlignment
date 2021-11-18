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