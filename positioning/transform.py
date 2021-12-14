"""Classes and Functions to manage image transforms and general transformations"""

import numpy as np
import quaternion
import os
from scipy import optimize
import cv2

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

    def get_cv2_image(this):
        this.image = cv2.imread(this.path,cv2.IMREAD_COLOR)
        return this.image
    


def dict_to_quaternion(dict):
    return np.quaternion(dict["w"],dict["x"],dict["y"],dict["z"])

def dict_to_np_vector3(dict):
    return np.array([dict["x"],dict["y"],dict["z"]])

def get_global_position_offset(testImageTransform: ImageTransform, refImageTransform: ImageTransform, refGlobalTransform: np.array, transformationMatrix: np.array):
    testGlobalTransform = 0

    # Put the refImage in global coordinate system using the global transform
    newPos = refImageTransform.pos + refGlobalTransform.pos
    newRot = refImageTransform.rot * refGlobalTransform.rot
    #globalRefImageTransform = Transform(refImageTransform.id,newPos ,newRot,1)

    #transform the new globalrefImage to the testImage
    #globalTestImageTransform = Transform(testImageTransform.id, 0,0,1)

    #print("array:" + str(np.array(transformationMatrix)) + ", position:" + str(np.transpose(globalRefImageTransform.pos)))
    #globalTestImageTransform.pos =  np.matmul(np.array(transformationMatrix), np.transpose(globalRefImageTransform.pos))


    #testGlobalTransform = globalTestImageTransform.pos - dict_to_np_vector3(testImageTransform.pos)

    print("The reference Image Global position: " + str(newPos))
    print("The reference Image Global rotation: " + str(newRot))
    print("The transformationMatrix: \n" + str(transformationMatrix))
    print("The test Image local position: " + str(dict_to_np_vector3(testImageTransform.pos)))
    #print("The test Image Global position: " + str(globalTestImageTransform.pos))
    print("The Calculated test Global offset:" + str(testGlobalTransform))


    return testGlobalTransform

def get_session_scale(image1: ImageTransform, image2: ImageTransform, transMatrix):
    """Calculates the pixel scale of a transformation matrix"""

    translation, rot = get_translation(transMatrix)
    if (np.linalg.norm(translation) == 0): return 0
    scale = np.linalg.norm(image1.pos - image2.pos) / np.linalg.norm(translation)
    return scale

def triangulate_session(image1: ImageTransform, image2: ImageTransform, transMatrix1, transMatrix2):
    """Calculates a new transform based on 2 Essential transformations"""
    
    scale = 1
    translation1, rot1 = get_translation(transMatrix1)
    translation2, rot2 = get_translation(transMatrix2)
    
    def get_distance(x): 
        pos1 = get_position(x, image1, translation1)
        pos2 = get_position(x, image2, translation2)
        return np.linalg.norm(pos2-pos1)
    def get_distance_inverse(x): 
        pos1 = get_position(x, image1, translation1)
        pos2 = get_position(x, image2, -translation2)
        return np.linalg.norm(pos2-pos1)
    def get_distance_array(x):
        pos1 = get_position(x[0], image1, translation1)
        pos2 = get_position(x[1], image2, translation2)
        return np.linalg.norm(pos2-pos1)

    minimum1 = optimize.fmin(get_distance, 1)
    minimum2 = optimize.fmin(get_distance_array, [1,1])
    minimum3 = optimize.fmin(get_distance_inverse, 1)

    if(get_distance_inverse(minimum3) < get_distance(minimum1)):
        scale = minimum3[0]
        pos1 = get_position(scale, image1, translation1)
        pos2 = get_position(scale, image2, -translation2)
        newPos =(pos1 + pos2)/2
        return newPos, rot1, pos1,pos2, scale
    if(get_distance(minimum1) < get_distance_array(minimum2)):
        scale = minimum1[0]
        pos1 = get_position(scale, image1, translation1)
        pos2 = get_position(scale, image2, translation2)
        newPos =(pos1 + pos2)/2
        return newPos, rot1, pos1,pos2, scale
    else:
        pos1 = get_position(minimum2[0], image1, translation1)
        pos2 = get_position(minimum2[1], image2, translation2)
        newPos =(pos1 + pos2)/2
        return newPos, rot1, pos1,pos2, minimum2
    

def get_position(scaleFactor, imageTransform: ImageTransform, translation : np.array):
    """Returns the translation in function of a scale factor"""
    newPosition = imageTransform.pos + scaleFactor * (quaternion.as_rotation_matrix(imageTransform.rot) @ translation.T).T
    return newPosition

# helper functions (source https://github.com/harish-vnkt/structure-from-motion)
def check_pose(E):
    """Retrieves the rotation and translation components from the essential matrix by decomposing it and verifying the validity of the 4 possible solutions"""

    R1, R2, t1, t2 = get_camera_from_E(E)  # decompose E
    if not check_determinant(R1):
        R1, R2, t1, t2 = get_camera_from_E(-E)  # change sign of E if R1 fails the determinant test

    return R1, R2, t1, t2

def get_translation(E):
    R1, R2, t1, t2 = get_camera_from_E(E)

    return t1, R1

def get_camera_from_E(E):
    """Calculates rotation and translation component from essential matrix"""

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)

    R1 = u @ W @ vt
    R2 = u @ W_t @ vt
    t1 = u[:, -1].reshape((1, 3))
    t2 = - t1
    return R1, R2, t1, t2

def check_determinant(R):
    """Validates using the determinant of the rotation matrix"""

    if np.linalg.det(R) + 1.0 < 1e-9:
        return False
    else:
        return True
