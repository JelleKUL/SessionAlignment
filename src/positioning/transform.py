"""Classes and Functions to manage image transforms and general transformations"""

import math
import os
import open3d as o3d

import cv2
import numpy as np
import quaternion
from scipy import optimize

IMG_EXTENSION = ".jpg"

class ImageTransform:
    id = ""
    pos = (0,0,0)
    rot = (0,0,0,0)
    transformationMatrix = []
    fov = 0
    path = "" # the path of the image
    cameraMatrix = None
    keypoints = []
    descriptors = []
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
        """ Returns the Image in color as a cv2/numpy array"""
        if(this.image is None):
            this.image = cv2.imread(this.path,cv2.IMREAD_COLOR)
        return this.image

    def get_camera_matrix(this):
        """Calculate the Camera matrix with the vertical fov"""

        if(this.cameraMatrix is None):
            imageSize = [this.get_cv2_image().shape[1]/2,this.get_cv2_image().shape[0]/2] #width, height
            aspectRatio = imageSize[0] / imageSize[1]
            a_x = this.fov * aspectRatio
            a_y = this.fov
            f_x = imageSize[0] / math.tan(math.radians(a_x) / 2 )
            f_y = imageSize[1] / math.tan(math.radians(a_y) / 2)
            this.cameraMatrix = np.array([[f_x, 0, imageSize[0]], [0, f_y, imageSize[1]],[0,0,1]])
        return this.cameraMatrix

    def get_projection_matrix(this):
        return this.get_camera_matrix() @ np.hstack((quaternion.as_rotation_matrix(this.rot), np.array([this.pos]).T))

    def get_camera_geometry(this, scale = 1):
        "Returns a geometry lineset object that represents a camera in 3D space"
        box = o3d.geometry.TriangleMesh.create_box(1.6,0.9, 0.1)
        box.translate((-0.8, -0.45, -0.05))
        box.scale(scale, center=(0, 0, 0))
        box.rotate(box.get_rotation_matrix_from_quaternion(quaternion.as_float_array(this.rot)))
        box.translate(this.pos)
        return box


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
    
    translation1, rot1 = get_translation(transMatrix1)
    translation2, rot2 = get_translation(transMatrix2)
    
    def get_distance_array(x):
        pos1 = get_position(x[0], image1, translation1)
        pos2 = get_position(x[1], image2, translation2)
        return np.linalg.norm(pos2-pos1)

    minimum = optimize.fmin(get_distance_array, [1,1])

    pos1 = get_position(minimum[0], image1, translation1)
    pos2 = get_position(minimum[1], image2, translation2)
    newPos =(pos1 + pos2)/2
    return newPos, rot1, pos1,pos2, minimum
    

def get_position(scaleFactor, imageTransform: ImageTransform, translation : np.array):
    """Returns the translation in function of a scale factor"""
    newPosition = imageTransform.pos + scaleFactor * (quaternion.as_rotation_matrix(imageTransform.rot) @ translation).T
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

    return t1.T, R1

def get_camera_from_E(E):
    """Calculates rotation and translation component from essential matrix"""

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)

    R1 = u @ W @ vt
    R2 = u @ W_t @ vt
    t1 = u[:, -1].reshape((3,1))
    t2 = - t1
    return R1, R2, t1, t2

def check_determinant(R):
    """Validates using the determinant of the rotation matrix"""

    if np.linalg.det(R) + 1.0 < 1e-9:
        return False
    else:
        return True

def check_triangulation(points, P):
    """Checks whether reconstructed points lie in front of the camera"""

    P = np.vstack((P, np.array([0, 0, 0, 1])))
    reprojected_points = cv2.perspectiveTransform(src=points[np.newaxis], m=P)
    z = reprojected_points[0, :, -1]
    if (np.sum(z > 0)/z.shape[0]) < 0.75:
        return False
    else:
        return True
