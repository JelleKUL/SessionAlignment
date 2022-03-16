"""Class for storing a localised image"""

import math
import open3d as o3d

import cv2
import numpy as np
import quaternion

import utils as utils
from rdfobject import RdfObject

class ImageTransform(RdfObject):
    """This class contains the image and its camera parameters"""

    id = ""                     # the id/name of the image without it's extension
    pos = (0,0,0)               # the position of the image in session space
    rot = (0,0,0,0)             # the rotation quaternion in sesison space
    fov = 0                     # the full vertical field of view of the camera
    path = ""                   # the full path of the image
    cameraMatrix = None         # the 3x3 Intrinsic camera matrix K
    transformationMatrix = None # the 3x4 Extrinsic pose matrix [R T]
    keypoints = None            # the cv2 generated keypoints 
    descriptors = None          # the cv2 generated descriptors
    image = None                # the cv2_image
    accuracy = 1

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
        self.pos = utils.dict_to_np_vector3(dict["pos"]) 
        self.rot = utils.dict_to_quaternion(dict["rot"])
        self.fov = dict["fov"]
        self.path = path
        return self

    def get_cv2_image(this):
        """ Returns the Image in color as a cv2/numpy array"""
        if(this.image is None):
            this.image = cv2.imread(this.path,cv2.IMREAD_COLOR)
        return this.image

    def get_cv2_features(self, max = 100):
        """Compute the image features and descriptors"""

        if(self.keypoints is None or self.descriptors is None):
            im1Gray = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(max)
            self.keypoints, self.descriptors = orb.detectAndCompute(im1Gray, None)
        return self.keypoints, self.descriptors

    def get_camera_matrix(this):
        """Calculate the Intrinsic Camera matrix with the vertical fov"""

        if(this.cameraMatrix is None):
            imageSize = [this.get_cv2_image().shape[1]/2,this.get_cv2_image().shape[0]/2] #width, height
            aspectRatio = imageSize[0] / imageSize[1] #width/height
            # determine the horizontal en vertival half fov's in radians
            a_y = math.radians(this.fov/2)
            a_x =  math.atan( math.tan(a_y) * aspectRatio)
            
            f_x = imageSize[0] / math.tan(a_x)
            f_y = imageSize[1] / math.tan(a_y)
            this.cameraMatrix = np.array([[f_x, 0, imageSize[0]], [0, f_y, imageSize[1]],[0,0,1]])
        return this.cameraMatrix

    def get_projection_matrix(this):
        """Returns the full 3x4 projection matrix m = K[R T] """
        
        return this.get_camera_matrix() @ np.hstack((quaternion.as_rotation_matrix(this.rot), np.array([this.pos]).T))

    def get_rotation_matrix(self):
        """Returns the 3x3 rotation matrix R """

        return quaternion.as_rotation_matrix(self.rot)

    def get_transformation_matrix(self):
        """Returns the 4x4 transformation matrix T """
        T = np.eye(4)
        T[:3, :3] = self.get_rotation_matrix()
        T[:3, 3] =  self.pos.T
        return T

    def get_camera_geometry(self, scale = 1):
        "Returns a open3d geometry object that represents a camera in 3D space"

        box = o3d.geometry.TriangleMesh.create_box(1.6,0.9, 0.1)
        box.translate((-0.8, -0.45, -0.05))
        box.scale(scale, center=(0, 0, 0))
        box.rotate(box.get_rotation_matrix_from_quaternion(quaternion.as_float_array(self.rot)))
        box.translate(self.pos)
        return box

    def get_features_image(self):
        """Returns the image with marked features"""
        
        return cv2.drawKeypoints(self.get_cv2_image(), self.keypoints, outImage=np.array([]), color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    def set_transformation_matrix(self, pos, rot):
        """Set the position (vector3) and rotation (quaternion) of the image"""

        transformationMatrix = np.array(rot)
        transformationMatrix = np.reshape(transformationMatrix, (3,3))
        self.transformationMatrix = np.hstack((transformationMatrix, np.array([pos]).T))
        self.pos = np.array(pos)
        self.rot = quaternion.from_rotation_matrix(np.reshape(rot, (3,3)))
        return self.transformationMatrix






