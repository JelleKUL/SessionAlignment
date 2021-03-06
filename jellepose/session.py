"""Classes and Functions to read and manage SessionData"""

import datetime
import json
import os
from math import sqrt

import numpy as np
import quaternion

import jellepose.params as params
import jellepose.utils as utils
from jellepose.estimation import PoseEstimation
from jellepose.geometrytransform import GeometryTransform
from jellepose.imagetransform import ImageTransform


class Session():
    """This class stores a full session, including all the images and meshes"""

    sessionId = ""                      # the id/name of the session
    dirPath = ""                        # the system path of session directory
    globalPosition = np.array([0,0,0])            # the global position of the session origin
    globalRotation = quaternion.from_float_array([0,0,0,1])          # the global rotation as a quaternion
    boundingBox = [[0,0,0],[0,0,0]]     # 3x2 matrix from min x to max z of all the elements in the session
    imageTransforms = []                # a list of all the image transforms
    geometries = []                     # a list of the open3d geometries (meshes/pcd's together)
    estimations: PoseEstimation = []    # a list of the estimated guasses including their confidence
    fidelity = 1
    recordingDate = datetime.datetime.now()
    accuracy = []

    def __init__(self, id = None, path= None, position= np.array([0,0,0]), rotation= quaternion.from_float_array([0,0,0,1]), images= None, meshes= None):
        """Initialise the session"""

        self.sessionId = id
        self.dirPath = path
        self.globalPosition = position
        self.globalRotation = rotation
        self.imageTransforms = images
        self.meshIds = meshes
        pass

    def from_dict(self, dict, path):
        """Create a session directly drom a dictionary containing all the data"""

        self.sessionId = dict["sessionId"]
        self.dirPath = path
        self.globalPosition = utils.dict_to_np_vector3(dict["globalPosition"])
        self.globalRotation = utils.dict_to_quaternion(dict["globalRotation"])
        self.imageTransforms = self.get_images(dict["imageTransforms"])
        self.geometries = self.get_geometries(dict["meshIds"])
        return self

    def from_path(self, path):
        """Create a session using the directory file path"""

        sessionFile = open(os.path.join(path, params.JSON_ID),)
        sessionData = json.load(sessionFile)
        self.from_dict(sessionData,path)
        return self

    def get_images(self, imageIds):
        " returns all the imageTransforms in the session"

        self.imageTransforms = []
        for file in os.listdir(self.dirPath):
            for image in imageIds:
                if file.find(image["id"]) != -1:
                    #a 2D format file is found, now check if it's a pcd or mesh
                    if file.endswith(tuple(params.IMG_EXTENSION)):
                        newImg = ImageTransform().from_dict(image, os.path.join(self.dirPath, file))
                        self.imageTransforms.append(newImg)

        return self.imageTransforms
    
    def get_geometries(self, meshIds):
        "returns a list of all the geometries in the session"

        self.geometries = []
        for file in os.listdir(self.dirPath):
            for geometry in meshIds:
                if file.find(geometry) != -1:
                    #a 3D format file is found, now check if it's a pcd or mesh
                    if(file.endswith(tuple(params.MESH_EXTENSION)) or file.endswith(tuple(params.PCD_EXTENSION))):
                        newGeometry = GeometryTransform().from_dict(geometry, os.path.join(self.dirPath, file), "mesh")
                        self.geometries.append(newGeometry)

        return self.geometries

    def get_session_3d_objects(self):
        """Returns all the meshes and image transforms as open3d object to plot"""

        objects = []

        for image in self.imageTransforms:
            objects.append(image.get_camera_geometry(0.2))
        for geometry in self.geometries:
            objects.append(geometry.get_geometry())
        return objects

    def get_bounding_box(self):
        """returns a 2x3 numpy matrix containing the min and max values of the sessionData"""

        self.boundingBox = np.concatenate((self.imageTransforms[0].pos, self.imageTransforms[0].pos),axis=0).reshape(2,3)
        for trans in self.imageTransforms:
            print(trans.pos)
            self.boundingBox = np.concatenate((np.minimum(trans.pos, self.boundingBox[0]), np.maximum(trans.pos, self.boundingBox[1])),axis=0).reshape(2,3)
        return self.boundingBox
    
    def get_bounding_radius(self):
        """Returns a radius from the center points where all the points are in"""

        radius = 0
        for trans in self.imageTransforms:
            distance = np.linalg.norm(trans.pos)
            radius = max(radius, distance)
        return radius

    def get_transformation_matrix(self):
        "returns the transformationmatrix of the session"

        matrix = quaternion.as_rotation_matrix(np.normalized(self.globalRotation))
        matrix = np.concatenate((matrix,self.globalPosition.T), axis = 1)
        matrix = np.concatenate((matrix, np.array([0,0,0,1])), axis = 0)
        return matrix

    def get_rotation_matrix(self):
        """Returns the 3x3 rotation matrix R """

        return quaternion.as_rotation_matrix(np.normalized(self.globalRotation))
        
    def add_pose_guess(self, otherSession, R,t, matches, method = ""):
        """Add a pose guess to the session"""

        globalRot = otherSession.get_rotation_matrix() @ R
        globalPos = np.reshape(np.array(otherSession.globalPosition), (3,1))
        trans = np.reshape(otherSession.get_rotation_matrix() @ np.reshape(t, (3,1)), (3,1))
        finalPos = globalPos + trans
        estimation = PoseEstimation(finalPos, globalRot, matches, method)
        self.estimations.append(estimation)

    def get_best_pose(self):
        """Determines the best pose based on the confidence and clustering"""

        rotations = []
        positions = []
        weights = []
        for estimation in self.estimations:
            rotations.append(quaternion.from_rotation_matrix(estimation.rotation))
            positions.append(estimation.position)
            weights.append(estimation.get_confidence())
        Q = np.array(rotations)
        T = np.array(positions)
        w = np.array(weights)/sum(weights)

        averageRotation = utils.weighted_average_quaternions(Q,w)
        averagePosition = np.average(T,axis = 0,weights = w)

        return averageRotation, averagePosition

    def convert_axis(self, mirrorAxis: str = "y"):
        posM = np.array([1,1,1])
        rotM = np.array([1,1,1,1])
        if(mirrorAxis.lower() == "x"):
            posM = np.array([-1,1,1])
            rotM = np.array([-1,1,1,-1])
        if(mirrorAxis.lower() == "y"):
            posM = np.array([1,-1,1])
            rotM = np.array([-1,1,-1,1])
        if(mirrorAxis.lower() == "z"):
            posM = np.array([1,1,-1])
            rotM = np.array([1,1,-1,-1])

        for image in self.imageTransforms:
            image.pos *= posM
            image.rot = quaternion.from_float_array(quaternion.as_float_array(image.rot) * rotM)
        for geometry in self.geometries:
            R = geometry.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi)) #rotate to match the opencv axis of Y down
            geometry.geometry.rotate(R, center=(0, 0, 0))
            #TODO add translation?
        
    def set_global_pos_rot(self,pos, rot):
        """Set the glbal position and rotation of the sesison"""

        self.globalPosition = pos
        self.globalRotation = rot

    def to_json():
        """converts this session object back to json"""

        print("converting to json is not yet implemented")
        return None

    def benchMark_to_session(self, path):
        """converts a benchmark dataset to a session object"""
        # crawl the folder to look for .camera files
        self.imageTransforms = []
        self.geometries = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".camera"):
                    # a .camera file is found
                    fileid = file.replace(".camera", "")
                    #print("found camera file:",fileid)
                    with open(os.path.join(root, file)) as f:
                        lines = f.readlines()
                        arr = []
                        for line in lines:
                            line = " ".join(line.split())
                            values = line.split(' ')
                            for value in values:
                                arr.append(float(value))
                        # add the array values to the imageTransform
                        cameraMatrix = np.array([
                            [arr[0],arr[1],arr[2]],
                            [arr[3],arr[4],arr[5]],
                            [arr[6],arr[7],arr[8]]
                            ])
                        rotationMatrix = np.array([
                            [arr[12],arr[13],arr[14]],
                            [arr[15],arr[16],arr[17]],
                            [arr[18],arr[19],arr[20]]
                            ])
                        position = np.array([arr[21],arr[22],arr[23]])

                        newImg = ImageTransform(
                            id = fileid, 
                            pos = position, 
                            rot= quaternion.from_rotation_matrix(rotationMatrix), 
                            path= os.path.join(path, "images", fileid))
                        newImg.cameraMatrix = cameraMatrix
                        self.imageTransforms.append(newImg)
                elif file.endswith(".obj"):
                    print(file)
                    newGeometry = GeometryTransform().from_path(os.path.join(root, file))
                    self.geometries.append(newGeometry)

        #print(self.imageTransforms)
        return self


def sphere_intersection(center1, radius1, center2, radius2):
    """returns true if the 2 spheres are intersecting"""

    centerDistance = sqrt(pow(center1[0] + center2[0], 2) + pow(center1[1] + center2[1], 2) + pow(center1[2] + center2[2], 2))
    print("centerDistance = " + str(centerDistance))
    return centerDistance < (radius1 + radius2)


def find_close_sessions(path: str, coordinates: np.array, maxDistance: float):
    """Finds all the close enough session from a given center point
    returns: A list of Session objects tht are within the range of the reference    
    """
    
    closeEnoughSessions = []
    for root, dir, files in os.walk(path, topdown=False):
        for name in files:
            if(name.endswith(params.JSON_ID)):
                print("Found Session data:", os.path.join(root, name))
                session = Session().from_path(root)

                if(sphere_intersection(session.globalPosition, session.get_bounding_radius(),coordinates, maxDistance)):
                    #the point is close enough
                    print(session.sessionId, ": is close enough")
                    closeEnoughSessions.append(session)
                else:
                    print(session.sessionId, ": is to far away")
    
    print("These are all the close enough sessions", closeEnoughSessions)
    return closeEnoughSessions
