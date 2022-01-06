"""Classes and Functions to read and manage SessionData"""

import os
import json
from math import sqrt
import open3d as o3d

import numpy as np
import quaternion
import transform

JSON_ID = "SessionData.json"
IMG_EXTENSION = ".jpg"
MESH_EXTENSION = [".obj",".fbx" ]
PCD_EXTENSION = [".pcd", ".pts", ".ply"]

class Session:
    sessionId = ""
    dirPath = ""
    globalPosition = [0,0,0]
    globalRotation = [0,0,0,0]
    boundingBox = [[0,0,0],[0,0,0]] #3x2 matrix from min x to max z
    imageTransforms = []
    meshIds = []
    geometries = []
    pcds = []

    def __init__(self, id = None, path= None, position= None, rotation= None, images= None, meshes= None):
        self.sessionId = id
        self.dirPath = path
        self.globalPosition = position
        self.globalRotation = rotation
        self.imageTransforms = images
        self.meshIds = meshes
        pass

    def from_dict(self, dict, path):
        self.sessionId = dict["sessionId"]
        self.dirPath = path
        self.globalPosition = transform.dict_to_np_vector3(dict["globalPosition"])
        self.globalRotation = transform.dict_to_quaternion(dict["globalRotation"])
        self.imageTransforms = []
        for data in enumerate(dict["imageTransforms"]):
            newTransform = transform.ImageTransform().from_dict(data[1], path)
            self.imageTransforms.append(newTransform)
        self.meshIds = dict["meshIds"]
        self.geometries = self.get_geometries()
        return self

    def from_path(self, path):
        """Create a session using the directory file path"""
        sessionFile = open(os.path.join(path, JSON_ID),)
        sessionData = json.load(sessionFile)
        self.from_dict(sessionData,path)
        return self

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

    def set_global_pos_rot(self,pos, rot):
        self.globalPosition = pos
        self.globalRotation = rot

    def get_geometries(self):
        "returns a list of all the geometries that are linked to this session"
        self.geometries = []
        for file in os.listdir(self.dirPath):
            for id in self.meshIds:
                if file.find(id) != -1:
                    #print("found file with a meshID", id)
                    #a 3D format file is found, now check if it's a pcd or mesh
                    if file.endswith(tuple(MESH_EXTENSION)):
                        newMesh = o3d.io.read_triangle_mesh(os.path.join(self.dirPath, file))
                        if not newMesh.has_vertex_normals():
                            newMesh.compute_vertex_normals()
                        self.geometries.append(newMesh)
                    elif file.endswith(tuple(PCD_EXTENSION)):
                        newPcd = o3d.io.read_point_cloud(os.path.join(self.dirPath, file))
                        self.geometries.append(newPcd)
        return self.geometries


    def get_transformation_matrix(self):
        "returns the transformationmatrix of the session"
        matrix = quaternion.as_rotation_matrix(self.globalRotation)
        matrix = np.concatenate((matrix,self.globalPosition.T), axis = 1)
        matrix = np.concatenate((matrix, np.array([0,0,0,1])), axis = 0)

        return matrix

    
    def get_new_global_transformation_matrix(self, transformation):
        "returns a global transformation matrix based on the original transform and the transformation to the new position"
        return transformation @ self.get_transformation_matrix()
        

    def to_json():
        """converts this session object back to json"""
        print("converting to json is not yet implemented")

def sphere_intersection(center1, radius1, center2, radius2):
    """returns true if the 2 spheres are intersecting"""
    centerDistance = sqrt(pow(center1[0] + center2[0], 2) + pow(center1[1] + center2[1], 2) + pow(center1[2] + center2[2], 2))
    print("centerDistance = " + str(centerDistance))
    return centerDistance < (radius1 + radius2)


def find_close_sessions(path: str, coordinates: np.array, maxDistance: float):
    """Finds all the close enough session from a given center point
    returns: A list of Session objects tht are withing the range of the reference    
    """
    
    closeEnoughSessions = []

    for root, dir, files in os.walk(path, topdown=False):
        for name in files:
            if(name.endswith(JSON_ID)):
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
