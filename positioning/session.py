"""Classes and Functions to read and manage SessionData"""

import os
import json

import numpy as np

import transform

JSON_ID = "SessionData.json"
IMG_EXTENSION = ".jpg"

class Session:
    sessionId = ""
    dirPath = ""
    globalPosition = [0,0,0]
    globalRotation = [0,0,0,0]
    boundingBox = [0,0,0]
    imageTransforms = []
    meshIds = []

    def __init__(self, id, path, position, rotation, images, meshes):
        self.sessionId = id
        self.dirPath = path
        self.globalPosition = position
        self.globalRotation = rotation
        self.imageTransforms = images
        self.meshIds = meshes
        pass

    @classmethod
    def from_dict(self, dict, path):
        self.sessionId = dict["sessionId"]
        self.dirPath = path
        self.globalPosition = transform.dict_to_np_vector3(dict["globalPosition"])
        self.globalRotation = transform.dict_to_quaternion(dict["globalRotation"])
        self.imageTransforms = []
        for data in enumerate(dict["imageTransforms"]):
            newTransform = transform.ImageTransform.from_dict(data[1], os.path.join(path, (self.sessionId + IMG_EXTENSION)))
            self.imageTransforms.append(newTransform)
        self.meshIds = dict["meshIds"]
        return self


def find_close_sessions(path: str, coordinates: np.array, maxDistance: float):
    """Finds all the close enough session from a given center point
    returns: A list of Session objects tht are withing the range of the reference    
    """
    
    closeEnoughSessions = []

    for root, dir, files in os.walk(path, topdown=False):
        for name in files:
            if(name.endswith(JSON_ID)):
                print("Found Session data:", os.path.join(root, name))
                sessionFile = open(os.path.join(root, name),)
                sessionData = json.load(sessionFile)
                session = Session.from_dict(sessionData, root)
                #print(session.__dict__)

                distance = np.linalg.norm(session.globalPosition - coordinates)
                print("Distance from coordinate:", distance)

                if(distance < maxDistance):
                    #the point is close enough
                    print(sessionData['sessionId'], ": is close enough")
                    closeEnoughSessions.append(session)
                else:
                    print(session.sessionID, ": is to far away")
    
    print("These are all the close enough sessions", closeEnoughSessions)
    return closeEnoughSessions
