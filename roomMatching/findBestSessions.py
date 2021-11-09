import numpy as np
import os
import json

# use the given global coordinates to search in all the folders and their SessionData.json

#finds the best session according to the input coordinates
def FindBestSessions(path: str, coordinates: np.array, maxDistance: float):
    
    closeEnoughSessions = []

    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            if(name.endswith('SessionData.json')):
                print("Found Session data:", os.path.join(root, name))
                sessionFile = open(os.path.join(root, name),)
                sessionData = json.load(sessionFile)

                referencePosition = np.array((sessionData["globalPosition"]["x"],sessionData["globalPosition"]["y"],sessionData["globalPosition"]["z"] ))
                print("Reference global position:" ,referencePosition)
                distance = np.linalg.norm(referencePosition - coordinates)
                print("Distance from coordinate:", distance)

                if(distance < maxDistance):
                    #the point is close enough
                    print(sessionData['sessionId'], ": is close enough")
                    closeEnoughSessions.append(os.path.join(root, name))
                else:
                    print(sessionData['sessionId'], ": is to far away")
                print(" ")
    
    print("These are all the close enough sessions", closeEnoughSessions)
    return closeEnoughSessions