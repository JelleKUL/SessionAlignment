
# Step 1: Gather the Test Inputs
    # The test session with all the pictures and coordinates
    # The global coordinates of the device
# Step 2: check for relevant Reference data
    # use the global coordinates to find all the reference data that is georeferenced close enough (GPS presision)
# Step 3: 2D Check
    # Compare all the test images against all the reference images
    # Find which session has the highest match rate
    # Find which Image has the highest match rate
    # Calculate the transformation between the two images
    # calculate the inverse transformation to give the test data a Reference global position
# Step 4: 3D Check
    # compare the test mesh against relevant point clouds
    # Compare the test mesh against BIM models
    # Perform a CCP for the final alignment
# Step 5: Choosing final Position
    # Use the different results from all the methods to come to a best position
    # Send the Position and rotation back to the device

from cv2 import transform
import numpy as np

# Step 1: Gather the Test Inputs
# These will be received by the server at a later stage in development
testGlobalPosition = (0,0,0) # a tuple containing the global coordinates received by the locationSender app
testGlobalAccuracy  = 20 # the global accuracy of the Coordinates sent by the locationSender app
testSessionDirLocation = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Hololens/Sessions/session-2021-10-12 14-28-27" # The location of the test session
# Reference Data
referenceSessionDirLocation = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Android"


# Step 2: check for relevant Reference data
from session import find_close_sessions
closeEnoughSessions = find_close_sessions(referenceSessionDirLocation, np.array(testGlobalPosition), testGlobalAccuracy)


# Step 3: 2D Check
from compareImageSession import CompareImageSession
from transform import ImageTransform
from best_result import BestResult

testSessionJsonPath = testSessionDirLocation + "/SessionData.json"
bestResult = BestResult(0,0,0,0,0)
bestRefImage = 0
bestTestImage = 0
bestSession = 0

for referenceSession in closeEnoughSessions:
    result,testImage,refImage = CompareImageSession(testSessionJsonPath, referenceSession)
    if(result.matchAmount > bestResult.matchAmount):
        bestResult = result
        bestRefImage = refImage
        bestTestImage = testImage
        bestSession = referenceSession

from transform import get_global_position_offset
from transform import Transform
import json

refSessionJson = json.load(open(bestSession,))
bestGlobalPosition = Transform(refSessionJson["sessionId"],refSessionJson["globalPosition"], refSessionJson["globalRotation"], [1,1,1])

newGlobalPosition =  get_global_position_offset(bestTestImage, bestRefImage, bestGlobalPosition, bestResult.transMatrix)

print(newGlobalPosition)
