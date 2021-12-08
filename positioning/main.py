
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
    # Compare the test mesh against relevant point clouds
    # Compare the test mesh against BIM models
    # Perform a CCP for the final alignment
# Step 5: Choosing final Position
    # Use the different results from all the methods to come to a best position
    # Send the Position and rotation back to the device

import numpy as np
from positioning.positioning2D import get_2D_transformation
from session import find_close_sessions, Session

def get_transformation_offset(testSessionDir, refsessionsdir, testGlobalPos, testGlobalAcc):
    """calculates the exact global offset for the testSession"""

    # Step 2: check for relevant Reference data
    closeEnoughSessions = find_close_sessions(refsessionsdir, np.array(testGlobalPos), testGlobalAcc)

    # Step 3: 2D Check
    testSession = Session().from_path(testSessionDir)
    offsetPos, offsetRot = get_2D_transformation(testSession,closeEnoughSessions)

    # Step 4: 3D Check
    # TODO

    # Step 5: final position
    newGlobalPosition = [0,0,0]
    newGlobalRotation = [0,0,0,0]

    return newGlobalPosition, newGlobalRotation

# Step 1: Gather the Test Inputs
# These will be received by the server at a later stage in development
testGlobalPosition = (0,0,0) # a tuple containing the global coordinates received by the locationSender app
testGlobalAccuracy  = 20 # the global accuracy of the Coordinates sent by the locationSender app
testSessionDirLocation = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Hololens/Sessions/session-2021-10-12 14-28-27" # The location of the test session
# Reference Data
referenceSessionDirLocation = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Android"

get_transformation_offset(testSessionDirLocation,referenceSessionDirLocation,testGlobalPosition,testGlobalAccuracy )
