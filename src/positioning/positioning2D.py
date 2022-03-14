# The 2D position is based on OpenCV computer vision and the Essential matrix that can be calculated with matched Features
# Compare all the test images against all the reference images
    # Find which session has the highest match rate
    # Find which Image has the highest match rate
    # Calculate the transformation between the two images
    # Calculate the inverse transformation to give the test data a Reference global position
import os

import cv2
import numpy as np
import quaternion
from scipy import optimize
import math

from session import Session
from imagetransform import ImageTransform
from imagematch import ImageMatch
import positioning3d as pos3d
import utils as utils


def get_2D_transformation(testSession : Session, refSessions : "list[Session]"):
    """returns a list of possible poses along with their confidence
        methods: 
            1: Cross referencing 2 refenrence images with one test sesison image
            2: Matching 2 reference images to retrieve 3D points, then pnp estimation of test image
            3: Matching 1 reference image from a session with 3D data, with a test image, getting the global pos by raycasting"""
    
    #find the image with the best match rate
    for referenceSession in refSessions:
        compare_session(testSession, referenceSession)

    return testSession.get_best_pose()

def compare_session(testSession : Session, refSession : Session):
    """Checks a test session against a reference session, returns the 3 best matching images"""

    print("Starting Comparing:", len(testSession.imageTransforms), "Against", len(refSession.imageTransforms), "Images")

    # loop over every test image in the session, find the 2 best referenc images and keep them
    for testImage  in testSession.imageTransforms:
        guesses = []
        if(len(refSession.imageTransforms) > 1): # we need 2 ref images to match
            guesses.append(cross_reference_matching(testImage, refSession))
            guesses.append(incremental_matching(testImage, refSession))
        if(len(refSession.geometries) > 0): # we need a mesh to raycast against
            guesses.append(raycast_matching(testImage, refSession))

        # once we get the image pose in reference session space, we determine the testSession pose in reference session space
        for guess in guesses:
            R = guess[0]
            t = guess[1]
            testOriginRot = testImage.get_rotation_matrix().T @ R
            testOrgininPos = - testOriginRot @ t
            testSession.add_pose_guess(refSession, testOriginRot, testOrgininPos, guess[2])
    
    return testSession.get_best_pose()




def get_best_matches(testImage, refImages, nr = 1) -> ImageMatch:
    """Check a test image against a list of reference images. Returns a list of the "nr" best matches"""

    results = [] # a list of all the results
    bestResults = [] # a list of the best results
    nrCheck = 0
    totalCheck = len(refImages)

    for refImage in refImages:
            newMatch = ImageMatch(refImage, testImage) #create a new match between 2 images
            newMatch.find_matches() # find the best matches 
            results.append(newMatch)

            # check if the newResult is in the top of results
            bestResults.append(newMatch)
            bestResults = sorted(bestResults, key= lambda x: x.matchError) #sort them from low to High
            if(len(bestResults) > nr): #remove the worst match
                bestResults = bestResults[:nr]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

    for result in bestResults:
        result.get_essential_matrix() # determin the transformation and inliers

    if(nr == 1): return bestResults[0]
    return bestResults


# METHOD 1: Cross referencing

def cross_reference_matching(testImage, refSession):
    """Finds the estimated pose of a 'testImage' based on 2 sepreate matches in a 'refSession' """
    
    bestMatches = get_best_matches(testImage, refSession.imageTransforms, 2) #find 2 best matches
    R,t,confidence = cross_reference_pose(bestMatches[0], bestMatches[1]) # get the estimated pose

    return R,t, bestMatches # the position of the test image transform in reference session space

def cross_reference_pose(match1: ImageMatch, match2: ImageMatch):
    """determines a pose of the 3rd image based on 2 seperate reference matches"""

    def get_position(scaleFactor, match : ImageMatch):
        """Returns the translation in function of a scale factor"""
        match.set_scaling_factor(scaleFactor)
        _,t = match.get_image2_pos()
        #newPosition = imageTransform.pos + scaleFactor * (imageTransform.get_rotation_matrix() @ translation).T
        return t

    def get_distance_array(x):
        pos1 = get_position(x[0], match1)
        pos2 = get_position(x[1], match2)
        return np.linalg.norm(pos2-pos1)

    minimum = optimize.fmin(get_distance_array, [1,1])

    pos1 = get_position(minimum[0], match1)
    pos2 = get_position(minimum[1], match2)
    t =(pos1 + pos2)/2 #return the average of the 2 positions
    R,_ = match1.get_image2_pos()
    confidence = match1.fidelity + match2.fidelity
    return R, t, confidence


# METHOD 2: Incremental matching

def incremental_matching(testImage, refSession):
    """tries to determine the pose by first matching reference images to create the initial 3D points"""

    #find the 3 highest linked matches
    #find the best single match for the test image
    bestMatch = get_best_matches(testImage, refSession.imageTransforms, nr=1)
    #find the best result for the matched reference image
    bestRefMatch = get_best_session_match(bestMatch.image1, refSession)

    R,t = bestMatch.get_pnp_pose(bestRefMatch) #get the rotation and translation with the pnp point algorithm
    confidence = bestMatch.fidelity + bestRefMatch.fidelity
    return R,t, [bestMatch, bestRefMatch]

def get_best_session_match(image, session : Session):
    """Finds the best match in the same session"""

    if(image not in session.imageTransforms): 
        print("ERROR: Image not in list")
        return None
    newList = session.imageTransforms.copy()
    newList.remove(image)
    bestRefMatch = get_best_matches(image, newList)
    #Calculate the 3D points in the scene with the know real world locations of the 2 reference images
    
    bestRefMatch.get_essential_matrix() #calculate the essential matrix and inliers
    bestRefMatch.get_reference_scaling_factor() # get the scene scale by using the real world distances
    bestRefMatch.triangulate(True) #calulate the 3d points

    return bestRefMatch

# METHOD 3: RayCasting
def raycast_matching(testImage, refSession):
    """Determines the estimated pose by matching with 1 reference image and raycasting against the 3d scene"""

    #find the best single match for the test image
    match = get_best_matches(testImage, refSession.imageTransforms, nr=1)
    match.get_essential_matrix() # Calculate the essential matrix
    match.triangulate(useCameraPose = True) # determine the 3D points
    rayCastImage = match.image1

    #cast a number of rays on the determined points in the scene
    scalingFactors = []
    for point in match.points3d:
        pointVector = point - (rayCastImage.pos)
        pointDistance = np.linalg.norm(pointVector)
        direction = pointVector / pointDistance
        rayDistance = refSession.geometries[0].get_distance_from_point(rayCastImage.pos, direction)
        if(not math.isinf(rayDistance)):
            scalingFactors.append(rayDistance/pointDistance)

    if(len(scalingFactors)>0):
        scalingFactor = sum(scalingFactors) / float(len(scalingFactors))
    else: 
        scalingFactor = 1
    match.set_scaling_factor(scalingFactor)
    match.triangulate(useCameraPose = True) # determine the 3D points

    R,t = match.get_image2_pos(False)
    return R,t, [match]

def raycast_image_matching(match, geometry):
    """Determines the estimated pose by matching with 1 reference image and raycasting against the 3d scene"""

    #find the best single match for the test image
    match.get_essential_matrix() # Calculate the essential matrix
    match.triangulate(useCameraPose = True) # determine the 3D points
    rayCastImage = match.image1

    #if(len(testSession.geometries) > 0):
    #        # the test sesison has geometry
    #        geometry = testSession.geometries[0]
    #        camera = testImage
    #    
    #    if(len(refSession.geometries) > 0):
    #        # the ref sesison has geometry
    #        geometry = refSession.geometries[0]
    #        camera = refImage
    #
    #    geometry.get_distance_from_point(camera.pos, camera.rot)

    #cast a number of rays on the determined points in the scene
    scalingFactors = []
    for point in match.points3d:
        pointVector = point - (rayCastImage.pos)
        pointDistance = np.linalg.norm(pointVector)
        direction = pointVector / pointDistance
        rayDistance = geometry.get_distance_from_point(rayCastImage.pos, direction)
        if(not math.isinf(rayDistance)):
            scalingFactors.append(rayDistance/pointDistance)

    if(len(scalingFactors)>0):
        filteredOutliers = utils.reject_outliers(np.array(scalingFactors))
        scalingFactor = np.average(filteredOutliers)
    else: 
        scalingFactor = 1
    print("ScalingFactor:", scalingFactor)
    match.set_scaling_factor(scalingFactor)
    match.triangulate(useCameraPose = True) # determine the 3D points

    R,t = match.get_image2_pos(False)
    confidence = match.fidelity
    return R,t



def get_global_position_offset():
    pass


    




