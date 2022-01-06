# The 2D position is based on OpenCV computer vision and the Essential matrix that can be calculated with matched Features
# Compare all the test images against all the reference images
    # Find which session has the highest match rate
    # Find which Image has the highest match rate
    # Calculate the transformation between the two images
    # Calculate the inverse transformation to give the test data a Reference global position
import os
import cv2
from compareImage import compare_image, ImageMatch
import transform
from session import Session



def get_2D_transformation(testSession : Session, refSessions : "list[Session]", method: float):
    """Returns the estimated Global transform offset in relation to the best reference session in the list
    Params: method: 1,2,3"""
    bestSessionResults = []
    #find the image with the best match rate
    for referenceSession in refSessions:
        results = compare_session(testSession, referenceSession)
        bestSessionResults.append(results)

    # calculate the transformation based on the match rate

    return bestSessionResults


def compare_session(testSession : Session, refSession : Session):
    """Checks a test session against a reference session, returns the 3 best matching images"""
    results = [] # a list of all the results
    bestResults = [] # a list of the 2 best results

    nrCheck = 0
    totalCheck = len(testSession.imageTransforms) * len(refSession.imageTransforms)

    print("Starting Comparing:", len(testSession.imageTransforms), "Against", len(refSession.imageTransforms), "Images")

    # loop over every test image in the session, find the 2 best referenc images and keep them
    for testImageTransform  in testSession.imageTransforms:

        testImageTransform.image = cv2.imread(testImageTransform.path,cv2.IMREAD_COLOR)
        bestResults = []

        for refImageTransform in refSession.imageTransforms:

            refImageTransform.image = cv2.imread(refImageTransform.path,cv2.IMREAD_COLOR)

            newResult = compare_image(testImageTransform, refImageTransform)
            results.append(newResult)
            #print(newResult.__dict__)

            # check if the newResult is in the top2 of results
            if(len(bestResults) < 2):
                bestResults.append(newResult)
            elif(newResult.matchScore > (min(bestResults[0].matchScore, bestResults[1].matchScore))):
                #the matchscore is higher than atleast on of the other results
                if(bestResults[0].matchScore > bestResults[1].matchScore):
                    bestResults = [newResult, bestResults[0]]
                else:
                    bestResults = [newResult, bestResults[1]]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

        # once The 2 best results are determined, calculate the transformation

    return bestResults

def cross_reference_matching(testImage, refSession):
    """Tries to find 2 seperate reference images that hav the best matches against a test Image"""
    
    bestResults = get_best_matches(testImage, refSession.imageTransforms, 2)
    
    #determine the transformation based on the 2 best images
    newPos, rot1, pos1,pos2, minimum2 = transform.triangulate_session(
        bestResults[0].refImage,bestResults[1].refImage, 
        bestResults[0].essentialMatrix,bestResults[1].essentialMatrix)

    return newPos # the position of the test image transform in reference session space

def incremental_matching(testImage, refSession):
    """tries to determine the pose by first matching reference images to create the initial 3D points"""

    #find the 3 highest linked matches
    #find the best single match for 
    bestResult = get_best_matches(testImage, refSession.imageTransforms)
    #find the best result for the matched reference image
    newList = refSession.imageTransforms.remove(bestResult.refImage)
    bestRefResult = get_best_matches(bestResult.refImage, newList)
    #Calculate the 3D points in the scene with the know real world locations of the 2 reference images
    
    cv2.triangulatePoints()
    #TODO


def get_best_matches(testImage, refImages, nr = 1):
    results = [] # a list of all the results
    bestResults = [] # a list of the best results
    nrCheck = 0
    totalCheck = len(refImages)

    for refImage in refImages:
            newResult = compare_image(testImage, refImage)
            results.append(newResult)

            # check if the newResult is in the top2 of results
            bestResults.append(newResult)
            bestResults = sorted(bestResults, key= lambda x: x.matchScore)
            if(len(bestResults) > nr):
                bestResults = bestResults[:(nr-1)]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

    return bestResults
