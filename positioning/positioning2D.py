# The 2D position is based on OpenCV computer vision and the Essential matrix that can be calculated with matched Features
# Compare all the test images against all the reference images
    # Find which session has the highest match rate
    # Find which Image has the highest match rate
    # Calculate the transformation between the two images
    # calculate the inverse transformation to give the test data a Reference global position
import os
import cv2
from compareImage import compare_image
import transform
import matrix
from session import Session

class BestResult:
        def __init__(self,testImage, refImage, transMatrix, matchAmount):
            self.testImage = testImage
            self.refImage = refImage
            self.transMatrix = transMatrix
            self.matchAmount = matchAmount

def get_2D_transformation(testSession : Session, refSessions : list[Session]):
    """Calculate the estimated camera transformation based on Images from sessions"""
    
    #find the image with the best match rate
    bestSession = None
    bestResult = BestResult(0,0,0,0)

    for referenceSession in refSessions:
        result = compare_session(testSession, referenceSession)
        if(result.matchAmount > bestResult.matchAmount):
            bestResult = result
            bestSession = referenceSession

    print("this is the very best session:" + str(bestSession.__dict__))
    # calculate the transformation based on the match rate

    pass


def compare_session(testSession : Session, refSession : Session):
    """Checks a test session against a reference session, returns the 3 best matching images"""
    results = [] # a list of all the results
    bestResults = [] # a list of the 2 best results

    nrCheck = 0
    totalCheck = len(testSession.imageTransforms) * len(refSession.imageTransforms)

    # loop over every test image in the session, find the 2 best referenc images and keep them
    for testImageTransform  in testSession.imageTransforms:

        testImageTransform.image = cv2.imread(testImageTransform.path,cv2.IMREAD_COLOR)
        bestResults = []

        for refImageTransform in refSession.imageTransforms:

            refImageTransform.image = cv2.imread(refImageTransform.path,cv2.IMREAD_COLOR)

            matchScore, essentialMatrix = compare_image(testImageTransform, refImageTransform)
            newResult = BestResult(testImageTransform,refImageTransform,essentialMatrix,matchScore)
            results.append(newResult)
            print("Matchscore: " , matchScore)
            print("EssentialMatrix: \n", essentialMatrix)

            # check if the newResult is in the top2 of results
            if(len(bestResults) < 2):
                bestResults.append(newResult)
            elif(matchScore > (min(bestResults[0].matchScore, bestResults[1].matchScore))):
                #the matchscore is higher than atleast on of the other results
                if(bestResults[0].matchScore > bestResults[1].matchScore):
                    bestResults = [newResult, bestResults[0].matchScore]
                else:
                    bestResults = [newResult, bestResults[1].matchScore]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

        # once The 2 best results are determined, calculate the transformation

    print("This is the best image with " + str(bestResult.matchScore) + " match amount" )
    print("Essential matrix : \n",  bestResult.transMatrix)

    return bestResult