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
    results = []
    bestResult = BestResult(0,0,0,0)
    nrCheck = 0
    totalCheck = len(testSession.imageTransforms) * len(refSession.imageTransforms)

    for refImageTransform in refSession.imageTransforms:

        refImageTransform.image = cv2.imread(refImageTransform.path,cv2.IMREAD_COLOR)

        for testImageTransform in testSession.imageTransforms:

            testImageTransform.image = cv2.imread(testImageTransform.path,cv2.IMREAD_COLOR)
            print("These are the image paths:")
            print(testImageTransform.path)
            print(refImageTransform.path)

            matchScore, essentialMatrix = compare_image(testImageTransform, refImageTransform)
            results.append(BestResult(testImageTransform,refImageTransform,essentialMatrix,matchScore))

            # if the new match amount is higher then the previous one, set this to the new best result
            if(bestResult.matchAmount < matchScore):
                bestResult = results[-1]
            
            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

    print("This is the best image with " + str(bestResult.matchAmount) + " match amount" )
    print("Estimated homography : \n",  bestResult.transMatrix)

    return bestResult