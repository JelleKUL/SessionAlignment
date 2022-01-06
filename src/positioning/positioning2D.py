# The 2D position is based on OpenCV computer vision and the Essential matrix that can be calculated with matched Features
# Compare all the test images against all the reference images
    # Find which session has the highest match rate
    # Find which Image has the highest match rate
    # Calculate the transformation between the two images
    # Calculate the inverse transformation to give the test data a Reference global position
import os
import cv2
from compareImage import compare_image
import transform
from session import Session

class BestResult:
    testImage = 0
    refImage = 0
    transMatrix = 0
    matchScore = 0
    featureMatches = None

    def __init__(self,testImage, refImage, transMatrix, matchScore):
        self.testImage = testImage
        self.refImage = refImage
        self.transMatrix = transMatrix
        self.matchScore = matchScore

def get_2D_transformation(testSession : Session, refSessions : "list[Session]"):
    """Returns the estimated Global transform offset in relation to the best reference session in the list"""
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

            matchScore, essentialMatrix, comparisonImage = compare_image(testImageTransform, refImageTransform)
            newResult = BestResult(testImageTransform,refImageTransform,essentialMatrix,matchScore)
            newResult.featureMatches = comparisonImage
            results.append(newResult)
            #print(newResult.__dict__)

            # check if the newResult is in the top2 of results
            if(len(bestResults) < 2):
                bestResults.append(newResult)
            elif(matchScore > (min(bestResults[0].matchScore, bestResults[1].matchScore))):
                #the matchscore is higher than atleast on of the other results
                if(bestResults[0].matchScore > bestResults[1].matchScore):
                    bestResults = [newResult, bestResults[0]]
                else:
                    bestResults = [newResult, bestResults[1]]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

        # once The 2 best results are determined, calculate the transformation

    return bestResults