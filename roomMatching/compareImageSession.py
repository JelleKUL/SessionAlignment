import cv2
import numpy as np
import os
import json
import compareImage
from imageTransform import ImageTransform
from bestResult import BestResult

# Compares a reference session of images agains a test session of images
def CompareImageSession(testSessionDataPath, refSessionDataPath):

    refSessionFile = open(refSessionDataPath,)
    refSessionData = json.load(refSessionFile)
    refSessionDirPath = os.path.dirname(refSessionDataPath)
    refImageTransforms = []

    # add all the reference session images and their data to a list to referenc against
    for imageTransform in refSessionData["imageTransforms"]:
        newImageTransform = ImageTransform(imageTransform["id"], imageTransform["pos"],imageTransform["rot"],imageTransform["fov"])
        refImageTransforms.append(newImageTransform)

    print(str(len(refImageTransforms)) + " reference Images found in: " + refSessionDirPath)

    testSessionFile = open(testSessionDataPath,)
    testSessionData = json.load(testSessionFile)
    testSessionDirPath = os.path.dirname(str(testSessionDataPath))
    testImageTransforms = []

    # Do the same for the test images
    for imageTransform in testSessionData["imageTransforms"]:
        newImageTransform = ImageTransform(imageTransform["id"], imageTransform["pos"],imageTransform["rot"],imageTransform["fov"])
        testImageTransforms.append(newImageTransform)

    print(str(len(testImageTransforms)) + " test Images found in: " + testSessionDirPath)

    results = []
    bestResult = BestResult(0,0,0,0,0)
    bestRefImage = 0
    bestTestImage = 0
    nrCheck = 0
    totalCheck = len(refImageTransforms) * len(testImageTransforms)

    for refImageTransform in refImageTransforms:

        refImage = cv2.imread(os.path.join(refSessionDirPath,refImageTransform.id) + ".jpg",cv2.IMREAD_COLOR)

        for testImageTransform in testImageTransforms:

            testImage = cv2.imread(os.path.join(testSessionDirPath,testImageTransform.id) + ".jpg",cv2.IMREAD_COLOR)
            
            imReg, h, matchAmount = compareImage.CompareImage(testImage, refImage)
            results.append(BestResult(testImage,refImage,imReg,h,matchAmount))

            # if the new match amount is higher then the previous one, set this to the new best result
            if(bestResult.matchAmount < matchAmount):
                bestResult = BestResult(testImage, refImage,imReg,h,matchAmount)
                bestRefImage = refImageTransform
                bestTestImage = testImageTransform
            
            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

    print("This is the best image with " + str(bestResult.matchAmount) + " match amount" )
    print("Estimated homography : \n",  bestResult.transMatrix)

    #return bestResult, bestTestImage, bestRefImage

    cv2.imshow("bestRefImage",bestResult.testImage)
    cv2.imshow("bestTestImage",bestResult.refImage)
    cv2.imshow("AlignedImage",bestResult.transImage)
    cv2.waitKey(0)
