import cv2
import numpy as np
import os

import compareImage

refPath =  "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Hololens/Sessions/session-2021-10-12 14-28-27"#"images/ref/"
testPath = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Android/sessions/session-2021-11-04 16-17-08"#"images/test/"
referenceImages = []

dirname = os.path.dirname(__file__) + "/"

# Import the reference images
print ("these are the reference files")
for file in os.listdir(dirname + refPath):
    print(file)
    referenceImages.append(cv2.imread(dirname + refPath + file,cv2.IMREAD_COLOR))

for refImage in referenceImages:
    print(refImage)

# Import the test image
print("this is the path:")
print(os.listdir(dirname + testPath)[0])
testImage = cv2.imread(dirname + testPath + os.listdir(dirname + testPath)[0], cv2.IMREAD_COLOR)
print ("this is the test file:")
print (testImage)

# Convert the images 

class BestResult:
    def __init__(self,refImage, transImage, transMatrix, matchAmount):
        self.refImage = refImage
        self.transImage = transImage
        self.transMatrix = transMatrix
        self.matchAmount = matchAmount

results = []
bestResult = BestResult(0,0,0,0)
nrCheck = 0

for refImage in referenceImages:
    imReg, h, matchAmount = compareImage.CompareImage(testImage, refImage)
    results.append(BestResult(refImage,imReg,h,matchAmount))

    # if the new match amount is higher then the previous one, set this to the new best result
    if(bestResult.matchAmount < matchAmount):
        bestResult = BestResult(refImage,imReg,h,matchAmount)
    
    nrCheck +=1
    print("check complete: ")
    print(nrCheck)

print("this is the best image:")
print("Estimated homography : \n",  h)
cv2.imshow("bestRefImage",bestResult.refImage)
cv2.imshow("AlignedImage",bestResult.transImage)
cv2.waitKey(0)