import compareImage as ci
import numpy as np
import cv2
import session

from transform import ImageTransform

refPath =  "roomMatching/images/ref/PXL_20211012_135353859.jpg"
testPath = "roomMatching/images/ref/PXL_20211012_135358300.jpg"

#matchScore, matches, keypoints1, keypoints2 = ci.find_matches(cv2.imread(testPath), cv2.imread(refPath))
match, mat = ci.compare_image(ImageTransform(0,0,0,42, refPath), ImageTransform(0,0,0,42, testPath))

print("this is the match Rate:" + str(match))
#print(matches)
print("This is the matrix:")
print (mat)

session.find_close_sessions("/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Android", np.array([0,0,0]), 10)