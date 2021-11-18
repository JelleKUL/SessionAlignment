import compareImage as ci
import numpy as np
import cv2
import session

from transform import ImageTransform
from session import Session
import positioning2D

#refPath =  "roomMatching/images/ref/PXL_20211012_135353859.jpg"
#testPath = "roomMatching/images/ref/PXL_20211012_135358300.jpg"

#matchScore, matches, keypoints1, keypoints2 = ci.find_matches(cv2.imread(testPath), cv2.imread(refPath))
#match, mat = ci.compare_image(ImageTransform(0,0,0,42, refPath), ImageTransform(0,0,0,42, testPath))

#print("this is the match Rate:" + str(match))
#print(matches)
#print("This is the matrix:")
#print (mat)

sessions = session.find_close_sessions("/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Android", np.array([10,10,0]), 10)

for sessionData in sessions:
    print("Bounding box:")
    print(sessionData.get_bounding_radius())

positioning2D.get_2D_transformation(sessions[0], sessions)