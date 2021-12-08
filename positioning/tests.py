import quaternion
import compareImage as ci
import numpy as np
import cv2
import session

from transform import ImageTransform, get_camera_from_E, triangulate_session,get_position
from session import Session
import positioning2D


refPath = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-11 Testbuilding Campus/RAW Data/android/session-2021-11-25 16-19-26"
testSessionDirLocation = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Hololens/Sessions/session-2021-10-12 14-28-27"

sessions = session.find_close_sessions(refPath, np.array([0,0,0]), 20)
testSession = Session().from_path(testSessionDirLocation)

testImage = sessions[0].imageTransforms[1]
refImage1 = sessions[0].imageTransforms[0]
refImage2 = sessions[0].imageTransforms[2]

#score, E, imMatches = ci.compare_image(testImage, refImage)

#cv2.imshow('refimage',refImage.get_cv2_image())
#cv2.imshow('testimage',testImage.get_cv2_image())
#imMatchesOrb = cv2.drawMatches(testImage.get_cv2_image(), keypoints1Orb1, refImage1.get_cv2_image(), keypoints2Orb1, matchesOrb1, None)
#cv2.imshow('matchesOrb',imMatchesOrb)
#

def do_stuff(refImage,i):

    matchScore, matchesOrb, keypoints1Orb, keypoints2Orb = ci.match_bfm_orb(testImage, refImage)
    print("MatchScore:",matchScore)
    
    E, E1,F,pt1,pt2, imMatches = ci.calculate_transformation_matrix(testImage, refImage,matchesOrb, keypoints1Orb, keypoints2Orb)
    R1, R2, t1, t2 = get_camera_from_E(E1)
    #ci.draw_epilines(cv2.cvtColor(testImage.image, cv2.COLOR_BGR2GRAY),cv2.cvtColor(refImage.image, cv2.COLOR_BGR2GRAY), pt1, pt2,F)

    #print("Image")
    #print("E:", E)
    #print("E1:",E1)
    #print("estimated Rotations:")
    #print(R1)
    #print(R2)
    #print("estimated translations:")
    #print(t1)
    #print(t2)
    cv2.imshow("image" + str(i),imMatches)
    cv2.waitKey(0)
    return E1, t1
#positioning2D.get_2D_transformation(sessions[0], sessions)

Essential1, t1 = do_stuff(refImage1,1)
Essential2, t2 = do_stuff(refImage2,2)

newPos, rot1, pos1, pos2, scale = triangulate_session(refImage1, refImage2, Essential1, Essential2)

print("refImage1 Pos:", refImage1.pos)
print("base translation:", t1)
print("refImage2 Pos:", refImage2.pos)
print("base translation:", t2)
print("testImage Pos:", testImage.pos)
print("Calculated test position:", newPos)
print("Calculated pos 1:", pos1)
print("Calculated pos 2:", pos2)
print("Calculated scale:", scale)
#print(newPos, rot1, scale, translation1)

#cv2.waitKey(0)