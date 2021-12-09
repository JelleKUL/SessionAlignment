import session
from session import Session
import compareImage as ci
from transform import ImageTransform, get_camera_from_E, triangulate_session,get_position
import cv2
import numpy as np


sessionPath = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-11 Testbuilding Campus/RAW Data/Hololens/session-2021-11-25 16-09-47"

refSession = Session().from_path(sessionPath)

image1 = refSession.imageTransforms[1]
image2 = refSession.imageTransforms[15]

print(image1)
print (image2)

#score, E, imMatch = ci.compare_image(image1, image2)

matchScore, matchesOrb, keypoints1Orb, keypoints2Orb = ci.find_matches(image1, image2)
E, E1,F,pt1,pt2, imMatches = ci.calculate_transformation_matrix(image1, image2,matchesOrb, keypoints1Orb, keypoints2Orb)

R1, R2, t1, t2 = get_camera_from_E(E)
ci.draw_epilines(cv2.cvtColor(image1.image, cv2.COLOR_BGR2GRAY),cv2.cvtColor(image2.image, cv2.COLOR_BGR2GRAY), pt1, pt2,F)

def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3] @ X
    X2 = T_2w[:3] @ X
    return X[:3], X1, X2 

#cv2.waitKey(0)
#newPos, rot1, pos1, pos2, scale = triangulate_session(image1, image2, E, Essential2)
