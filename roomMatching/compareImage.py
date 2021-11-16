import cv2
import numpy as np
import cameraMatrixCalculator as CMC
from imageTransform import ImageTransform

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


# Compares 2 images and returns the transformation and likelyhood of being a good match
def CompareImage(imTest,imRef):
    """Compares 2 images and returns the transformation and likelyhood of being a good match"""
    
    [matchScore, matches, keypoints1, keypoints2] = FindMatches(imTest, imRef)
    essentialMatrix = CalculateTransformationMatrix(imTest, imRef, matches, keypoints1,keypoints2)
    return matchScore, essentialMatrix
   

def FindMatches(imTest, imRef):
    """Finds matches between 2 images"""

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(imTest, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(imRef, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

     # calculate the match score
     # right now, it's just the average distances of the best points
    matchScore = sum(matches.disance) / len(matches)

    return matchScore, matches, keypoints1, keypoints2


def CalculateTransformationMatrix(imTest: ImageTransform, imRef : ImageTransform, matches, keypoints1, keypoints2):
    """Calculates the tranformation between 2 matched images"""
    
    #Calculate the camera matrices
    imTestCam = CMC.GetCameraMatrix(imTest.fov, imTest.path)
    imRefCam = CMC.GetCameraMatrix(imRef.fov, imRef.path)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    #find the fundamental matrix
    F, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = points1[mask.ravel()==1]
    pts2 = points2[mask.ravel()==1]

    E = imTestCam.T @ F @ imRefCam

    return E

def DrawMatches():
    """Generates a new image with all the matches displayed"""

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = imRef.shape
    im1Reg = cv2.warpPerspective(imTest, h, (width, height))