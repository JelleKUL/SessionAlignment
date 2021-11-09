import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# this script compares 2 images and finds their matching rate, if the matching rate is high enough, we can estimate the 3D transformation

# Compares 2 images and returns the transformation and likelyhood of being a good match
def CompareImage(imTest,imRef):
    
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(imTest, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(imRef, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    #matcher2 = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)

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

    return im1Reg, h, numGoodMatches

# calculate the transformationmatrix between 2 images using the matched features
def CalculateTransformation(im1, im2, matches):
    pass