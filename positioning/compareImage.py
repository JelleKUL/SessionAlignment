import math
import cv2
import numpy as np
import matrix as mat
from transform import ImageTransform
from matplotlib import pyplot as plt

MAX_FEATURES = 500
MAX_MATCHES = 100

def compare_image(imTest : ImageTransform,imRef: ImageTransform):
    """Compares 2 images and returns the transformation and likelyhood of being a good match"""
    
    [matchScore, matches, keypoints1, keypoints2] = find_matches(imTest, imRef)
    print("found matches")
    E, E1,F, pts1,pts2, imMatches = calculate_transformation_matrix(imTest, imRef, matches, keypoints1,keypoints2)
    print("got matrix")
    return matchScore, E, imMatches
   

def find_matches(imTest : ImageTransform, imRef : ImageTransform):
    """Finds matches between 2 images"""

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(cv2.imread(imTest.path), cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(cv2.imread(imRef.path), cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key = lambda x:x.distance)
    # only use the best features
    if(len(matches) < MAX_MATCHES):
        print("only found", len(matches), "good matches")
        matchScore = math.inf
    else:
        matches = matches[:MAX_MATCHES]
        # calculate the match score
        # right now, it's just the average distances of the best points
        matchScore = 0
        for match in matches:
            matchScore += match.distance
        matchScore /= len(matches)

    return matchScore, matches, keypoints1, keypoints2


def calculate_transformation_matrix(imTest: ImageTransform, imRef : ImageTransform, matches, keypoints1, keypoints2):
    """Calculates the tranformation between 2 matched images"""
    
    #Calculate the camera matrices
    imTestCam = mat.camera_matrix(imTest.fov, imTest.path)
    imRefCam = mat.camera_matrix(imRef.fov, imRef.path)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    #find the fundamental matrix
    F, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS)
    E1, mask1 = cv2.findEssentialMat(points1,points2,imTestCam,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = points1[mask.ravel()==1]
    pts2 = points2[mask.ravel()==1]

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = mask, # draw only inliers
                flags = 2)
    imMatches = cv2.drawMatches(imTest.get_cv2_image(),keypoints1,imRef.get_cv2_image(),keypoints2,matches,None, flags=2)

    E = imTestCam.T @ F @ imRefCam

    return E, E1,F, pts1,pts2, imMatches

def draw_matches(im1, keypoints1, im2, keypoints2, matches, imRef, imTest):
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

def match_bfm_orb(imTest : ImageTransform, imRef : ImageTransform):
    # Convert images to grayscale
    im1Gray = (cv2.imread(imTest.path))
    im2Gray = (cv2.imread(imRef.path))

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    kp1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    kp2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)
    print("Matches:", matches)
    # Sort matches by score
    matches = sorted(matches, key = lambda x:x.distance)

    # only use the best features
    if(len(matches) < MAX_MATCHES):
        print("only found", len(matches), "good matches")
        matchScore = math.inf
    else:
        matches = matches[:MAX_MATCHES]
        # calculate the match score
        #the average distances of the best points
        matchScore = 0
        for match in matches:
            matchScore += match.distance
        matchScore /= len(matches)

    return matchScore, matches, kp1, kp2

def match_bfm_sift(imTest : ImageTransform, imRef : ImageTransform):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(cv2.imread(imTest.path), cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(cv2.imread(imRef.path), cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1Gray,None)
    kp2, des2 = sift.detectAndCompute(im2Gray,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    matches = good
    # only use the best features
    if(len(matches) < MAX_MATCHES):
        print("only found", len(matches), "good matches")
        matchScore = math.inf
    else:
        matches = matches[:MAX_MATCHES]
        # calculate the match score
        #the average distances of the best points
        matchScore = 0
        for match in matches:
            matchScore += match.distance
        matchScore /= len(matches)

    return matchScore, matches, kp1, kp2

def draw_epilines(img1,img2,pts1,pts2,F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
    return img1,img2