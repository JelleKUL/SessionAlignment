import math
import cv2
import numpy as np
import transform
from transform import ImageTransform
from matplotlib import pyplot as plt

MAX_FEATURES = 10000
MAX_MATCHES = 2000

class ImageMatch:
    testImage = None
    refImage = None
    matches = []
    mask = []
    testInliers = []
    refInliers = []
    matchScore = math.inf #lower is better
    fundamentalMatrix = []
    essentialMatrix = []
    

    def __init__(self, testImage, refImage):
        self.testImage = testImage
        self.refImage = refImage
    
    def find_matches(self):
        """Finds matches between 2 images"""

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(cv2.imread(self.testImage.path), cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(cv2.imread(self.refImage.path), cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        self.testImage.keypoints, self.testImage.descriptors = orb.detectAndCompute(im1Gray, None)
        self.refImage.keypoints, self.refImage.descriptors = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self.testImage.descriptors, self.refImage.descriptors, None)

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

        self.matches = matches
        self.matchScore = matchScore
        return matches

    def calculate_transformation_matrix(self):
        """Calculates the tranformation between 2 matched images"""
        
        #Calculate the camera matrices
        imTestCam = self.testImage.get_camera_matrix()
        imRefCam = self.refImage.get_camera_matrix()

        # Extract location of good matches
        pointsTest = np.zeros((len(self.matches), 2), dtype=np.float32)
        pointsRef = np.zeros((len(self.matches), 2), dtype=np.float32)

        for i, match in enumerate(self.matches):
            pointsTest[i, :] = self.testImage.keypoints[match.queryIdx].pt
            pointsRef[i, :] = self.refImage.keypoints[match.trainIdx].pt

        #find the fundamental & essential matrix
        F, mask = cv2.findFundamentalMat(pointsTest,pointsRef,cv2.RANSAC)
        E = imRefCam.T @ F @ imTestCam
        #E, mask = cv2.findEssentialMat(pointsTest,pointsRef,imTestCam,cv2.FM_LMEDS)
        #TODO figure this out

        # We select only inlier points
        self.testInliers = pointsTest[mask.ravel()==1]
        self.refInliers = pointsRef[mask.ravel()==1]

        self.mask = mask
        self.fundamentalMatrix = F
        self.essentialMatrix = E
        return E
            

    def draw_image_matches(self):
        """Draws the matches on the 2 images"""
        imMatches = cv2.drawMatches(self.testImage.get_cv2_image(),self.testImage.keypoints,
                                    self.refImage.get_cv2_image(),self.refImage.keypoints,
                                    self.matches,None, flags=2)
        return imMatches
    def draw_image_inliers(self):
        """Draws the matches on the 2 images"""
        matchesMask = self.mask.ravel().tolist()
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        imMatches = cv2.drawMatches(self.testImage.get_cv2_image(),self.testImage.keypoints,
                                    self.refImage.get_cv2_image(),self.refImage.keypoints,
                                    self.matches,None, **draw_params)
        return imMatches

    def get_keypoints_from_indices(self):
        """Filters a list of keypoints based on the indices given"""

        points1 = np.array([kp.pt for kp in self.testImage.keypoints])[self.testInliers]
        points2 = np.array([kp.pt for kp in self.refImage.keypoints])[self.refInliers]
        return points1, points2

    def get_projectionMatrices(self):
        """Returns 2 projection matrices, assuming the First camera is at the origin"""
        cam1 = self.testImage.get_camera_matrix() @ np.hstack((np.eye(3), np.zeros((3,1))))

        R1, R2, t1, t2 = transform.check_pose(self.essentialMatrix)
        mat1 = np.hstack((R1, t1))
        mat2 = np.hstack((R1, t2))   
        mat3 = np.hstack((R2, t1))
        mat4 = np.hstack((R2, t2))
        cam2_1 = self.refImage.get_camera_matrix() @ mat1
        cam2_2 = self.refImage.get_camera_matrix() @ mat2
        cam2_3 = self.refImage.get_camera_matrix() @ mat3
        cam2_4 = self.refImage.get_camera_matrix() @ mat4
        return cam1, cam2_1, cam2_2, cam2_3, cam2_4


def compare_image(imTest : ImageTransform,imRef: ImageTransform):
    """Compares 2 images and returns a match object containing the transformation and likelyhood of being a good match"""
    
    match = ImageMatch(imTest, imRef)
    match.find_matches()
    print("found matches")
    match.calculate_transformation_matrix()
    print("got matrix")
    return match
   

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