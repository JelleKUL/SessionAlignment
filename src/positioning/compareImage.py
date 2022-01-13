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
    points3d = []
    matchScore = math.inf #lower is better
    fundamentalMatrix = []
    essentialMatrix = []
    rotationMatrix = []
    translation = []
    

    def __init__(self, testImage, refImage):
        self.testImage = testImage
        self.refImage = refImage
    
    def find_matches(self):
        """Finds matches between 2 images"""

        # get cv2 ORb features
        self.testImage.get_cv2_features(MAX_FEATURES)
        self.refImage.get_cv2_features(MAX_FEATURES)

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
        #F, mask = cv2.findFundamentalMat(pointsTest,pointsRef,cv2.RANSAC)
        #E = imRefCam.T @ F @ imTestCam
        #E, mask = cv2.findEssentialMat(pointsTest,pointsRef,imTestCam,cv2.FM_LMEDS)
        #TODO figure this out
        retval, E, R, t, mask = cv2.recoverPose(points1= pointsTest,
                                                points2= pointsRef, 
                                                cameraMatrix1= imTestCam,
                                                distCoeffs1= np.zeros((4,1)),
                                                cameraMatrix2= imRefCam, 
                                                distCoeffs2= np.zeros((1,4)))

        # We select only inlier points
        self.testInliers = pointsTest[mask.ravel()==1]
        self.refInliers = pointsRef[mask.ravel()==1]

        self.mask = mask
        #self.fundamentalMatrix = F
        self.essentialMatrix = E
        self.rotationMatrix = R
        self.translation = t
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

    def get_projectionMatrices(self, useTestPose = False):
        """Returns 2 projection matrices, assuming the First camera is at the origin"""

        if(useTestPose):
            testPos = -np.array([self.testImage.pos]).T
            testRot = self.testImage.get_rotation_matrix().T
        else:
            testPos = np.zeros((3,1))
            testRot = np.eye(3)

        #print("rotation:\n", testRot)
        #print("translation:\n", testPos, "Shape:", testPos.shape)
        cam1 = self.testImage.get_camera_matrix() @ np.hstack((testRot, testPos))
        cam2 = self.refImage.get_camera_matrix() @ np.hstack((self.rotationMatrix @ testRot, self.translation + testPos))

        #R1, R2, t1, t2 = transform.check_pose(self.essentialMatrix)
        #mat1 = np.hstack((R1, t1))
        #mat2 = np.hstack((R1, t2))   
        #mat3 = np.hstack((R2, t1))
        #mat4 = np.hstack((R2, t2))
        #cam2_1 = self.refImage.get_camera_matrix() @ mat1
        #cam2_2 = self.refImage.get_camera_matrix() @ mat2
        #cam2_3 = self.refImage.get_camera_matrix() @ mat3
        #cam2_4 = self.refImage.get_camera_matrix() @ mat4
        return cam1, cam2

    def triangulate(self, useTestPose = False):
        """triangulates the matched points and returns 3d points in the first camera space"""

        matxs = self.get_projectionMatrices(useTestPose)
        points3d = cv2.triangulatePoints(matxs[0], matxs[1], self.testInliers.T, self.refInliers.T)
        points3d = points3d /points3d[3]
        self.points3d = np.array(points3d[:3]).T
        return self.points3d

    
    def calculate_scaling_factor(self):
        """Uses the real world distance to scale the translationvector"""

        scalingFactor = np.linalg.norm(self.refImage.pos - self.testImage.pos)
        self.translation = self.translation * scalingFactor / np.linalg.norm(self.translation)
        return scalingFactor

    def get_pnp_pose(self, OtherMatch):
        """Calculates the pose of a third camera with matched """
        
        # match old descriptors against the descriptors in the new view
        matches = self.find_matches()
        points_3D, points_2D = np.zeros((0, 3)), np.zeros((0, 2))

        # build corresponding array of 2D points and 3D points
        for match in matches:
            old_image_idx, new_image_kp_idx, old_image_kp_idx = match.imgIdx, match.queryIdx, match.trainIdx

            for i, oldmatch in enumerate(OtherMatch.matches):
                if(oldmatch.queryIdx == match.trainIdx):
                    if(OtherMatch.mask[i]):
                        #the index of the old reference image match is also in the new test image

                        # obtain the 2D point from match
                        point_2D = np.array(self.refImage.keypoints[new_image_kp_idx].pt).T.reshape((1, 2))
                        points_2D = np.concatenate((points_2D, point_2D), axis=0)

                        # obtain the 3D point from the point_map
                        point_3D = OtherMatch.points3d[i]
                        points_3D = np.concatenate((points_3D, point_3D), axis=0)

        # compute new pose using solvePnPRansac
        _, R, t, _ = cv2.solvePnPRansac(points_3D[:, np.newaxis], points_2D[:, np.newaxis], self.refImage.get_camera_matrix(), None,
                                        confidence=0.99, reprojectionError=8.0, flags=cv2.SOLVEPNP_DLS)
        R, _ = cv2.Rodrigues(R)
        return R, t



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