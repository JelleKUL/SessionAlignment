import math

import cv2
import numpy as np

MAX_FEATURES = 20000
MAX_MATCHES = 1000

class ImageMatch:
    """This class stores all the data of 2 matched images"""

    image1 = None           # the train/reference ImageTransform
    image2 = None           # the query ImageTransform
    matches = []            # [N] the matches between the 2 images [kp1_n,kp2_m, distance] x N
    mask = []               # [1xN] array of the inlier matches
    inliers1 = []           # [Nx2] array of all the pixel values of image1.keypoints
    inliers2 = []           # [Nx2] array of all the pixel values of image2.keypoints
    points3d = []           # [Nx3] array of all the 3D points
    matchScore = math.inf   # The match score of the image match (lower is better)
    essentialMatrix = []    # [3x3] matrix E
    translation = []        # [3,1] matrix t
    rotationMatrix = []     # [3,3] matrix R
    fidelity = 0            # a measurement for the quality of the match

    def __init__(self, image1, image2):
        """Initialise the class with 2 images"""

        self.image1 = image1
        self.image2 = image2
    
    def find_matches(self):
        """Finds matches between the 2 images"""

        # get cv2 ORb features
        self.image1.get_cv2_features(MAX_FEATURES)
        self.image2.get_cv2_features(MAX_FEATURES)
        # Match features.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self.image1.descriptors, self.image2.descriptors, None)
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
        
        # Extract location of good matches
        points1 = np.zeros((len(self.matches), 2), dtype=np.float32)
        points2 = np.zeros((len(self.matches), 2), dtype=np.float32)
        for i, match in enumerate(self.matches):
            points1[i, :] = self.image1.keypoints[match.queryIdx].pt
            points2[i, :] = self.image2.keypoints[match.trainIdx].pt

        _, E, R, t, mask = cv2.recoverPose(points1= points1,
                                           points2= points2, 
                                           cameraMatrix1= self.image1.get_camera_matrix(),
                                           distCoeffs1= None,
                                           cameraMatrix2= self.image2.get_camera_matrix(), 
                                           distCoeffs2= None)

        # assign the values
        self.inliers1 = points1[mask.ravel()==1]
        self.inliers2 = points2[mask.ravel()==1]
        self.mask = mask
        self.essentialMatrix = E
        self.rotationMatrix = R
        self.translation = t
        return E

    def draw_image_matches(self):
        """Draws the matches on the 2 images and returns a cv2 image"""

        imMatches = cv2.drawMatches(self.image1.get_cv2_image(),self.image1.keypoints,
                                    self.image2.get_cv2_image(),self.image2.keypoints,
                                    self.matches,None, flags=2)
        return imMatches

    def draw_image_inliers(self):
        """Draws the inliers on the 2 images"""

        matchesMask = self.mask.ravel().tolist()
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        imMatches = cv2.drawMatches(self.image1.get_cv2_image(),self.image1.keypoints,
                                    self.image2.get_cv2_image(),self.image2.keypoints,
                                    self.matches,None, **draw_params)
        return imMatches

    def get_projectionMatrices(self):
        """Returns 2 projection matrices, the second is the transformation in relation to the first"""

        cam1 = self.image1.get_camera_matrix() @ np.hstack((np.eye(3), np.zeros((3,1)))) #the identity transformation matrix
        cam2 = self.image2.get_camera_matrix() @ np.hstack((self.rotationMatrix,self.translation))
        return cam1, cam2

    def triangulate(self, useCameraPose = False):
        """triangulates the matched points and returns 3d points in the first camera space"""

        matxs = self.get_projectionMatrices()
        points3d = cv2.triangulatePoints(matxs[0], matxs[1], self.inliers1.T, self.inliers2.T) #4xn array of 3D homogenious points
        points3d = points3d /points3d[3] # normalise the points
        
        if(useCameraPose): # transform all the points by the transformation matrix of the first camera to place them in session space
            points3d = self.image1.get_transformation_matrix() @ points3d
        self.points3d = np.array(points3d[:3]).T # remove homogenious coordinate and reshape to [Nx3]
        return self.points3d
    
    def calculate_scaling_factor(self):
        """Uses the real world distance to scale the translationvector"""

        scalingFactor = np.linalg.norm(self.image2.pos - self.image1.pos)
        self.translation = self.translation * scalingFactor / np.linalg.norm(self.translation)
        return scalingFactor

    def get_pnp_pose(self, OtherMatch):
        """Calculates the pose of a third camera with matched """
        
        # match old descriptors against the descriptors in the new view
        matches = self.find_matches()
        points_3D = []
        points_2D = []

        # build corresponding array of 2D points and 3D points
        for match in matches:
            old_image_idx, new_image_kp_idx, old_image_kp_idx = match.imgIdx, match.queryIdx, match.trainIdx
            j = 0
            for i, oldmatch in enumerate(OtherMatch.matches):
                if(OtherMatch.mask[i] == 1): #the match has a 3d point
                    if(oldmatch.queryIdx == match.trainIdx):
                    
                        #the index of the old reference image match is also in the new test image
                        
                        # obtain the 2D point from match
                        point_2D = np.array(self.image2.keypoints[new_image_kp_idx].pt).T.reshape((1, 2))
                        points_2D.append(point_2D)

                        # obtain the 3D point from the point_map
                        point_3D = OtherMatch.points3d[j]
                        points_3D.append(point_3D)
                        print("point",j,":\n2D:",point_2D, "\n3D:",point_3D)
                    j+=1

        # compute new pose using solvePnPRansac
        _, R, t, _ = cv2.solvePnPRansac(np.array(points_3D), np.array(points_2D), self.image2.get_camera_matrix(), None,
                                        confidence=0.99, reprojectionError=8.0, flags=cv2.SOLVEPNP_DLS)
        R, _ = cv2.Rodrigues(R)
        return R, t
   
    def draw_epilines(self):
        """Draw epilines in the 2 images"""

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(self.image2.keypoints.reshape(-1,1,2), 2,self.fundamentalMatrix)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = self.drawlines(self.image1.get_cv_image(),self.image2.get_cv_image(),lines1,self.image1.keypoints,self.image2.keypoints)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(self.image1.keypoints.reshape(-1,1,2), 1,self.fundamentalMatrix)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = self.drawlines(self.image2.get_cv_image(),self.image1.get_cv_image(),lines2,self.image2.keypoints,self.image1.keypoints)

        return img5, img6

    def drawlines(self, img1,img2,lines,pts1,pts2):
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
