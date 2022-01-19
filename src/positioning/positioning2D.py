# The 2D position is based on OpenCV computer vision and the Essential matrix that can be calculated with matched Features
# Compare all the test images against all the reference images
    # Find which session has the highest match rate
    # Find which Image has the highest match rate
    # Calculate the transformation between the two images
    # Calculate the inverse transformation to give the test data a Reference global position
import os

import cv2
import numpy as np
import quaternion
from scipy import optimize

from session import Session
from imagetransform import ImageTransform
import utils


def get_2D_transformation(testSession : Session, refSessions : "list[Session]", method: float):
    """Returns the estimated Global transform offset in relation to the best reference session in the list
    Params: 
        method: 
            1: Cross referencing 2 refenrence images with one test sesison image
            2: Matching 2 reference images to retrieve 3D points, then pnp estimation of test image
            3: Matching 1 reference image from a session with 3D data, with a test image, getting the global pos by raycasting"""
    bestSessionResults = []
    #find the image with the best match rate
    for referenceSession in refSessions:
        results = compare_session(testSession, referenceSession)
        bestSessionResults.append(results)

    # calculate the transformation based on the match rate

    return bestSessionResults


def compare_session(testSession : Session, refSession : Session):
    """Checks a test session against a reference session, returns the 3 best matching images"""
    results = [] # a list of all the results
    bestResults = [] # a list of the 2 best results

    nrCheck = 0
    totalCheck = len(testSession.imageTransforms) * len(refSession.imageTransforms)

    print("Starting Comparing:", len(testSession.imageTransforms), "Against", len(refSession.imageTransforms), "Images")

    # loop over every test image in the session, find the 2 best referenc images and keep them
    for testImageTransform  in testSession.imageTransforms:

        testImageTransform.image = cv2.imread(testImageTransform.path,cv2.IMREAD_COLOR)
        bestResults = []

        for refImageTransform in refSession.imageTransforms:

            refImageTransform.image = cv2.imread(refImageTransform.path,cv2.IMREAD_COLOR)

            newResult = compare_image(testImageTransform, refImageTransform)
            results.append(newResult)
            #print(newResult.__dict__)

            # check if the newResult is in the top2 of results
            if(len(bestResults) < 2):
                bestResults.append(newResult)
            elif(newResult.matchScore > (min(bestResults[0].matchScore, bestResults[1].matchScore))):
                #the matchscore is higher than atleast on of the other results
                if(bestResults[0].matchScore > bestResults[1].matchScore):
                    bestResults = [newResult, bestResults[0]]
                else:
                    bestResults = [newResult, bestResults[1]]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

        # once The 2 best results are determined, calculate the transformation

    return bestResults

def cross_reference_matching(testImage, refSession):
    """Tries to find 2 seperate reference images that hav the best matches against a test Image"""
    
    bestResults = get_best_matches(testImage, refSession.imageTransforms, 2)
    
    #determine the transformation based on the 2 best images
    newPos, rot1, pos1,pos2, minimum2 = triangulate_session(
        bestResults[0].refImage,bestResults[1].refImage, 
        bestResults[0].essentialMatrix,bestResults[1].essentialMatrix)

    return newPos # the position of the test image transform in reference session space

def incremental_matching(testImage, refSession):
    """tries to determine the pose by first matching reference images to create the initial 3D points"""

    #find the 3 highest linked matches
    #find the best single match for 
    bestResult = get_best_matches(testImage, refSession.imageTransforms)
    #find the best result for the matched reference image
    newList = refSession.imageTransforms.remove(bestResult.refImage)
    bestRefResult = get_best_matches(bestResult.refImage, newList)
    #Calculate the 3D points in the scene with the know real world locations of the 2 reference images
    
    cv2.triangulatePoints()
    #TODO


def get_best_matches(testImage, refImages, nr = 1):
    results = [] # a list of all the results
    bestResults = [] # a list of the best results
    nrCheck = 0
    totalCheck = len(refImages)

    for refImage in refImages:
            newResult = compare_image(testImage, refImage)
            results.append(newResult)

            # check if the newResult is in the top2 of results
            bestResults.append(newResult)
            bestResults = sorted(bestResults, key= lambda x: x.matchScore)
            if(len(bestResults) > nr):
                bestResults = bestResults[:(nr-1)]

            nrCheck +=1
            print(str(nrCheck) + "/" + str(totalCheck) + " checks complete")

    return bestResults

def get_global_position_offset(testImageTransform: ImageTransform, refImageTransform: ImageTransform, refGlobalTransform: np.array, transformationMatrix: np.array):
    testGlobalTransform = 0

    # Put the refImage in global coordinate system using the global transform
    newPos = refImageTransform.pos + refGlobalTransform.pos
    newRot = refImageTransform.rot * refGlobalTransform.rot
    #globalRefImageTransform = Transform(refImageTransform.id,newPos ,newRot,1)

    #transform the new globalrefImage to the testImage
    #globalTestImageTransform = Transform(testImageTransform.id, 0,0,1)

    #print("array:" + str(np.array(transformationMatrix)) + ", position:" + str(np.transpose(globalRefImageTransform.pos)))
    #globalTestImageTransform.pos =  np.matmul(np.array(transformationMatrix), np.transpose(globalRefImageTransform.pos))


    #testGlobalTransform = globalTestImageTransform.pos - dict_to_np_vector3(testImageTransform.pos)

    print("The reference Image Global position: " + str(newPos))
    print("The reference Image Global rotation: " + str(newRot))
    print("The transformationMatrix: \n" + str(transformationMatrix))
    print("The test Image local position: " + str(utils.dict_to_np_vector3(testImageTransform.pos)))
    #print("The test Image Global position: " + str(globalTestImageTransform.pos))
    print("The Calculated test Global offset:" + str(testGlobalTransform))


    return testGlobalTransform

def get_session_scale(image1: ImageTransform, image2: ImageTransform, transMatrix):
    """Calculates the pixel scale of a transformation matrix"""

    translation, rot = get_translation(transMatrix)
    if (np.linalg.norm(translation) == 0): return 0
    scale = np.linalg.norm(image1.pos - image2.pos) / np.linalg.norm(translation)
    return scale

def triangulate_session(image1: ImageTransform, image2: ImageTransform, transMatrix1, transMatrix2):
    """Calculates a new transform based on 2 Essential transformations"""
    
    translation1, rot1 = get_translation(transMatrix1)
    translation2, rot2 = get_translation(transMatrix2)
    
    def get_distance_array(x):
        pos1 = get_position(x[0], image1, translation1)
        pos2 = get_position(x[1], image2, translation2)
        return np.linalg.norm(pos2-pos1)

    minimum = optimize.fmin(get_distance_array, [1,1])

    pos1 = get_position(minimum[0], image1, translation1)
    pos2 = get_position(minimum[1], image2, translation2)
    newPos =(pos1 + pos2)/2
    return newPos, rot1, pos1,pos2, minimum
    

def get_position(scaleFactor, imageTransform: ImageTransform, translation : np.array):
    """Returns the translation in function of a scale factor"""
    newPosition = imageTransform.pos + scaleFactor * (quaternion.as_rotation_matrix(imageTransform.rot) @ translation).T
    return newPosition

# helper functions (source https://github.com/harish-vnkt/structure-from-motion)
def check_pose(E):
    """Retrieves the rotation and translation components from the essential matrix by decomposing it and verifying the validity of the 4 possible solutions"""

    R1, R2, t1, t2 = get_camera_from_E(E)  # decompose E
    if not check_determinant(R1):
        R1, R2, t1, t2 = get_camera_from_E(-E)  # change sign of E if R1 fails the determinant test

    return R1, R2, t1, t2

def get_translation(E):
    R1, R2, t1, t2 = get_camera_from_E(E)

    return t1.T, R1

def get_camera_from_E(E):
    """Calculates rotation and translation component from essential matrix"""

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_t = W.T
    u, w, vt = np.linalg.svd(E)

    R1 = u @ W @ vt
    R2 = u @ W_t @ vt
    t1 = u[:, -1].reshape((3,1))
    t2 = - t1
    return R1, R2, t1, t2

def check_determinant(R):
    """Validates using the determinant of the rotation matrix"""

    if np.linalg.det(R) + 1.0 < 1e-9:
        return False
    else:
        return True

def check_triangulation(points, P):
    """Checks whether reconstructed points lie in front of the camera"""

    P = np.vstack((P, np.array([0, 0, 0, 1])))
    reprojected_points = cv2.perspectiveTransform(src=points[np.newaxis], m=P)
    z = reprojected_points[0, :, -1]
    if (np.sum(z > 0)/z.shape[0]) < 0.75:
        return False
    else:
        return True
