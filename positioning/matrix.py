import numpy as np
import cv2
import math

# Calculate the Camera matrix with the vertical fov
def camera_matrix(fov, referenceImagePath):
    """Calculate the Camera matrix with the vertical fov"""
    image = cv2.imread(referenceImagePath, cv2.IMREAD_COLOR)

    imageSize = [image.shape[1]/2,image.shape[0]/2] #width, height
    aspectRatio = imageSize[0] / imageSize[1]

    a_x = fov * aspectRatio
    a_y = fov
    
    f_x = imageSize[0] / math.tan(math.radians(a_x) / 2 )
    f_y = imageSize[1] / math.tan(math.radians(a_y) / 2)

    print("imageSize: " + str(imageSize))

    return np.array([[f_x, 0, imageSize[0]], [0, f_y, imageSize[1]],[0,0,1]])

#path = "/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-07 House Trekweg/RAW Data/Hololens/Sessions/session-2021-10-12 14-28-27/img-2021-10-12 14-28-27.jpg"
#print(GetCameraMatrix(29,path))

def fundamental_matrix():
    pass

def essential_matrix():
    pass