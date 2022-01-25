"""A collections of utility functions for the package"""

import numpy as np
import quaternion

def dict_to_quaternion(dict):
    return np.quaternion(dict["w"],dict["x"],dict["y"],dict["z"])

def dict_to_np_vector3(dict):
    return np.array([dict["x"],dict["y"],dict["z"]])

#TODO make generic for different types
def convert_to_open3d(pos : np.array, rot : np.array):
    "converts the stored coordinates in (right, up, forward) to the open3D standard (right, down, forward)"
    newPos = pos #* np.array([1,-1,1])
    newRot = rot #quaternion.from_float_array(quaternion.as_float_array(rot) * np.array([-1,1,-1,1]))
    return newPos, newRot


# Average multiple quaternions with specific weights
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
#source: https://github.com/christophhagen/averaging-quaternions/
def weighted_average_quaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))
    weightSum = 0
    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]
    # scale
    A = (1.0/weightSum) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)