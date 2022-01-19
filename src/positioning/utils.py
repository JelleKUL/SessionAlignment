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