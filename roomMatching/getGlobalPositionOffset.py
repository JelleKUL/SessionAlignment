from imageTransform import ImageTransform
import numpy as np
import quaternion
from transform import Transform


def DictToQuaternion(rotation):
    return np.quaternion(rotation["w"],rotation["x"],rotation["y"],rotation["z"])

def DictToNPVector3(dict):
    return np.array([dict["x"],dict["y"],dict["z"]])

def GetGlobalPositionOffset(testImageTransform, refImageTransform, refGlobalTransform, transformationMatrix):
    testGlobalTransform = 0

    # Put the refImage in global coordinate system using the global transform
    newPos = DictToNPVector3(refImageTransform.pos) + DictToNPVector3(refGlobalTransform.pos)
    newRot = DictToQuaternion(refImageTransform.rot) * DictToQuaternion(refGlobalTransform.rot)
    globalRefImageTransform = Transform(refImageTransform.id,newPos ,newRot,1)

    #transform the new globalrefImage to the testImage
    globalTestImageTransform = Transform(testImageTransform.id, 0,0,1)

    #print("array:" + str(np.array(transformationMatrix)) + ", position:" + str(np.transpose(globalRefImageTransform.pos)))
    globalTestImageTransform.pos =  np.matmul(np.array(transformationMatrix), np.transpose(globalRefImageTransform.pos))


    testGlobalTransform = globalTestImageTransform.pos - DictToNPVector3(testImageTransform.pos)

    print("The reference Image Global position: " + str(newPos))
    print("The reference Image Global rotation: " + str(newRot))
    print("The transformationMatrix: \n" + str(transformationMatrix))
    print("The test Image local position: " + str(DictToNPVector3(testImageTransform.pos)))
    print("The test Image Global position: " + str(globalTestImageTransform.pos))
    print("The Calculated test Global offset:" + str(testGlobalTransform))


    return testGlobalTransform
    