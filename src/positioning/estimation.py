from match import Match
import params

import numpy as np
import datetime

class PoseEstimation():
    """Contains an estimated pose and all it's parameters to calculate it's validity"""

    position = [0,0,0]
    rotation = [0,0,0,1]

    matches: Match = []
    method = ""


    def __init__(self, position, rotation, matches) -> None:
        self.position = position
        self.rotation = rotation
        self.matches = matches
        

    def get_confidence(self, session) -> float:
        """Returns the confidence of an estimation based on the matching parameter value from 0 to 1"""
        
        # the starting confidence is 1
        factors = []

        # The match specific parameters
        matchErrorFactor = 1        # the error radius of the match
        matchAmountFactor = 1       # the amount of good matches/inliers

        for match in self.matches: #type: Match
            if(match.matchType == "2d"):
                matchErrorFactor = 1 - (min(params.MAX_ERROR_2D, match.matchError)/params.MAX_ERROR_2D) #remap from 0-MaxError to 1-0
                matchAmountFactor = match.matchAmount / params.MAX_2D_MATCHES
                factors.append([matchErrorFactor, params.ERROR_2D])
                factors.append([matchAmountFactor, params.MATCHES_2D])
            elif(match.matchType == "3d"):
                matchErrorFactor = 1 - (min(params.MAX_ERROR_3D, match.matchError)/params.MAX_ERROR_3D) #remap from 0-MaxError to 1-0
                matchAmountFactor = match.matchAmount / params.MAX_3D_MATCHES
                factors.append([matchErrorFactor, params.ERROR_3D])
                factors.append([matchAmountFactor, params.MATCHES_3D])

        

        # The method Parameters
        methodFactor = 1
        if (self.method == "leastDistance"):
            methodFactor *= params.LEAST_DISTANCE
        elif (self.method == "incremental"):
            methodFactor *= params.INCREMENTAL
        elif (self.method == "raycasting"):
            methodFactor *= params.RAYCASTING
        factors.append([methodFactor, params.METHOD])


        # The Other session Parameters
        dateFactor = (datetime.datetime.now() - session.recordingDate).total_seconds()
        factors.append(dateFactor, params.SESSION_DATE)
        sensorFactor = session.fidelity
        factors.append(sensorFactor, params.SENSOR_TYPE)

        confidence = np.average(np.array(factors)[:,0], weights = np.array(factors)[:,1])

        return confidence
                
