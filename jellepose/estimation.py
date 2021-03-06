import datetime

import numpy as np
import quaternion

import jellepose.params as params
from jellepose.geometrymatch import GeometryMatch
from jellepose.match import Match


class PoseEstimation():
    """Contains an estimated pose and all it's parameters to calculate it's validity"""

    position = [0,0,0]
    rotation = [0,0,0,1]

    matches: Match = []
    method = ""
    referenceSessionId = ""


    def __init__(self, position, rotation, matches, method) -> None:
        self.position = np.reshape(position, (3,1))
        self.rotation = rotation
        self.matches = matches
        self.method = method
        

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
        factors.append([dateFactor, params.SESSION_DATE])
        sensorFactor = session.fidelity
        factors.append([sensorFactor, params.SENSOR_TYPE])

        confidence = np.average(np.array(factors)[:,0], weights = np.array(factors)[:,1])

        return confidence

    def to_table_string(self, nr, session):
        """Returns the Paper formatted string for a table"""

        averageError = 0
        averageMatches = 0
        for match in self.matches: #type: Match
            averageError += match.matchError
            averageMatches += match.matchAmount
        averageError /= len(self.matches)
        averageMatches /= len(self.matches)

        value = "Image "
        value += str(nr)
        value += " & "
        value += str(np.around(self.position.T, decimals=2))
        value += " & "
        value += str(np.around(quaternion.as_float_array(quaternion.from_rotation_matrix(self.rotation)), decimals=2))
        value += " & "
        value += str(np.around(averageError, decimals=2))
        value += " & "
        value += str(np.around(averageMatches, decimals=2))
        value += " & 100 & "
        value += str(np.around(self.get_confidence(session), decimals=2))
        value += " \\\\"

        return value

    def to_string_array(self, session):
        """Converts all the paramerters and data to a string array"""
        decimals = 3
        array = []

        averageError = 0
        averageMatches = 0
        averageFitness = 0
        for match in self.matches: #type: Match
            averageError += match.matchError
            averageMatches += match.matchAmount
            averageFitness += match.get_fitness()
        averageError /= len(self.matches)
        averageMatches /= len(self.matches)
        averageFitness /= len(self.matches)

        if(isinstance(self.matches[0], GeometryMatch)):
            array.append(self.matches[0].geometry2.id) #test Image
            array.append(self.matches[0].geometry1.id) #ref image
        else:
            array.append(self.matches[0].image2.id) #test Image
            array.append(self.matches[0].image1.id) #ref image
        array.append(self.method) #ref image
        array.append(self.referenceSessionId) # the reference session Id
        # position
        position = np.around(self.position, decimals=decimals)
        array.append(position[0][0])
        array.append(position[1][0])
        array.append(position[2][0])

        # rotation
        rotation = np.around(quaternion.as_float_array(quaternion.from_rotation_matrix(self.rotation)), decimals=decimals)
        array.append(rotation[0])
        array.append(rotation[1])
        array.append(rotation[2])
        array.append(rotation[3])

        #params
        array.append(np.around(averageError, decimals=decimals))
        array.append(np.around( averageMatches, decimals=decimals))
        array.append(np.around( averageFitness, decimals=decimals))
        array.append(np.around(self.get_confidence(session), decimals=decimals))

        return array



