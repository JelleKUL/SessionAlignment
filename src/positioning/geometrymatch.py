import math
import numpy as np

import open3d as o3d
from matplotlib import transforms

from geometrytransform import GeometryTransform
from match import Match


class GeometryMatch(Match):
    """This class stores all the data of a 3D match between 2 Geometries"""

    geometry1 : GeometryTransform = None    # the reference GeometryTransform
    geometry2 : GeometryTransform = None    # the query GeometryTransform
    voxelSize = 0.1
    transformation = []
    matchError = math.inf           # The match score of the image match (lower is better)
    result = None
    fidelity = 1                    # a measurement for the quality of the match
    matchType = "3d"


    def __init__(self, geometry1, geometry2, voxelSize = 0.1):

        self.geometry1 = geometry1
        self.geometry2 = geometry2
        self.voxelSize = voxelSize
        pass

    def find_matches(self):
        """Finds the matches between 2 pcds"""
        self.get_transformation()


    def get_transformation(self, method:str = "fgr"):
        """Computes the estimated transformation from-to the two geometries"""

        if(self.result is None):
            result = None
            if(method.lower() == "fgr"):
                result = execute_fast_global_registration(
                    self.geometry2.get_voxel_pcd(self.voxelSize), self.geometry1.get_voxel_pcd(self.voxelSize),
                    self.geometry2.get_features(), self.geometry1.get_features(),
                    self.voxelSize)
            self.transformation = result.transformation
            self.matchError = result.inlier_rmse
            self.result = result
        return result

    def get_translation_and_rotation(self):
        """Returns the translation and rotation matrices"""

        R = self.transformation[:3,:3]
        t = self.transformation[:3,3]

        return R,t

    def draw_matches(self, maxLines = -1):
        """Returns an open3d lineset object of all the matches"""
        
        lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(self.geometry2.get_voxel_pcd(), self.geometry1.get_voxel_pcd(), np.asarray(self.result.correspondence_set))
        if maxLines < 0: maxLines = len(np.asarray(lineset.lines))
        lineset.lines = o3d.cpu.pybind.utility.Vector2iVector(np.asarray(lineset.lines)[:(min(len(np.asarray(lineset.lines)), maxLines))])

        return lineset



def execute_global_registration(source_down, target_down, source_fpfh,target_fpfh, voxel_size):

    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result

def execute_fast_global_registration(source_pcd, target_pcd, source_fpfh,target_fpfh, radius):
    
    print(":: Apply fast global registration with distance threshold %.3f" \
            % radius)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=radius))

    return result
