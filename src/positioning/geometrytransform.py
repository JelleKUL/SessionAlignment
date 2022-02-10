import open3d as o3d
import numpy as np
import math

import params as params

from rdfobject import RdfObject

class GeometryTransform(RdfObject):
    """Stores all the data about a geometry"""

    id = ""                     # the id/name of the image without it's extension
    path = ""                   # the full path of the image
    features = None             # the cv2 generated keypoints
    geometry = None             # the open3d.geometry
    accuracy = []
    voxelPcd = None
    voxelSize = 1
    


    def __init__(self, id = None, path= None):
        """the input path is the location of the folder"""
        
        self.id = id
        self.path = path

    def from_dict(self, dict, path, type):
        """the input path is the location of the folder, type = 'mesh' or 'pcd' """

        #self.id = dict["id"]
        self.id = dict
        self.path = path
        self.get_geometry()
        return self

    def from_path(self, path):
        """the input path is the location of the folder, type = 'mesh' or 'pcd' """

        #self.id = dict["id"]
        self.id = path
        self.path = path
        self.get_geometry()
        return self

    def get_geometry(self):
        """Returns the open3d geometry object"""

        if(self.geometry is None):
            if(self.path.endswith(tuple(params.MESH_EXTENSION))):
                newGeometry = o3d.io.read_triangle_mesh(self.path)
                #if not newGeometry.has_vertex_normals():
                newGeometry.compute_vertex_normals()
            if(self.path.endswith(tuple(params.PCD_EXTENSION))):
                newGeometry = o3d.io.read_point_cloud(self.path)
            self.geometry = newGeometry
        return self.geometry

    def get_bounding_box(self):
        "returns the bounding box "
        return self.geometry.get_axis_aligned_bounding_box()

    def get_voxel_pcd(self, voxelSize: float = -1):
        """Returns the geometry as a point cloud from source or sampled from mesh"""

        if(self.voxelPcd is None):
            if(voxelSize != -1): self.voxelSize = voxelSize

            if (isinstance(self.geometry, o3d.geometry.TriangleMesh)):
                voxelMesh = self.geometry.simplify_vertex_clustering(self.voxelSize)
                voxelPcd = o3d.geometry.PointCloud()
                voxelPcd.points = o3d.utility.Vector3dVector(voxelMesh.vertices)
                self.voxelPcd = voxelPcd
            elif (isinstance(self.geometry, o3d.geometry.PointCloud)):
                self.voxelPcd = self.geometry.voxel_down_sample(self.voxelSize)

            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize*2, max_nn=30)
            self.voxelPcd.estimate_normals(search_param)

        return self.voxelPcd
    
    def get_features(self, voxelSize: float = -1, type: str = "fpfh"):
        """Returns the calculated or stored features of the geometry"""

        if(voxelSize != -1): self.voxelSize = voxelSize

        if(self.features is None):
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize*2, max_nn=30)
            self.voxelPcd.estimate_normals(search_param)
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxelSize * 5, max_nn=100)
            self.features = o3d.pipelines.registration.compute_fpfh_feature(self.voxelPcd,search_param)

        return self.features

    def get_distance_from_point(self, origin, direction):
        """Returns the distance from a raycast from a point to the geometry"""

        distance = math.inf
        #Check the type of the geometry
        if (isinstance(self.geometry, o3d.geometry.TriangleMesh)):
            distance = cast_ray_in_mesh(self.get_geometry(), origin, direction)
        elif (isinstance(self.geometry, o3d.geometry.PointCloud)):
            distance = voxel_traversal(self.get_geometry(), origin, direction, self.voxelSize)

        return distance



def cast_ray_in_mesh(mesh, origin : np.array, direction : np.array):
    "Casts a ray in a certain direction on a mesh, returns the distance to the hit"

    #create a scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    direction = direction / np.linalg.norm(direction)
    rays = o3d.core.Tensor([[origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]]],
                       dtype=o3d.core.Dtype.Float32)
    rayCast = scene.cast_rays(rays)
    distance = float(o3d.core.Tensor.numpy(rayCast['t_hit']))
    print("raycast distance:" , distance)
    return distance

#TODO add range check based on bounding box distance from origin
def voxel_traversal(pcd, origin, direction, voxelSize, maxRange = 100): 
    """Cast a ray in a voxel array created from the pcd"""

    pcd_Voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxelSize)
    #the starting index of the voxels
    voxel_origin = pcd.get_voxel(origin)
    iX = math.floor(origin[0] * voxelSize) / voxelSize
    iY = math.floor(origin[1] * voxelSize) / voxelSize
    iZ = math.floor(origin[2] * voxelSize) / voxelSize

    stepX = np.sign(direction[0])
    stepY = np.sign(direction[1])
    stepZ = np.sign(direction[2])

    tDeltaX = 1/direction[0] * voxelSize
    tDeltaY = 1/direction[1] * voxelSize
    tDeltaZ = 1/direction[2] * voxelSize

    tMaxX = origin[0]
    tMaxY = origin[1]
    tMaxZ = origin[2]

    for i in range(0,maxRange):
        # check if the current point is in a occupied voxel
        if(pcd_Voxel.check_if_included(o3d.utility.Vector3dVector([[iX * voxelSize, iY * voxelSize, iY * voxelSize]]))[0]):
            distance = np.linalg.norm(np.array([iX * voxelSize, iY * voxelSize, iY * voxelSize]) - origin)
            return True, distance

        if(tMaxX < tMaxY):
            if(tMaxX < tMaxZ):
                #traverse in the X direction
                tMaxX += tDeltaX
                iX += stepX
            else:
                #traverse in the Z direction
                tMaxZ += tDeltaZ
                iZ += stepZ
        else:
            if(tMaxY < tMaxZ):
                #traverse in the Y direction
                tMaxY += tDeltaY
                iY += stepY
            else:
                #traverse in the Z direction
                tMaxZ += tDeltaZ
                iZ += stepZ

    return False, math.inf