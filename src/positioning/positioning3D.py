import math
from numpy import linalg
import open3d as o3d
import numpy as np
import quaternion
import matplotlib

from session import Session



def get_3D_transformation(testSession : Session, refSessions : "list[Session]", resolution = 0.05):
    "Calculate the estimated transformation between 2 point clouds with a given voxelSise"
    sessionTransformations = []

    for refSession in refSessions:
        sessionTransformations.append(compare_session(testSession, refSession, resolution))

    return sessionTransformations

    

def compare_session(testSession, refSession, resolution = 0.05):
    "compare 2 session against each other and returs the estimated transformation matrix"
    nrCheck = 0
    totalCheck = len(testSession.geometries) * len(refSession.geometries)
    transformations = []

    print("Starting Comparing:", len(testSession.geometries), "Against", len(refSession.geometries), "Geometries")

    for testGeometry in testSession.geometries:
        testPcd = testGeometry
        if isinstance(testGeometry, o3d.geometry.TriangleMesh):
            #the geometry is a mesh
            testPcd = to_pcd(testGeometry, 100000,1)
        for refGeometry in refSession.geometries:
            refPcd = refGeometry
            if isinstance(refGeometry, o3d.geometry.TriangleMesh):
                #the geometry is a mesh
                refPcd = to_pcd(refGeometry, 100000,1)
            transformations.append(get_pcd_transformation(testPcd, refPcd, resolution))
    
    return transformations

#### Triangle Mesh ####

def import_mesh(path : str):
    "Imports a mesh from a specific path"

    print("Importing mesh from:", path)
    mesh = o3d.io.read_triangle_mesh(path)
    #if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()
    print("Importing complete:",mesh)
    return mesh

def to_pcd(mesh : o3d.geometry, nrOfPoints : int, factor : int = 2):
    "Converts a triangle mesh to a point cloud with a given amount of points"
    print("converting mesh to PCD with", nrOfPoints, "points")
    pcd = mesh.sample_points_poisson_disk(nrOfPoints, factor)
    print("Converting complete", pcd)
    return pcd

def cast_ray_in_mesh(mesh, startPoint : np.array, direction : np.array):
    "Casts a ray in a certain direction on a mesh, returns the distance to the hit"

    #create a scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    direction = direction / np.linalg.norm(direction)
    rays = o3d.core.Tensor([[startPoint[0], startPoint[1], startPoint[2], direction[0], direction[1], direction[2]]],
                       dtype=o3d.core.Dtype.Float32)
    rayCast = scene.cast_rays(rays)
    distance = float(o3d.core.Tensor.numpy(rayCast['t_hit']))
    print("raycast distance:" , distance)
    return distance

def cast_ray_from_camera(mesh, startPoint : np.array, rotation : np.array):
    "Casts a ray in a certain direction on a mesh, returns the distance to the hit"

    #create a scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    direction = quaternion.as_rotation_matrix(rotation) @ np.array([0,0,1]).T
    direction = direction.T
    rays = o3d.core.Tensor([[startPoint[0], startPoint[1], startPoint[2], direction[0], direction[1], direction[2]]],
                       dtype=o3d.core.Dtype.Float32)
    rayCast = scene.cast_rays(rays)
    distance = float(o3d.core.Tensor.numpy(rayCast['t_hit']))
    print("raycast distance:" , distance)
    return distance, startPoint + direction * distance


#### Point Cloud ####

def import_point_cloud(path : str):
    "Imports a point cloud from a path"
    print("Importing point cloud from:", path)
    pcd = o3d.io.read_point_cloud(path)
    print("Importing complete:",pcd)
    return pcd

def save_pcd(pcd, path : str):
    o3d.io.write_point_cloud(path, pcd)

def get_fpfh_features(pcd, radius):

    print(":: Compute FPFH feature with search radius %.3f." % radius)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    return pcd_fpfh

def execute_fast_global_registration(source_pcd, target_pcd, source_fpfh,target_fpfh, radius):
    
    print(":: Apply fast global registration with distance threshold %.3f" \
            % radius)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=radius))
    return result

def get_pcd_transformation(pcdTest : o3d.geometry, pcd2Ref : o3d.geometry, voxelSize : float):
    "Calculate the estimated transformation between 2 point clouds with a given voxelSise"

    voxel_pcdTest = pcdTest.voxel_down_sample(voxelSize)
    voxel_pcdRef = pcd2Ref.voxel_down_sample(voxelSize)

    voxel_pcdTest.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxelSize*2, max_nn=30))
    voxel_pcdRef.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxelSize*2, max_nn=30))

    fpfh_pcdTest = get_fpfh_features(voxel_pcdTest, voxelSize * 5)
    fpfh_pcdRef = get_fpfh_features(voxel_pcdRef, voxelSize * 5)

    result_fast = execute_fast_global_registration(voxel_pcdTest, voxel_pcdRef,fpfh_pcdTest, fpfh_pcdRef,voxelSize * 5)
    return result_fast.transformation


#### Generic 3D ####

def get_bounding_box(geometry : o3d.geometry):
    "returns the bounding box "
    return geometry.get_axis_aligned_bounding_box()


def get_bounding_radius(geometry : o3d.geometry):
    "returns the radius from the center of the mesh of the bounding sphere"

    # get the bounding box
    box = geometry.get_axis_aligned_bounding_box()
    # find the furthest point from the origin
    points = box.get_box_points()
    maxDist = 0
    for point in points:
        maxDist = max(maxDist, np.linalg.norm(point))
    return maxDist

def create_3d_camera(pos = [0,0,0], rotation = [1,0,0,0], scale = 1.0):
    "Returns a geometry lineset object that represents a camera in 3D space"
    box = o3d.geometry.TriangleMesh.create_box(1.6,0.9, 0.1)
    box.translate((-0.8, -0.45, -0.05))
    box.scale(scale, center=(0, 0, 0))
    box.rotate(box.get_rotation_matrix_from_quaternion(rotation))
    box.translate(pos)
    return box

def convert_to_open3d(pos : np.array, rot : np.array):
    "converts the stored coordinates in (right, up, forward) to the open3D standard (left, up, forward)"
    newPos = np.multiply(pos ,np.array([-1,1,1]))
    newRot = np.multiply(rot ,np.array([-1,-1,1,1]))

def show_geometries(geometries, color = False):
    "displays the array of meshes in a 3D view"

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    viewer.add_geometry(frame)
    for i, geometry in enumerate(geometries):
        if color:
            geometry.paint_uniform_color(matplotlib.colors.hsv_to_rgb([float(i)/len(geometries),0.8,0.8]))
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    opt.light_on = True
    viewer.run()


#### Helper functions


def draw_registration_result(source, target, transformation):
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    source.transform(transformation)
    o3d.visualization.draw_geometries([source, target],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])



def test():

    mesh = import_mesh("/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-11 Testbuilding Campus/RAW Data/Hololens/session-2021-11-25 16-09-47/mesh-2021-11-25 16-16-01.obj")
    mesh.compute_vertex_normals()

    #print("max distance:",get_bounding_radius(mesh))
    #mesh_smp = mesh.simplify_vertex_clustering(
    #    voxel_size=max(mesh.get_max_bound() - mesh.get_min_bound()) / 32,
    #    contraction=o3d.geometry.SimplificationContraction.Average)

    #startPoint = np.array([1,0,1])
    #direction = np.array([-1,-0.5,1])

    #distance = cast_ray_in_mesh(mesh, startPoint, direction)

    #if(distance != math.inf):
    #   print("The distance was not infinity:", distance)
    #   raycastLine = o3d.geometry.LineSet(
    #       points=o3d.utility.Vector3dVector([startPoint, startPoint + distance * (direction/ np.linalg.norm(np.array([-1,-0.5,1])))]),
    #       lines=o3d.utility.Vector2iVector([[0,1]]),
    #       )

    camPos = np.array([6.5,-0.2,1.4])
    camRot = np.quaternion(-0.96,-0.13,0.24,-0.02)

    distance, target = cast_ray_from_camera(mesh, camPos, camRot)

    if(distance != math.inf):
        print("The distance was not infinity:", distance)
        raycastLine = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([camPos, target]),
        lines=o3d.utility.Vector2iVector([[0,1]]),
        )
        show_geometries([mesh,raycastLine])

    #show_geometries([mesh,create_3d_camera([9.71,-0.07,1.5], [-0.987,-0.098,-0.1173,0.024]),create_3d_camera([6.5,-0.2,1.4], [-0.96,-0.13,0.24,-0.02])])
        #show_geometries([mesh,raycastLine])
    #viewer.destroy_window()

def test3D():
    
#    mesh1 = import_mesh("/Volumes/Data drive/Documents/Doctoraat Local/PythonDataAlignment/positioning/images/ref/mesh-2021-11-25 16-16-01.obj")
#    mesh2 = import_mesh("/Volumes/Data drive/Documents/Doctoraat Local/PythonDataAlignment/positioning/images/ref/mesh-2021-11-25 16-17-19.obj")
#
#    nrOfPoints = 100000
#    pcd1 = to_pcd(mesh1, nrOfPoints)
#    pcd2 = to_pcd(mesh2, nrOfPoints)
#
#    save_pcd(pcd1, "/Volumes/Data drive/Documents/Doctoraat Local/PythonDataAlignment/positioning/images/ref/pcd1.pcd")
#    save_pcd(pcd2, "/Volumes/Data drive/Documents/Doctoraat Local/PythonDataAlignment/positioning/images/ref/pcd2.pcd")

    pcd1 = o3d.io.read_point_cloud("/Volumes/Data drive/Documents/Doctoraat Local/PythonDataAlignment/src/positioning/images/ref/pcd1.pcd")
    pcd2 = o3d.io.read_point_cloud("/Volumes/Data drive/Documents/Doctoraat Local/PythonDataAlignment/src/positioning/images/ref/pcd2.pcd")

    print("Downsampling Pointclouds")
    voxelSize = 0.1
    voxel_pcd1 = pcd1.voxel_down_sample(voxelSize)
    voxel_pcd2 = pcd2.voxel_down_sample(voxelSize)

    fpfh_pcd1 = get_fpfh_features(voxel_pcd1, voxelSize * 5)
    fpfh_pcd2 = get_fpfh_features(voxel_pcd2, voxelSize * 5)

    result_fast = execute_fast_global_registration(voxel_pcd2, voxel_pcd1,fpfh_pcd2, fpfh_pcd1,voxelSize * 5)
    #draw_registration_result(voxel_pcd1, voxel_pcd2, result_fast.transformation)
    moved_pcd = voxel_pcd2.transform(result_fast.transformation)
    

    #visualise
    voxel_pcd1.paint_uniform_color([1, 0.706, 0])
    voxel_pcd2.paint_uniform_color([0.706,1, 0])
    #visualise
#    pcd1.paint_uniform_color([0.5, 1, 0.5])
#    pcd2.paint_uniform_color([1,0.5, 0.5])

    show_geometries([voxel_pcd1, voxel_pcd2, moved_pcd])

test3D()



    


