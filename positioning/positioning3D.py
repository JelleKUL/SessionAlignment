import math
from numpy import linalg
import open3d as o3d
import numpy as np


def get_3D_transformation():
    pass

#### Triangle Mesh ####

def import_mesh(path : str):
    "Imports a mesh from a specific path"

    print("Importing mesh from:", path)
    mesh = o3d.io.read_triangle_mesh(path)
    print("Importing complete:",mesh)
    return mesh


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

#### Point Cloud ####

def import_point_cloud(path : str):
    "Imports a point cloud from a path"
    print("Importing point cloud from:", path)
    pcd = o3d.io.read_point_cloud(path)
    print("Importing complete:",pcd)
    return pcd

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

def show_geometries(geometries):
    "displays the array of meshes in a 3D view"

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    viewer.add_geometry(frame)
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()



mesh = import_mesh("/Volumes/GeomaticsProjects1/Projects/2025-03 Project FWO SB Jelle/7.Data/21-11 Testbuilding Campus/RAW Data/Hololens/session-2021-11-25 16-09-47/mesh-2021-11-25 16-16-01.obj")
mesh.compute_vertex_normals()

print("max distance:",get_bounding_radius(mesh))
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

#show_geometries([ mesh,mesh.get_axis_aligned_bounding_box(),mesh.get_oriented_bounding_box()])
    #show_geometries([mesh,raycastLine])
#viewer.destroy_window()