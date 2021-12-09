import math
from numpy import linalg
import open3d as o3d
import numpy as np
import quaternion



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