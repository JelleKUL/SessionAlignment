import open3d as o3d
import numpy as np


def get_3D_transformation():
    pass

def import_mesh(path : str):
    print("Testing IO for textured meshes ...")
    textured_mesh = o3d.io.read_triangle_mesh(path)
    print(textured_mesh)
    return textured_mesh


def cast_ray_in_mesh(mesh, startPoint : np.array, direction : np.array):
    "Casts a ray in a certain direction on a mesh, returns the distance to the hit"

    #create a scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh)
    direction = direction / np.linalg.norm(direction)
    rays = o3d.core.Tensor([startPoint[0], startPoint[1], startPoint[2], direction[0], direction[1], direction[2]],
                       dtype=o3d.core.Dtype.Float32)
    rayCast = scene.cast_rays(rays)

    return rayCast['t_hit']