import pyrender
import numpy as np
import trimesh
from config import nbFace,device

class Viewer(pyrender.Viewer):
    def __init__(self, vertice, landmark, faces, show_joints=False, show_vertices=False):
        for i in range(nbFace):
            vertices = vertice[i].detach().to(device).numpy().squeeze()
            joints = landmark[i].detach().to(device).numpy().squeeze()
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.925, 0.72, 0.519, 1.0]

            tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene = pyrender.Scene()
            scene.add(mesh)

            if show_joints:
                # Joints (landmarks)
                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (len(joints), 1, 1))
                tfs[:, :3, 3] = joints
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)
            if show_vertices:
                # Vertices (points of topologie)
                sm = trimesh.creation.uv_sphere(radius=0.002)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
                tfs[:, :3, 3] = vertices
                vertices_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(vertices_pcl)
            
            pyrender.Viewer.__init__(self, scene, use_raymond_lighting=True)
