"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face 

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import numpy as np
import torch
from FLAME import FLAME
import pyrender
import trimesh
from config import get_config,nbFace

config = get_config()
radian = np.pi/180.0
flamelayer = FLAME(config)

# Creating a batch of mean shapes
shape_params = torch.tensor(np.random.uniform(-2,2,[nbFace,300]), dtype=torch.float32).cpu()

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotaation
# pose_params_numpy[:, 3:] : jaw rotaation
pose_params_numpy = np.array([[45.0*radian, 45.0*radian, 90.0*radian, 0.0, 0.0, 0.0]*nbFace], dtype=np.float32)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cpu()

# Cerating a batch of neutral expressions
expression_params = torch.tensor(np.random.uniform(-2,2,[nbFace,100]), dtype=torch.float32).cpu()
flamelayer.cpu()

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
vertice, landmark = flamelayer(shape_params, expression_params, pose_params) # For RingNet project
print(vertice.size(), landmark.size())

if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(nbFace, 3).cpu()
    eye_pose = torch.zeros(nbFace, 6).cpu()
    vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

# Visualize Landmarks
# This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
faces = flamelayer.faces
for i in range(nbFace):
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints = landmark[i].detach().cpu().numpy().squeeze()
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.925, 0.72, 0.519, 1.0]

    tri_mesh = trimesh.Trimesh(vertices, faces,
                                vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    """# Joints (landmarks)
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)
    """

    pyrender.Viewer(scene, use_raymond_lighting=True)