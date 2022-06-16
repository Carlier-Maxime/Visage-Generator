"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Edited by M_EtOuais for Visage-Generator
"""

import numpy as np
import torch
from FLAME import FLAME
import pyrender
from config import get_config,nbFace,device
from Viewer import Viewer
import util

config = get_config()
radian = np.pi/180.0
flamelayer = FLAME(config)

# Creating a batch of mean shapes
shape_params = torch.tensor(np.random.uniform(0,0,[nbFace,300]), dtype=torch.float32).to(device)

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotation
# pose_params_numpy[:, 3:] : jaw rotation
pose_params_numpy = np.array([[70.0*radian, 70.0*radian, 70.0*radian, 0.0, 0.0, 0.0]]*nbFace, dtype=np.float32)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(device)

# Creating a batch of expressions
expression_params = torch.tensor(np.random.uniform(0,0,[nbFace,100]), dtype=torch.float32).to(device)
flamelayer.to(device)

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
vertice, landmark = flamelayer(shape_params, expression_params, pose_params) # For RingNet project

if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(nbFace, 3).to(device)
    eye_pose = torch.zeros(nbFace, 6).to(device)
    vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

t=[]
minX=0.05
minZ=-0.03
maxZ=0.15
for vertices in vertice:
    l = []
    for i in range(len(vertices)):
        inZone = vertices[i][0]>minX and vertices[i][2]>minZ and vertices[i][2]<maxZ
        eyeL = [0.108, -0.178, 0.085]
        eyeR = [0.108, -0.1155, 0.085]
        p = vertices[i]
        distEyeL = np.sqrt((p[0]-eyeL[0])**2+(p[1]-eyeL[1])**2+(p[2]-eyeL[2])**2)
        distEyeR = np.sqrt((p[0]-eyeR[0])**2+(p[1]-eyeR[1])**2+(p[2]-eyeR[2])**2)
        if inZone and distEyeL>0.0139 and distEyeR>0.0139:
            l.append(vertices[i])
    t.append(l)
#util.genDirectionnalMatrix(t)
#exit(0)

Viewer(vertice, landmark, flamelayer.faces, t)