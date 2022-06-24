"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Edited by M_EtOuais for Visage-Generator
"""

import numpy as np
import torch
from FLAME import FLAME
from config import get_config
from Viewer import Viewer
import util
from renderer import Renderer
import torch.nn.functional as F

class VisageGenerator():
    def __init__(self, minShapeParam=-2, maxShapeParam=2,
                minExpressionParam=-2, maxExpressionParam=2,
                globalPoseParam1=45, globalPoseParam2=45, globalPoseParam3=90,
                mainLaunch = False
                ) -> None:
        print("Load config")
        config = get_config()
        nbFace = config.number_faces
        device = config.device
        if mainLaunch:
            minShapeParam = config.min_shape_param
            maxShapeParam = config.max_shape_param
            minExpressionParam = config.min_expression_param
            maxExpressionParam = config.max_expression_param
            globalPoseParam1 = config.global_pose_param_1
            globalPoseParam2 = config.global_pose_param_2
            globalPoseParam3 = config.global_pose_param_3
        radian = np.pi/180.0

        flamelayer = FLAME(config)
        render = Renderer(512,"visage.obj",512).to(device)

        print('Generate random parameters')
        shape_params = torch.tensor(np.random.uniform(minShapeParam,maxShapeParam,[nbFace,300]), dtype=torch.float32).to(device)
        pose_params_numpy = np.array([[globalPoseParam1*radian, globalPoseParam2*radian, globalPoseParam3*radian, 0.0, 0.0, 0.0]]*nbFace, dtype=np.float32)
        pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(device)
        expression_params = torch.tensor(np.random.uniform(minExpressionParam,maxExpressionParam,[nbFace,100]), dtype=torch.float32).to(device)
        texture_params = torch.tensor(np.random.uniform(-2,2,[nbFace, 50])).float().to(device)

        print("Create Visage")
        flamelayer.to(device)
        vertice, landmark = flamelayer(shape_params, expression_params, pose_params)
        if config.optimize_eyeballpose and config.optimize_neckpose:
            neck_pose = torch.zeros(nbFace, 3).to(device)
            eye_pose = torch.zeros(nbFace, 6).to(device)
            vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

        print('Texturing')
        tex_space = np.load("model/FLAME_texture.npz")
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        texture_mean = torch.from_numpy(texture_mean).float()[None,...]
        texture_basis = torch.from_numpy(texture_basis[:,:50]).float()[None,...]
        texture = texture_mean + (texture_basis*texture_params[:,None,:]).sum(-1)
        texture = texture.reshape(texture_params.shape[0], 512, 512, 3).permute(0,3,1,2)
        texture = texture[:,[2,1,0], :,:]
        albedos = texture / 255

        print("Save")
        for i in range(nbFace):
            render.save_obj('output/visage'+str(i)+'.obj',vertice[i],albedos[i])

    def view(self,otherObjetcs=None):
        print("View is temporary disable. (wait update)")
        #Viewer(self._vertice, self._landmark, self._faces, otherObjects=otherObjetcs)
        
    def getVertices(self,i):
        return self._vertice[i]

    def getFaces(self):
        return self._faces


if __name__ == "__main__":
    VisageGenerator(mainLaunch=True).view()