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
import sys

class VisageGenerator():
    def __init__(self, minShapeParam=-2, maxShapeParam=2,
                minExpressionParam=-2, maxExpressionParam=2,
                globalPoseParam1=45, globalPoseParam2=45, globalPoseParam3=90,
                mainLaunch = False
                ) -> None:
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

        shape_params = torch.tensor(np.random.uniform(minShapeParam,maxShapeParam,[nbFace,300]), dtype=torch.float32).to(device)
        pose_params_numpy = np.array([[globalPoseParam1*radian, globalPoseParam2*radian, globalPoseParam3*radian, 0.0, 0.0, 0.0]]*nbFace, dtype=np.float32)
        pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(device)
        expression_params = torch.tensor(np.random.uniform(minExpressionParam,maxExpressionParam,[nbFace,100]), dtype=torch.float32).to(device)

        flamelayer.to(device)
        vertice, landmark = flamelayer(shape_params, expression_params, pose_params)

        if config.optimize_eyeballpose and config.optimize_neckpose:
            neck_pose = torch.zeros(nbFace, 3).to(device)
            eye_pose = torch.zeros(nbFace, 6).to(device)
            vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

        self._vertice = vertice
        self._landmark = landmark
        self._faces = flamelayer.faces

    def view(self,otherObjetcs=None):
        Viewer(self._vertice, self._landmark, self._faces, otherObjects=otherObjetcs)

    def save(self):
        util.saveFaces(self._vertice)
        


if __name__ == "__main__":
    VisageGenerator(mainLaunch=True).view()