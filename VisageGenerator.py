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

class VisageGenerator():
    def __init__(self, minShapeParam=-2, maxShapeParam=2,
                minExpressionParam=-2, maxExpressionParam=2,
                globalPoseParam1=45, globalPoseParam2=45, globalPoseParam3=90
                ) -> None:
        config = get_config()
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

    def view(self):
        t,m = util.getVerticeBalises(self._vertice)
        Viewer(self._vertice, self._landmark, self._faces, t)

    def save(self):
        util.saveFaces(self._vertice)


if __name__ == "__main__":
    VisageGenerator(0,0,0,0,70,70,70).view()