"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Edited by M_EtOuais for Visage-Generator
"""

import os

import numpy as np
import torch

from FLAME import FLAME
from Viewer import Viewer
from config import get_config
from renderer import Renderer


class VisageGenerator:
    def __init__(self, min_shape_param=-2, max_shape_param=2,
                 min_expression_param=-2, max_expression_param=2,
                 global_pose_param_1=45, global_pose_param_2=45, global_pose_param_3=90,
                 main_launch=False
                 ) -> None:
        print("Load config")
        config = get_config()
        nb_face = config.number_faces
        self._nbFace = nb_face
        device = config.device
        if main_launch:
            min_shape_param = config.min_shape_param
            max_shape_param = config.max_shape_param
            min_expression_param = config.min_expression_param
            max_expression_param = config.max_expression_param
            global_pose_param_1 = config.global_pose_param_1
            global_pose_param_2 = config.global_pose_param_2
            global_pose_param_3 = config.global_pose_param_3
        radian = np.pi / 180.0

        flame_layer = FLAME(config)
        render = Renderer(512, "visage.obj", 512).to(device)

        print('Generate random parameters')
        shape_params = torch.tensor(np.random.uniform(min_shape_param, max_shape_param, [nb_face, 300]),
                                    dtype=torch.float32).to(device)
        pose_params_numpy = np.array(
            [[global_pose_param_1 * radian, global_pose_param_2 * radian, global_pose_param_3 * radian, 0.0, 0.0, 0.0]]
            * nb_face,
            dtype=np.float32)
        pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(device)
        expression_params = torch.tensor(np.random.uniform(min_expression_param, max_expression_param, [nb_face, 100]),
                                         dtype=torch.float32).to(device)
        texture_params = torch.tensor(np.random.uniform(-2, 2, [nb_face, 50])).float().to(device)

        print("Create Visage")
        flame_layer.to(device)
        vertex, landmark = flame_layer(shape_params, expression_params, pose_params)
        if config.optimize_eyeballpose and config.optimize_neckpose:
            neck_pose = torch.zeros(nb_face, 3).to(device)
            eye_pose = torch.zeros(nb_face, 6).to(device)
            vertex, landmark = flame_layer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

        print('Texturing')
        tex_space = np.load("model/FLAME_texture.npz")
        texture_mean = tex_space['mean'].reshape(1, -1)
        texture_basis = tex_space['tex_dir'].reshape(-1, 200)
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :50]).float()[None, ...]
        texture = texture_mean + (texture_basis * texture_params[:, None, :]).sum(-1)
        texture = texture.reshape(texture_params.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = texture[:, [2, 1, 0], :, :]
        texture = texture / 255

        print("Save")
        if not os.path.isdir('output'):
            os.mkdir('output')
        for i in range(nb_face):
            if config.save_obj:
                render.save_obj('output/visage' + str(i) + '.obj', vertex[i], texture[i])
            if config.save_lmks3D:
                np.save(f'output/visage{str(i)}.npy', landmark[i])
            if config.save_lmks2D:
                pass
            elif config.save_png:
                pass

        self._landmark = landmark
        self._vertex = vertex
        self._faces = flame_layer.faces

    def view(self, other_objects=None):
        file_obj = []
        for i in range(self._nbFace):
            file_obj.append('output/visage' + str(i) + '.obj')
        Viewer(self._vertex, self._landmark, self._faces, file_obj, other_objects=other_objects)

    def get_vertices(self, i):
        return self._vertex[i]

    def get_faces(self):
        return self._faces


if __name__ == "__main__":
    VisageGenerator(main_launch=True).view()
