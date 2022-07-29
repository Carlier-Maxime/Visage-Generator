"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Edited by M_EtOuais for Visage-Generator
"""

import os

import numpy as np
import torch
import trimesh

import getLandmark2D
from FLAME import FLAME
from Viewer import Viewer
from config import get_config
from renderer import Renderer


class VisageGenerator:
    def __init__(self, nb_face: int = 1, device: str = "cpu", min_shape_param: float = -2, max_shape_param: float = 2,
                 min_expression_param: float = -2, max_expression_param: float = 2,
                 global_pose_param_1: float = 45, global_pose_param_2: float = 45, global_pose_param_3: float = 90,
                 texturing: bool = True, save_lmks2D: bool = False, save_lmks3D: bool = False, save_png: bool = False,
                 save_obj: bool = True, lmk2d_format: str = "npy") -> None:
        """
        Args:
            nb_face (int): number of faces to generate
            device (str): device used ('cpu' or 'cuda')
            min_shape_param (float): minimum value for shape param
            max_shape_param (float): maximum value for shape param
            min_expression_param (float): minimum value for expression param
            max_expression_param (float): maximum value for expression param
            global_pose_param_1 (float): value of first global pose param
            global_pose_param_2 (float): value of second global pose param
            global_pose_param_3 (float): value of third global pose param
            texturing (bool): enable texturing
            save_lmks2D (bool): save 2D landmarks
            save_lmks3D (bool): save 3D landmarks
            save_png (bool): save an image of the 3D face preview
            save_obj (bool): save 3D object to the file
            lmk2d_format (str): format used for save 2D landmark
        """
        print("Load config")
        config = get_config()
        self._nbFace = nb_face
        radian = np.pi / 180.0

        flame_layer = FLAME(config)
        self.render = Renderer(512, "visage.obj", 512).to(device)

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

        texture = None
        if texturing:
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
        for folder in ['output', 'tmp']:
            if not os.path.isdir(folder):
                os.mkdir(folder)
        lmks_paths = ""
        visage_paths = ""
        save_paths = ""
        for i in range(nb_face):
            if save_obj:
                tex = None if texture is None else texture[i]
                self.save_obj(f'output/visage{str(i)}.obj', vertex[i], flame_layer.faces, tex)
            if save_lmks3D:
                np.save(f'output/visage{str(i)}.npy', landmark[i])
            if save_lmks2D:
                lmks_path = f'output/visage{str(i)}.npy'
                if not save_lmks3D:
                    np.save(f'tmp/visage{str(i)}.npy', landmark[i])
                    lmks_path = f'tmp/visage{str(i)}.npy'
                visage_path = f'output/visage{str(i)}.obj'
                if not save_obj:
                    visage_path = f'tmp/visage{str(i)}.obj'
                    tex = None if texture is None else texture[i]
                    self.save_obj(visage_path, vertex[i], flame_layer.faces, tex)
                if i != 0:
                    lmks_paths += ";"
                    visage_paths += ";"
                    save_paths += ";"
                lmks_paths += lmks_path
                visage_paths += visage_path
                save_paths += f'output/visage{str(i)}_lmks2d.{lmk2d_format}'
            elif save_png:
                pass
        if save_lmks2D:
            getLandmark2D.run(visage_paths, lmks_paths, save_paths, save_png)

        self._landmark = landmark
        self._vertex = vertex
        self._faces = flame_layer.faces

    def save_obj(self, path: str, vertices: list, faces: list, texture=None) -> None:
        """
        Save 3D object to the obj format
        Args:
            path (str): save path
            vertices (list): array of all vertex
            faces (list): array of all face
            texture: texture

        Returns: None

        """
        if texture is not None:
            self.render.save_obj(path, vertices, texture)
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            normals = mesh.vertex_normals  # generate normals
            with open(path, 'w') as f:
                f.write(trimesh.exchange.obj.export_obj(mesh, True, False, False))

    def view(self, other_objects=None) -> None:
        """
        View visage generate
        Args:
            other_objects: other object to display with faces

        Returns: None
        """
        file_obj = []
        for i in range(self._nbFace):
            file_obj.append('output/visage' + str(i) + '.obj')
        Viewer(self._vertex, self._landmark, self._faces, file_obj, other_objects=other_objects)

    def get_vertices(self, i: int) -> list:
        """
        Obtain vertices for face
        Args:
            i: index of face

        Returns: vertices
        """
        return self._vertex[i]

    def get_faces(self) -> list:
        """
        Obtain faces
        Returns: array of all faces
        """
        return self._faces


if __name__ == "__main__":
    cfg = get_config()
    VisageGenerator(cfg.number_faces, cfg.device, cfg.min_shape_param, cfg.max_shape_param, cfg.min_expression_param,
                    cfg.max_expression_param, cfg.global_pose_param_1, cfg.global_pose_param_2, cfg.global_pose_param_3,
                    cfg.texturing, cfg.save_lmks2D, cfg.save_lmks3D, cfg.save_png, cfg.save_obj, cfg.lmk2d_format)
