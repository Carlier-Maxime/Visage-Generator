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
import click

import getLandmark2D
from FLAME import FLAME
from Viewer import Viewer
from renderer import Renderer
from config import Config

class VisageGenerator:
    def __init__(self, nb_face:int = 1, device: str = "cpu", min_shape_param: float = -2, max_shape_param: float = 2,
                 min_expression_param: float = -2, max_expression_param: float = 2,
                 global_pose_param_1: float = 45, global_pose_param_2: float = 45, global_pose_param_3: float = 90,
                 texturing: bool = True, save_lmks2D: bool = False, save_lmks3D: bool = False, save_png: bool = False,
                 save_obj: bool = True, lmk2d_format: str = "npy", flame_model_path=Config.flame_model_path, batch_size=Config.batch_size,
                 use_face_contour=Config.use_face_contour, use_3D_translation=Config.use_3D_translation, shape_params=Config.shape_params, expression_params=Config.expression_params, 
                 static_landmark_embedding_path=Config.static_landmark_embedding_path, dynamic_landmark_embedding_path=Config.dynamic_landmark_embedding_path,
                 optimize_eyeballpose=Config.optimize_eyeballpose, optimize_neckpose=Config.optimize_neckpose
        ):
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
        self._nbFace = nb_face
        radian = np.pi / 180.0

        flame_layer = FLAME(flame_model_path, nb_face, use_face_contour, use_3D_translation, shape_params, expression_params, static_landmark_embedding_path, dynamic_landmark_embedding_path)
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
        if optimize_eyeballpose and optimize_neckpose:
            neck_pose = torch.zeros(nb_face, 3).to(device)
            eye_pose = torch.zeros(nb_face, 6).to(device)
            vertex, landmark = flame_layer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

        texture = None
        if texturing:
            print('Texturing')
            tex_space = np.load("model/FLAME_texture.npz")
            texture_mean = tex_space['mean'].reshape(1, -1)
            texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            texture_mean = torch.from_numpy(texture_mean).float()[None, ...].to(device)
            texture_basis = torch.from_numpy(texture_basis[:, :50]).float()[None, ...].to(device)
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

@click.command()
@click.option('--nb-faces', type=int, default=Config.nb_faces, help='number faces generate')
@click.option('--lmk2D-format', type=str, default=Config.lmk2D_format, help='format used for save lmk2d. (npy and pts is supported)')
@click.option('--not-texturing', 'texturing', type=bool,  default=Config.texturing,  help='enable texture', is_flag=True)
@click.option('--save-obj',  type=bool,  default=Config.save_obj,  help='enable save into file obj', is_flag=True)
@click.option('--save-png',  type=bool,  default=Config.save_png,  help='enable save into file png', is_flag=True)
@click.option('--save-lmks3D', 'save_lmks3D', type=bool,  default=Config.save_lmks3D,  help='enable save landmarks 3D into file npy', is_flag=True)
@click.option('--save-lmks2D', 'save_lmks2D',  type=bool,  default=Config.save_lmks2D,  help='enable save landmarks 2D into file npy', is_flag=True)
@click.option('--min-shape_param',  type=float,  default=Config.min_shape_param,  help='minimum value for shape param')
@click.option('--max-shape_param',  type=float,  default=Config.max_shape_param,  help='maximum value for shape param')
@click.option('--min-expression-param',  type=float,  default=Config.min_expression_param,  help='minimum value for expression param')
@click.option('--max-expression-param',  type=float,  default=Config.min_expression_param,  help='maximum value for expression param')
@click.option('--global-pose-param1',  type=float,  default=Config.global_pose_param1,  help='value of first global pose param')
@click.option('--global-pose-param2',  type=float,  default=Config.global_pose_param2,  help='value of second global pose param')
@click.option('--global-pose-param3',  type=float,  default=Config.global_pose_param3,  help='value of third global pose param')
@click.option('--device',  type=str,  default=Config.device,  help='choice your device for generate face. ("cpu" or "cuda")')
@click.option('--view',  type=bool,  default=Config.view,  help='enable view', is_flag=True)
@click.option('--flame-model-path', type=str, default=Config.flame_model_path)
@click.option('--batch-size', type=int, default=Config.batch_size)
@click.option('--not-use-face-contour', 'use_face_contour', type=bool, default=Config.use_face_contour, is_flag=True)
@click.option('--not-use-3D-translation', 'use_3D_translation', type=bool, default=Config.use_3D_translation, is_flag=True)
@click.option('--shape-params', type=int, default=Config.shape_params)
@click.option('--expression-params', type=int, default=Config.expression_params)
@click.option('--static-landmark-embedding-path', type=str, default=Config.static_landmark_embedding_path)
@click.option('--dynamic-landmark-embedding-path', type=str, default=Config.dynamic_landmark_embedding_path)
@click.option('--not-optimize-eyeballpose', 'optimize_eyeballpose', type=bool, default=Config.optimize_eyeballpose, is_flag=True)
@click.option('--not-optimize-neckpose', 'optimize_neckpose', type=bool, default=Config.optimize_neckpose, is_flag=True)
def main(
    nb_faces,
    lmk2d_format,
    texturing,
    save_obj,
    save_png,
    save_lmks3D,
    save_lmks2D,
    min_shape_param,
    max_shape_param,
    min_expression_param,
    max_expression_param,
    global_pose_param1,
    global_pose_param2,
    global_pose_param3,
    device,
    view,
    flame_model_path,
    batch_size,
    use_face_contour,
    use_3D_translation,
    shape_params,
    expression_params,
    static_landmark_embedding_path,
    dynamic_landmark_embedding_path,
    optimize_eyeballpose,
    optimize_neckpose
):
    vg = VisageGenerator(nb_faces, device, min_shape_param, max_shape_param, min_expression_param, max_expression_param, global_pose_param1, global_pose_param2, global_pose_param3, texturing, save_lmks2D, save_lmks3D, save_png, save_obj, lmk2d_format, flame_model_path, batch_size, use_face_contour, use_3D_translation, shape_params, expression_params, static_landmark_embedding_path, dynamic_landmark_embedding_path, optimize_eyeballpose, optimize_neckpose)
    if view:
        vg.view()

if __name__ == "__main__":
    main()
