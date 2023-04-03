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

class VisageGenerator():
    def __init__(self, device:str = Config.device, min_shape_param:float = Config.min_shape_param, max_shape_param:float = Config.max_shape_param,
                 min_expression_param:float = Config.min_expression_param, max_expression_param:float = Config.max_expression_param,
                 global_pose_param1:float = Config.global_pose_param1, global_pose_param2:float = Config.global_pose_param2, global_pose_param3:float = Config.global_pose_param3,
                 flame_model_path=Config.flame_model_path, batch_size=Config.batch_size, use_face_contour=Config.use_face_contour,
                 use_3D_translation=Config.use_3D_translation, shape_params=Config.shape_params, expression_params=Config.expression_params, 
                 static_landmark_embedding_path=Config.static_landmark_embedding_path, dynamic_landmark_embedding_path=Config.dynamic_landmark_embedding_path
        ):
        self.flame_layer = FLAME(flame_model_path, batch_size, use_face_contour, use_3D_translation, shape_params, expression_params, static_landmark_embedding_path, dynamic_landmark_embedding_path).to(device)
        self.render = Renderer(512, "visage.obj", 512).to(device)
        self.device = device
        self.min_shape_param = min_shape_param
        self.max_shape_param = max_shape_param
        self.min_expression_param = min_expression_param
        self.max_expression_param = max_expression_param
        self.global_pose_param1 = global_pose_param1
        self.global_pose_param2 = global_pose_param2
        self.global_pose_param3 = global_pose_param3
        self.batch_size = batch_size

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

    def generate(self, nb_faces:int = Config.nb_faces, texturing:bool = Config.texturing, optimize_eyeballpose=Config.optimize_eyeballpose, optimize_neckpose=Config.optimize_neckpose):
        print('Generate random parameters')
        radian = np.pi / 180.0
        shape_params = torch.tensor(np.random.uniform(self.min_shape_param, self.max_shape_param, [nb_faces, 300]),dtype=torch.float32).to(self.device)
        pose_params_numpy = np.array(
            [[self.global_pose_param1 * radian, self.global_pose_param2 * radian, self.global_pose_param3 * radian, 0.0, 0.0, 0.0]] * nb_faces,
            dtype=np.float32)
        pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(self.device)
        expression_params = torch.tensor(np.random.uniform(self.min_expression_param, self.max_expression_param, [nb_faces, 100]),dtype=torch.float32).to(self.device)
        texture_params = torch.tensor(np.random.uniform(-2, 2, [nb_faces, 50])).float().to(self.device)

        print("Create Visage")
        if optimize_eyeballpose and optimize_neckpose:
            neck_pose = torch.zeros(nb_faces, 3).to(self.device)
            eye_pose = torch.zeros(nb_faces, 6).to(self.device)
            vertex, landmark = self.flame_layer(shape_params, expression_params, pose_params, neck_pose, eye_pose)
        else: vertex, landmark = self.flame_layer(shape_params, expression_params, pose_params)

        texture = None
        if texturing:
            print('Texturing')
            tex_space = np.load("model/FLAME_texture.npz")
            texture_mean = tex_space['mean'].reshape(1, -1)
            texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            texture_mean = torch.from_numpy(texture_mean).float()[None, ...].to(self.device)
            texture_basis = torch.from_numpy(texture_basis[:, :50]).float()[None, ...].to(self.device)
            texture = texture_mean + (texture_basis * texture_params[:, None, :]).sum(-1)
            texture = texture.reshape(texture_params.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
            texture = texture[:, [2, 1, 0], :, :]
            texture = texture / 255

        self._landmark = landmark
        self._vertex = vertex
        self._faces = self.flame_layer.faces
        self._texture = texture

    def save(self, save_obj:bool = Config.save_obj, save_png:bool = Config.save_png, save_lmks2D:bool = Config.save_lmks2D, save_lmks3D:bool=Config.save_lmks3D, lmk2D_format:str=Config.lmk2D_format):
        print("Save")
        for folder in ['output', 'tmp']:
            if not os.path.isdir(folder):
                os.mkdir(folder)
        lmks_paths = ""
        visage_paths = ""
        save_paths = ""
        for i in range(len(self._vertex)):
            if save_obj:
                tex = None if self._texture is None else self._texture[i]
                self.save_obj(f'output/visage{str(i)}.obj', self._vertex[i], self._faces, tex)
            if save_lmks3D:
                np.save(f'output/visage{str(i)}.npy', self._landmark[i])
            if save_lmks2D:
                lmks_path = f'output/visage{str(i)}.npy'
                if not save_lmks3D:
                    np.save(f'tmp/visage{str(i)}.npy', self._landmark[i])
                    lmks_path = f'tmp/visage{str(i)}.npy'
                visage_path = f'output/visage{str(i)}.obj'
                if not save_obj:
                    visage_path = f'tmp/visage{str(i)}.obj'
                    tex = None if self._texture is None else self._texture[i]
                    self.save_obj(visage_path, self._vertex[i], self._faces, tex)
                if i != 0:
                    lmks_paths += ";"
                    visage_paths += ";"
                    save_paths += ";"
                lmks_paths += lmks_path
                visage_paths += visage_path
                save_paths += f'output/visage{str(i)}_lmks2d.{lmk2D_format}'
            elif save_png:
                pass
        if save_lmks2D:
            getLandmark2D.run(visage_paths, lmks_paths, save_paths, save_png)


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
    vg = VisageGenerator(device, min_shape_param, max_shape_param, min_expression_param, max_expression_param, global_pose_param1, global_pose_param2, global_pose_param3, flame_model_path, batch_size, use_face_contour, use_3D_translation, shape_params, expression_params, static_landmark_embedding_path, dynamic_landmark_embedding_path)
    vg.generate(nb_faces, texturing, optimize_eyeballpose, optimize_neckpose)
    vg.save(save_obj, save_png, save_lmks2D, save_lmks3D, lmk2d_format)
    if view:
        vg.view()

if __name__ == "__main__":
    main()
