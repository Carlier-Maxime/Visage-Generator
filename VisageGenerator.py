"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Edited by M_EtOuais for Visage-Generator
"""

import os

import numpy as np
import torch
import click
import PIL

import getLandmark2D
from FLAME import FLAME
from Viewer import Viewer
from renderer import Renderer
from config import Config
from tqdm import trange
import util

class VisageGenerator():
    def __init__(self, cfg):
        self.flame_layer = FLAME(cfg.flame_model_path, cfg.batch_size, cfg.use_face_contour, cfg.use_3D_translation, cfg.shape_params, cfg.expression_params, cfg.static_landmark_embedding_path, cfg.dynamic_landmark_embedding_path).to(cfg.device)
        self.device = cfg.device
        self.min_shape_param = cfg.min_shape_param
        self.max_shape_param = cfg.max_shape_param
        self.min_expression_param = cfg.min_expression_param
        self.max_expression_param = cfg.max_expression_param
        self.global_pose_param1 = cfg.global_pose_param1
        self.global_pose_param2 = cfg.global_pose_param2
        self.global_pose_param3 = cfg.global_pose_param3
        self.min_jaw_param1 = cfg.min_jaw_param1
        self.max_jaw_param1 = cfg.max_jaw_param1
        self.min_jaw_param2_3 = cfg.min_jaw_param2_3
        self.max_jaw_param2_3 = cfg.max_jaw_param2_3
        self.min_texture_param = cfg.min_texture_param
        self.max_texture_param = cfg.max_texture_param
        self.min_neck_param = cfg.min_neck_param
        self.max_neck_param = cfg.max_neck_param
        self.batch_size = cfg.batch_size

    def save_obj(self, path: str, vertices: list, faces=None, texture=None) -> None:
        """
        Save 3D object to the obj format
        Args:
            path (str): save path
            vertices (list): array of all vertex
            faces (list): array of all face
            texture: texture

        Returns: None

        """
        uvcoords = []
        uvfaces = []
        basename = path.split(".obj")[0]
        if faces is None: faces = self.render.faces[0].detach().cpu().numpy()
        if texture is not None:
            img = PIL.Image.fromarray(texture)
            img.save(basename+"_texture.png")
            uvcoords = self.render.uvcoords.cpu().numpy().reshape((-1, 2))
            uvfaces=self.render.uvfaces
            with open(basename+".mtl","w") as f:
                f.write(f'newmtl material_0\nmap_Kd {basename.split("/")[-1]}_texture.png\n')
        with open(path, 'w') as f:
            if texture is not None: f.write(f'mtllib {basename.split("/")[-1]}.mtl\n')
            for v in vertices: f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for vt in uvcoords: f.write(f'vt {vt[0]} {vt[1]}\n')
            if texture is None:
                for face in faces: f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
            else:
                f.write(f'usemtl material_0\n')
                for face, ivt in zip(faces, uvfaces): f.write(f'f {face[0]+1}/{ivt[0]+1} {face[1]+1}/{ivt[1]+1} {face[2]+1}/{ivt[2]+1}\n')

    def view(self, other_objects=None) -> None:
        """
        View visage generate
        Args:
            other_objects: other object to display with faces

        Returns: None
        """
        Viewer(self._vertex, self._textures, self._landmark, self._faces, other_objects=other_objects, device=self.device)

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

    def genParams(self, cfg):
        print('Generate random parameters')
        radian = torch.pi / 180.0
        shape_params = torch.rand(1 if cfg.fixed_shape else cfg.nb_faces, 300, dtype=torch.float32, device=self.device) * (self.max_shape_param - self.min_shape_param) + self.min_shape_param
        if cfg.fixed_shape: shape_params = shape_params.repeat(cfg.nb_faces, 1)
        pose_params = torch.tensor([[self.global_pose_param1 * radian, self.global_pose_param2 * radian, self.global_pose_param3 * radian, 0, 0, 0]], dtype=torch.float32, device=self.device).repeat(cfg.nb_faces, 1)
        jaw1_param = (torch.rand(1 if cfg.fixed_jaw else cfg.nb_faces, dtype=torch.float32, device=self.device) * (self.max_jaw_param1 - self.min_jaw_param1) + self.min_jaw_param1) * radian
        jaw2_3_param = (torch.rand(1 if cfg.fixed_jaw else cfg.nb_faces, 2, dtype=torch.float32, device=self.device) * (self.max_jaw_param2_3 - self.min_jaw_param2_3) + self.min_jaw_param2_3) * radian 
        jaw_params = torch.cat([jaw1_param[:,None], jaw2_3_param], dim=1)
        if cfg.fixed_jaw: jaw_params = jaw_params.repeat(cfg.nb_faces, 1)
        pose_params[:,3:6] = jaw_params
        expression_params = torch.rand(1 if cfg.fixed_expression else cfg.nb_faces, 100, dtype=torch.float32, device=self.device) * (self.max_expression_param - self.min_expression_param) + self.min_expression_param
        if cfg.fixed_expression: expression_params = expression_params.repeat(cfg.nb_faces, 1)
        neck_pose = (torch.rand(1 if cfg.fixed_neck else cfg.nb_faces, 3, dtype=torch.float32, device=self.device) * (self.max_neck_param - self.min_neck_param) + self.min_neck_param) * radian
        if cfg.fixed_neck: neck_pose = neck_pose.repeat(cfg.nb_faces, 1)
        eye_pose = torch.zeros(cfg.nb_faces, 6).to(self.device)
        if cfg.texturing: 
            texture_params = torch.rand(1 if cfg.fixed_texture else cfg.nb_faces, 50, dtype=torch.float32, device=self.device) * (self.max_texture_param - self.min_texture_param) + self.min_texture_param
            if cfg.fixed_texture: texture_params = texture_params.repeat(cfg.nb_faces, 1)
        else: texture_params = None
        return shape_params, pose_params, expression_params, texture_params, neck_pose, eye_pose

    def generate(self, cfg):
        shape_params, pose_params, expression_params, texture_params, neck_pose, eye_pose = self.genParams(cfg)

        self._vertex = []
        self._landmark = []
        for i in trange(cfg.nb_faces//self.batch_size+(1 if cfg.nb_faces%self.batch_size>0 else 0), desc='generate visages', unit='step'):
            sp = shape_params[i*self.batch_size:(i+1)*self.batch_size]
            ep = expression_params[i*self.batch_size:(i+1)*self.batch_size]
            pp = pose_params[i*self.batch_size:(i+1)*self.batch_size]
            neck = neck_pose[i*self.batch_size:(i+1)*self.batch_size]
            eye = eye_pose[i*self.batch_size:(i+1)*self.batch_size]
            vertices, lmks = self.flame_layer(sp, ep, pp, neck, eye)
            self._vertex.append(vertices.cpu())
            self._landmark.append(lmks.cpu())
        self._vertex = torch.cat(self._vertex)
        self._landmark = torch.cat(self._landmark)
        self._faces = self.flame_layer.faces

        self._textures = None
        if cfg.texturing:
            tex_space = np.load("model/FLAME_texture.npz")
            texture_mean = tex_space['mean'].reshape(1, -1)
            texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            texture_mean = torch.from_numpy(texture_mean).float()[None, ...].to(self.device)
            texture_basis = torch.from_numpy(texture_basis[:, :50]).float()[None, ...].to(self.device)
            self._textures = torch.zeros((cfg.nb_faces, 3, 512, 512), dtype=torch.float32, device='cpu')
            for i in trange(cfg.nb_faces//cfg.texture_batch_size+(1 if cfg.nb_faces%cfg.texture_batch_size>0 else 0), desc='texturing', unit='step'):
                tp = texture_params[i*cfg.texture_batch_size:(i+1)*cfg.texture_batch_size]
                texture = texture_mean + (texture_basis * tp[:, None, :]).sum(-1)
                texture = texture.reshape(tp.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
                texture = texture[:, [2, 1, 0], :, :]
                texture = texture / 255
                self._textures[i*cfg.texture_batch_size:(i+1)*cfg.texture_batch_size] = texture.cpu()


    def save(self, cfg):
        out = 'output'
        tmp = 'tmp'
        outObj = (out if cfg.save_obj else tmp)+"/obj"
        outLmk3D_npy = (out if cfg.save_lmks3D_npy else tmp)+"/lmks/3D"
        outLmk2D = (out if cfg.save_lmks2D else tmp)+"/lmks/2D"
        outVisagePNG = (out if cfg.save_png else tmp)+"/png/default"
        outLmks3D_PNG = (out if cfg.save_lmks3D_png else tmp)+"/png/lmks"
        outMarkersPNG = (out if cfg.save_markers else tmp)+"/png/markers"
        for folder in [out, tmp, outObj, outLmk3D_npy, outLmk2D, outVisagePNG, outLmks3D_PNG, outMarkersPNG]: os.makedirs(folder, exist_ok=True)
        if cfg.save_lmks2D:
            lmks_paths = ""
            visage_paths = ""
            save_paths = ""
        if cfg.save_markers: markers = np.load("markers.npy")
        save_any_png = cfg.save_png or cfg.save_lmks3D_png or cfg.save_markers
        self.render = Renderer(cfg.img_resolution[0], cfg.img_resolution[1], device=self.device, show=cfg.show_window)
        for i in trange(len(self._vertex), desc='saving', unit='visage'):
            vertices = self._vertex[i].to(self.device)
            lmk = self._landmark[i].to(self.device)
            if self._textures is None: texture=None
            else: 
                texture = self._textures[i].to(self.device)
                texture = texture * 255
                texture = texture.detach().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            basename=f"visage{str(i)}"
            visage_path = f'{outObj}/{basename}.obj'
            lmks3Dnpy_path = f'{outLmk3D_npy}/{basename}.npy'
            if cfg.save_obj or cfg.save_lmks2D:
                self.save_obj(visage_path, vertices, texture=texture)
            if cfg.save_lmks3D_npy or cfg.save_lmks2D:
                np.save(lmks3Dnpy_path, lmk)
            if cfg.save_lmks2D:
                if i != 0:
                    lmks_paths += ";"
                    visage_paths += ";"
                    save_paths += ";"
                lmks_paths += lmks3Dnpy_path
                visage_paths += visage_path
                save_paths += f'{outLmk2D}/{basename}.{cfg.lmk2D_format}'
            if save_any_png:
                if cfg.save_png: self.render.save_to_image(f'{outVisagePNG}/{basename}.png', vertices, texture)
                if cfg.save_lmks3D_png: self.render.save_to_image(f'{outLmks3D_PNG}/{basename}.png', vertices, texture, pts=lmk, ptsInAlpha=cfg.pts_in_alpha)
                if cfg.save_markers:
                    mks = util.read_all_index_opti_tri(vertices, self._faces, markers)
                    self.render.save_to_image(f'{outMarkersPNG}/{basename}.png', vertices, texture, pts=torch.tensor(np.array(mks), device=self.device), ptsInAlpha=cfg.pts_in_alpha)
        if cfg.save_lmks2D:
            getLandmark2D.run(visage_paths, lmks_paths, save_paths, cfg.save_png)

cfg = Config()
@click.command()
# General
@click.option('--nb-faces', type=int, default=cfg.nb_faces, help='number faces generate')
@click.option('--not-texturing', 'texturing', type=bool,  default=cfg.texturing,  help='disable texture', is_flag=True)
@click.option('--device',  type=str,  default=cfg.device,  help='choice your device for generate face. ("cpu" or "cuda")')
@click.option('--view',  type=bool,  default=cfg.view,  help='enable view', is_flag=True)
@click.option('--batch-size', type=int, default=cfg.batch_size, help='number of visage generate in the same time')
@click.option('--texture-batch-size', type=int, default=cfg.texture_batch_size, help='number of texture generate in same time')

# Generator parameter
@click.option('--min-shape-param',  type=float,  default=cfg.min_shape_param,  help='minimum value for shape param')
@click.option('--max-shape-param',  type=float,  default=cfg.max_shape_param,  help='maximum value for shape param')
@click.option('--min-expression-param',  type=float,  default=cfg.min_expression_param,  help='minimum value for expression param')
@click.option('--max-expression-param',  type=float,  default=cfg.min_expression_param,  help='maximum value for expression param')
@click.option('--global-pose-param1',  type=float,  default=cfg.global_pose_param1,  help='value of first global pose param')
@click.option('--global-pose-param2',  type=float,  default=cfg.global_pose_param2,  help='value of second global pose param')
@click.option('--global-pose-param3',  type=float,  default=cfg.global_pose_param3,  help='value of third global pose param')
@click.option('--min-jaw-param1',  type=float,  default=cfg.min_jaw_param1,  help='minimum value for jaw param 1')
@click.option('--max-jaw-param1',  type=float,  default=cfg.max_jaw_param1,  help='maximum value for jaw param 1')
@click.option('--min-jaw-param2-3',  type=float,  default=cfg.min_jaw_param2_3,  help='minimum value for jaw param 2-3')
@click.option('--max-jaw-param2-3',  type=float,  default=cfg.max_jaw_param2_3,  help='maximum value for jaw param 2-3')
@click.option('--min-texture-param',  type=float,  default=cfg.min_texture_param,  help='minimum value for texture param')
@click.option('--max-texture-param',  type=float,  default=cfg.max_texture_param,  help='maximum value for texture param')
@click.option('--min-neck-param',  type=float,  default=cfg.min_neck_param,  help='minimum value for neck param')
@click.option('--max-neck-param',  type=float,  default=cfg.max_neck_param,  help='maximum value for neck param')
@click.option('--fixed-shape', type=bool, default=cfg.fixed_shape, help='fixed the same shape for all visage generated', is_flag=True)
@click.option('--fixed-expression', type=bool, default=cfg.fixed_expression, help='fixed the same expression for all visage generated', is_flag=True)
@click.option('--fixed-jaw', type=bool, default=cfg.fixed_jaw, help='fixed the same jaw for all visage generated', is_flag=True)
@click.option('--fixed-texture', type=bool, default=cfg.fixed_texture, help='fixed the same texture for all visage generated', is_flag=True)
@click.option('--fixed-neck', type=bool, default=cfg.fixed_neck, help='fixed the same neck for all visage generated', is_flag=True)

# Flame parameter
@click.option('--not-use-face-contour', 'use_face_contour', type=bool, default=cfg.use_face_contour, is_flag=True, help='not use face contour for generate visage')
@click.option('--not-use-3D-translation', 'use_3D_translation', type=bool, default=cfg.use_3D_translation, is_flag=True, help='not use 3D translation for generate visage')
@click.option('--shape-params', type=int, default=cfg.shape_params, help='a number of shape parameter used')
@click.option('--expression-params', type=int, default=cfg.expression_params, help='a number of expression parameter used')

# Saving
@click.option('--lmk2D-format', 'lmk2D_format', type=str, default=cfg.lmk2D_format, help='format used for save lmk2d. (npy and pts is supported)')
@click.option('--save-obj',  type=bool,  default=cfg.save_obj,  help='enable save into file obj', is_flag=True)
@click.option('--save-png',  type=bool,  default=cfg.save_png,  help='enable save into file png', is_flag=True)
@click.option('--save-lmks3D-npy', 'save_lmks3D_npy', type=bool,  default=cfg.save_lmks3D_npy,  help='enable save landmarks 3D into file npy', is_flag=True)
@click.option('--save-lmks3D-png', 'save_lmks3D_png', type=bool,  default=cfg.save_lmks3D_png,  help='enable save landmarks 3D with visage into file png', is_flag=True)
@click.option('--save-lmks2D', 'save_lmks2D',  type=bool,  default=cfg.save_lmks2D,  help='enable save landmarks 2D into file npy', is_flag=True)
@click.option('--save-markers', type=bool,  default=cfg.save_markers,  help='enable save markers into png file', is_flag=True)
@click.option('--img-resolution', type=str, default=cfg.img_resolution, help='resolution of image')
@click.option('--show-window', type=bool,  default=cfg.show_window,  help='show window during save png (enable if images is the screenshot or full black)', is_flag=True)
@click.option('--not-pts-in-alpha', 'pts_in_alpha', type=bool, default=cfg.pts_in_alpha, help='not save landmarks/markers png version to channel alpha', is_flag=True)

# Path
@click.option('--flame-model-path', type=str, default=cfg.flame_model_path, help='path for acess flame model')
@click.option('--static-landmark-embedding-path', type=str, default=cfg.static_landmark_embedding_path, help='path for static landmark embedding file')
@click.option('--dynamic-landmark-embedding-path', type=str, default=cfg.dynamic_landmark_embedding_path, help='path for dynamic landmark embedding file')

def main(**kwargs):
    cfg.set(**kwargs)
    img_resolution = cfg.img_resolution[1:-1].split(",")
    for i in range(len(img_resolution)): img_resolution[i] = int(img_resolution[i])
    cfg.img_resolution = img_resolution
    vg = VisageGenerator(cfg)
    vg.generate(cfg)
    vg.save(cfg)
    if cfg.view:
        vg.view()

if __name__ == "__main__":
    main()
