"""
Author: Carlier Maxime
Copyright (c) 2023, Carlier Maxime
All rights reserved.
"""

import os

import numpy as np
import torch
import click
import PIL

from FLAME import FLAME
from Viewer import Viewer
from renderer import Renderer
from config import Config
from tqdm import trange, tqdm
import util
from Params import *
from savers import *

radian = torch.pi / 180.0

class VisageGenerator():
    def __init__(self, cfg: Config):
        self.flame_layer = FLAME(cfg.flame_model_path, cfg.batch_size, cfg.use_face_contour, cfg.use_3D_translation, cfg.shape_params, cfg.expression_params, cfg.static_landmark_embedding_path, cfg.dynamic_landmark_embedding_path).to(cfg.device)
        self.device = cfg.device
        self.params_generators = [ # shape, expression, pose, texture, neck, eye
            ParamsGenerator(cfg.shape_params, cfg.min_shape_param, cfg.max_shape_param, cfg.device),
            ParamsGenerator(cfg.expression_params, cfg.min_expression_param, cfg.max_expression_param, cfg.device),
            MultiParamsGenerator([3,1,2], [cfg.min_rotation_param * radian, cfg.min_jaw_param1 * radian, cfg.min_jaw_param2_3 * radian], [cfg.max_rotation_param * radian, cfg.max_jaw_param1 * radian, cfg.max_jaw_param2_3 * radian], cfg.device),
            ParamsGenerator(50, cfg.min_texture_param, cfg.max_texture_param, cfg.device) if cfg.texturing else BaseParamsGenerator(0,0,0,cfg.device),
            ParamsGenerator(3, cfg.min_neck_param*radian, cfg.max_neck_param*radian, cfg.device),
            ParamsGenerator(6,0,0,cfg.device)
        ]
        self.batch_size = cfg.batch_size
        self.cameras = None
        self.filenames = None
        self.shape_params, self.expression_params, self.pose_params, self.texture_params, self.neck_pose, self.eye_pose = self.genParams(cfg) if cfg.input_folder is None else self.load_params(cfg)
        if cfg.pose_for_camera:
            if self.cameras is None: self.cameras = torch.tensor(cfg.camera, device=self.device).repeat(cfg.nb_faces,1)
            self.cameras[:,4:] = self.pose_params[:,:3] / radian
            self.pose_params[:,:3] = 0
        self._faces = self.flame_layer.faces
        if cfg.texturing:
            tex_space = np.load("model/FLAME_texture.npz")
            self.texture_mean = tex_space['mean'].reshape(1, -1)
            self.texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            self.texture_mean = torch.from_numpy(self.texture_mean).float()[None, ...].to(self.device)
            self.texture_basis = torch.from_numpy(self.texture_basis[:, :50]).float()[None, ...].to(self.device)
            self._textures = None

        self.render = Renderer(cfg.img_resolution[0], cfg.img_resolution[1], device=self.device, show=cfg.show_window, camera=cfg.camera)
        self.markers = np.load("markers.npy") if cfg.save_markers else None
        if (cfg.save_camera_default or cfg.save_camera_matrices or cfg.save_camera_json) and not cfg.pose_for_camera: print("WARNING : pose for camera not enable, all camera is same (use --pose-for-camera) !!!")
        self.obj_Saver = ObjSaver(cfg.outdir+"/obj", self.render, cfg.save_obj)
        self.lmk3D_npy_Saver = NumpySaver(cfg.outdir+"/lmks/3D", cfg.save_lmks3D_npy)
        self.lmk2D_Saver = Lmks2DSaver(cfg.outdir+"/lmks/2D", self.render, cfg.save_lmks2D)
        self.visage_png_Saver = VisageImageSaver(cfg.outdir+"/png/default", self.render, cfg.save_png)
        self.lmk3D_png_Saver = VisageImageSaver(cfg.outdir+"/png/lmks", self.render, cfg.save_lmks3D_png)
        self.markers_png_Saver = VisageImageSaver(cfg.outdir+"/png/markers", self.render, cfg.save_markers)
        self.camera_default_Saver = TorchSaver(cfg.outdir+"/camera/default", cfg.save_camera_default)
        self.camera_matrices_Saver = TorchSaver(cfg.outdir+"/camera/matrices", cfg.save_camera_matrices)
        self.camera_json_Saver = CameraJSONSaver(cfg.outdir+"/camera", self.render, cfg.save_camera_json)
        self.batch_index=None

    def view(self, cfg: Config, other_objects=None) -> None:
        print("Open Viewer...")
        Viewer(self, other_objects=other_objects, device=self.device, window_size=cfg.img_resolution, cameras=self.cameras)

    def getFaces(self):
        return self._faces
    
    def getVisage(self, index:int):
        assert index>=0 and index<cfg.nb_faces
        batch_index = index//self.batch_size
        if self.batch_index is None or self.batch_index!=batch_index: self.generate_batch(batch_index)
        i = index%self.batch_size
        return self._vertices[i], self._textures[i], self._lmks[i]
    
    def nbFaces(self):
        return self.shape_params.shape[0]

    def genParams(self, cfg: Config):
        print('Generate random parameters')
        fixed = [cfg.fixed_shape, cfg.fixed_jaw, cfg.fixed_expression, cfg.fixed_texture, cfg.fixed_neck, False]
        return [generator.zeros(cfg.nb_faces) if cfg.zeros_params else generator.get(cfg.nb_faces, fix) for generator, fix in zip(self.params_generators, fixed)]

    def load_params(self, cfg: Config):
        if cfg.nb_faces==-1: cfg.nb_faces = sum(len(files) for _, _, files in os.walk(cfg.input_folder))
        pbar = trange(cfg.nb_faces, desc='load params', unit='visage')
        all_params = [[] for _ in range(len(self.params_generators))]
        keys = ['shape', 'pose', 'expression', 'texture', 'neck_pose', 'eye_pose']
        self.cameras = torch.tensor(cfg.camera, device=self.device).repeat(cfg.nb_faces,1)
        self.filenames = []
        for root, _, filenames in os.walk(cfg.input_folder):
            for filename in filenames:
                file = os.path.join(root, filename)
                if filename.endswith(('.npy')):
                    params = np.load(file, allow_pickle=True).item()
                    for i, gen in enumerate(self.params_generators):
                        if gen is not None: all_params[i].append(torch.tensor(params[keys[i]], device=cfg.device) if keys[i] in params else gen.zeros() if cfg.zeros_params else gen.one())
                    assert all_params[0][-1].shape[0] == cfg.shape_params, f'shape params not a good, expected {cfg.shape_params}, but got {all_params[0][-1].shape[0]} ! (file : {filename})'
                    assert all_params[2][-1].shape[0] == cfg.expression_params, f'expression params not a good, expected {cfg.expression_params}, but got {all_params[2][-1].shape[0]} ! (file : {filename})'
                    if 'cam' in params: 
                        cam = params['cam']
                        if len(cam)==3: self.cameras[pbar.n,:3]=cam
                        else: self.cameras[pbar.n]=cam
                    self.filenames.append(filename.split('.')[0])
                pbar.update(1)
                if pbar.n >= cfg.nb_faces: break
            if pbar.n >= cfg.nb_faces: break
        pbar.close()
        for i, element in enumerate(zip(all_params, [cfg.shape_params, 6, cfg.expression_params, 50, 3, 6])):
            params, nb_params = element
            if all_params[i] is not None: all_params[i] = torch.cat(params).reshape(cfg.nb_faces, nb_params)
        return all_params

    def generate_batch(self, batch_index:int):
        sp = self.shape_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        ep = self.expression_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        pp = self.pose_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        neck = self.neck_pose[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        eye = self.eye_pose[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        self._vertices, self._lmks = self.flame_layer(sp, ep, pp, neck, eye)

        if cfg.texturing:
            tp = self.texture_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
            self._textures = self.texture_mean + (self.texture_basis * tp[:, None, :]).sum(-1)
            self._textures = self._textures.reshape(tp.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
            self._textures = self._textures[:, [2, 1, 0], :, :]
            self._textures = self._textures / 255
        self.batch_index = batch_index

    def save_batch(self, cfg:Config, leave_pbar:bool=True):
        for i in trange(len(self._vertices), desc='saving batch', unit='visage', leave=leave_pbar):
            vertices = self._vertices[i].to(self.device)
            lmk = self._lmks[i].to(self.device)
            camera = self.render.getCamera() if self.cameras is None else self.cameras[i]
            if self._textures is None: texture=None
            else: 
                texture = self._textures[i].to(self.device)
                texture = texture * 255
                texture = texture.detach().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            basename=format(i,'08d') if self.filenames is None else self.filenames[i]
            if cfg.random_bg: self.render.randomBackground()
            
            self.obj_Saver(i, basename+'.obj', vertices.cpu().numpy(), self._faces, texture=texture)
            self.lmk3D_npy_Saver(i, basename+'.npy', lmk.cpu().numpy())
            self.lmk2D_Saver(i, basename+f'.{cfg.lmk2D_format}', lmk)
            self.visage_png_Saver(i, basename+'.png', vertices, texture, camera=camera)
            self.lmk3D_png_Saver(i, basename+'.png', vertices, texture, pts=lmk, ptsInAlpha=cfg.pts_in_alpha, camera=camera)
            self.markers_png_Saver(i, basename+'.png', vertices, texture, pts=torch.tensor(np.array(util.read_all_index_opti_tri(vertices, self._faces, self.markers)), device=self.device) if self.markers is not None else None, ptsInAlpha=cfg.pts_in_alpha, camera=camera)
            self.camera_default_Saver(i, basename+'.pt', camera)
            self.camera_matrices_Saver(i, basename+'.pt', self.render.getCameraMatrices(camera))
            self.camera_json_Saver(i, basename, camera)
    
    def save_all(self, cfg:Config):
        pbar = tqdm(total=cfg.nb_faces, desc='saving all visages', unit='visage')
        for i in range(cfg.nb_faces//self.batch_size+(1 if cfg.nb_faces%self.batch_size>0 else 0)):
            self.generate_batch(i)
            self.save_batch(cfg, leave_pbar=False)
            pbar.update(self._vertices.shape[0])
        pbar.close()

cfg = Config()
@click.command()
# General
@click.option('--nb-faces', type=int, default=cfg.nb_faces, help='number faces generate')
@click.option('--not-texturing', 'texturing', type=bool,  default=cfg.texturing,  help='disable texture', is_flag=True)
@click.option('--device',  type=str,  default=cfg.device,  help='choice your device for generate face. ("cpu" or "cuda")')
@click.option('--view',  type=bool,  default=cfg.view,  help='enable view', is_flag=True)
@click.option('--batch-size', type=int, default=cfg.batch_size, help='number of visage generate in the same time')
@click.option('--pose-for-camera', type=bool, default=cfg.pose_for_camera, help='use pose rotation parameter for camera instead of visage generation', is_flag=True)
@click.option('--camera', type=str, default=cfg.camera, help='default camera for renderer')

# Generator parameter
@click.option('--input-folder', type=str, default=cfg.input_folder, help='input folder for load parameter (default : None)')
@click.option('--zeros-params', type=bool, default=cfg.zeros_params, help='zeros for all params not loaded', is_flag=True)
@click.option('--min-shape-param',  type=float,  default=cfg.min_shape_param,  help='minimum value for shape param')
@click.option('--max-shape-param',  type=float,  default=cfg.max_shape_param,  help='maximum value for shape param')
@click.option('--min-expression-param',  type=float,  default=cfg.min_expression_param,  help='minimum value for expression param')
@click.option('--max-expression-param',  type=float,  default=cfg.max_expression_param,  help='maximum value for expression param')
@click.option('--min-rotation-param',  type=float,  default=cfg.min_rotation_param,  help='minimum value for rotation param')
@click.option('--max-rotation-param',  type=float,  default=cfg.max_rotation_param,  help='maximum value for rotation param')
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
@click.option('--outdir', type=str, default=cfg.outdir, help='path directory for output')
@click.option('--lmk2D-format', 'lmk2D_format', type=str, default=cfg.lmk2D_format, help='format used for save lmk2d. (npy and pts is supported)')
@click.option('--save-obj',  type=bool,  default=cfg.save_obj,  help='enable save into file obj', is_flag=True)
@click.option('--save-png',  type=bool,  default=cfg.save_png,  help='enable save into file png', is_flag=True)
@click.option('--random-bg', type=bool, default=cfg.random_bg, help='enable random background color for renderer', is_flag=True)
@click.option('--save-lmks3D-npy', 'save_lmks3D_npy', type=bool,  default=cfg.save_lmks3D_npy,  help='enable save landmarks 3D into file npy', is_flag=True)
@click.option('--save-lmks3D-png', 'save_lmks3D_png', type=bool,  default=cfg.save_lmks3D_png,  help='enable save landmarks 3D with visage into file png', is_flag=True)
@click.option('--save-lmks2D', 'save_lmks2D',  type=bool,  default=cfg.save_lmks2D,  help='enable save landmarks 2D into file npy', is_flag=True)
@click.option('--save-markers', type=bool,  default=cfg.save_markers,  help='enable save markers into png file', is_flag=True)
@click.option('--img-resolution', type=str, default=cfg.img_resolution, help='resolution of image')
@click.option('--show-window', type=bool,  default=cfg.show_window,  help='show window during save png (enable if images is the screenshot or full black)', is_flag=True)
@click.option('--not-pts-in-alpha', 'pts_in_alpha', type=bool, default=cfg.pts_in_alpha, help='not save landmarks/markers png version to channel alpha', is_flag=True)
@click.option('--save-camera-default', type=bool, default=cfg.save_camera_default, help='save camera in default format', is_flag=True)
@click.option('--save-camera-matrices', type=bool, default=cfg.save_camera_matrices, help='save camera in matrices format', is_flag=True)
@click.option('--save-camera-json', type=bool, default=cfg.save_camera_json, help='save camera in json format', is_flag=True)

# Path
@click.option('--flame-model-path', type=str, default=cfg.flame_model_path, help='path for acess flame model')
@click.option('--static-landmark-embedding-path', type=str, default=cfg.static_landmark_embedding_path, help='path for static landmark embedding file')
@click.option('--dynamic-landmark-embedding-path', type=str, default=cfg.dynamic_landmark_embedding_path, help='path for dynamic landmark embedding file')

def main(**kwargs):
    cfg.set(**kwargs)
    vg = VisageGenerator(cfg)
    vg.save_all(cfg)
    if cfg.view: vg.view(cfg)

if __name__ == "__main__":
    main()
