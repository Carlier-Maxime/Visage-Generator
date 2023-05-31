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

class VisageGenerator():
    def __init__(self, cfg: Config):
        self.flame_layer = FLAME(cfg.flame_model_path, cfg.batch_size, cfg.use_face_contour, cfg.use_3D_translation, int(sum(cfg.shape_params[::3])), int(sum(cfg.expression_params[::3])), cfg.static_landmark_embedding_path, cfg.dynamic_landmark_embedding_path).to(cfg.device)
        self.device = cfg.device
        self.params_generators = [ # shape, expression, pose, texture, neck, eye
            MultiParamsGenerator.from_params(cfg.shape_params, cfg.device),
            MultiParamsGenerator.from_params(cfg.expression_params, cfg.device),
            MultiParamsGenerator.from_params(cfg.pose_params, cfg.device, deg2rad=True),
            MultiParamsGenerator.from_params(cfg.texture_params, cfg.device) if cfg.texturing else BaseParamsGenerator(0,0,0,cfg.device),
            MultiParamsGenerator.from_params(cfg.neck_params, cfg.device, deg2rad=True),
            MultiParamsGenerator.from_params(cfg.eye_params, cfg.device, deg2rad=True),
            MultiParamsGenerator.from_params(cfg.camera_params, cfg.device)
        ]
        self.batch_size = cfg.batch_size
        self.filenames = None
        self.shape_params, self.expression_params, self.pose_params, self.texture_params, self.neck_pose, self.eye_pose, self.cameras = self.genParams(cfg) if cfg.input_folder is None else self.load_params(cfg)
        self._faces = self.flame_layer.faces
        self._textures = None
        if cfg.texturing:
            print("Loading Texture... ", end="", flush=True)
            tex_space = np.load("model/FLAME_texture.npz")
            self.texture_mean = tex_space['mean'].reshape(1, -1)
            self.texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            self.texture_mean = torch.from_numpy(self.texture_mean).float()[None, ...].to(self.device)
            self.texture_basis = torch.from_numpy(self.texture_basis[:, :50]).float()[None, ...].to(self.device)
            print("Done")
        else: self.texture_mean = None

        self.render = Renderer(cfg.img_resolution[0], cfg.img_resolution[1], device=self.device, show=cfg.show_window, camera=cfg.camera)
        self.markers = np.load("markers.npy") if cfg.save_markers else None
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
        assert index>=0 and index<self.nbFaces()
        batch_index = index//self.batch_size
        if self.batch_index is None or self.batch_index!=batch_index: self.generate_batch(batch_index)
        i = index%self.batch_size
        return self._vertices[i], self._textures[i], self._lmks[i]
    
    def nbFaces(self):
        return self.shape_params.shape[0]

    def genParams(self, cfg: Config):
        print('Generate random parameters')
        fixed = [cfg.fixed_shape, cfg.fixed_expression, cfg.fixed_pose, cfg.fixed_texture, cfg.fixed_neck, cfg.fixed_eye, cfg.fixed_cameras]
        return [generator.zeros(cfg.nb_faces) if cfg.zeros_params else generator.get(cfg.nb_faces, fix) for generator, fix in zip(self.params_generators, fixed)]

    def load_params(self, cfg: Config):
        if cfg.nb_faces==-1: cfg.nb_faces = sum(len(files) for _, _, files in os.walk(cfg.input_folder))
        pbar = trange(cfg.nb_faces, desc='load params', unit='visage')
        all_params = [[] for _ in range(len(self.params_generators))]
        keys = ['shape', 'expression', 'pose', 'texture', 'neck_pose', 'eye_pose', 'cameras']
        self.filenames = []
        for root, _, filenames in os.walk(cfg.input_folder):
            for filename in filenames:
                file = os.path.join(root, filename)
                if filename.endswith(('.npy')):
                    params = np.load(file, allow_pickle=True).item()
                    for i, gen in enumerate(self.params_generators):
                        if gen is None: continue 
                        all_params[i].append(torch.tensor(params[keys[i]], device=cfg.device) if keys[i] in params else gen.zeros() if cfg.zeros_params else gen.one())
                        assert all_params[i][-1].shape[0] == gen.nbParams(), f'{keys[i]} params not a good, expected {gen.nbParams()}, but got {all_params[i][-1].shape[0]} ! (file : {filename})'
                    self.filenames.append(filename.split('.')[0])
                pbar.update(1)
                if pbar.n >= cfg.nb_faces: break
            if pbar.n >= cfg.nb_faces: break
        pbar.close()
        for i, gen in enumerate(self.params_generators):
            if all_params[i] is not None: all_params[i] = torch.cat(all_params[i]).reshape(cfg.nb_faces, gen.nbParams())
        return all_params

    def generate_batch(self, batch_index:int):
        sp = self.shape_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        ep = self.expression_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        pp = self.pose_params[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        neck = self.neck_pose[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        eye = self.eye_pose[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        self._vertices, self._lmks = self.flame_layer(sp, ep, pp, neck, eye)

        if self.texture_mean is not None:
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
            index = self.batch_index*self.batch_size+i
            basename=format(index,'08d') if self.filenames is None else self.filenames[i]
            if cfg.random_bg: self.render.randomBackground()
            
            self.obj_Saver(index, basename+'.obj', vertices.cpu().numpy(), self._faces, texture=texture)
            self.lmk3D_npy_Saver(index, basename+'.npy', lmk.cpu().numpy())
            self.lmk2D_Saver(index, basename+f'.{cfg.lmk2D_format}', lmk)
            self.visage_png_Saver(index, basename+'.png', vertices, texture, camera=camera)
            self.lmk3D_png_Saver(index, basename+'.png', vertices, texture, pts=lmk, ptsInAlpha=cfg.pts_in_alpha, camera=camera)
            self.markers_png_Saver(index, basename+'.png', vertices, texture, pts=torch.tensor(np.array(util.read_all_index_opti_tri(vertices, self._faces, self.markers)), device=self.device) if self.markers is not None else None, ptsInAlpha=cfg.pts_in_alpha, camera=camera)
            self.camera_default_Saver(index, basename+'.pt', camera)
            self.camera_matrices_Saver(index, basename+'.pt', self.render.getCameraMatrices(camera))
            self.camera_json_Saver(index, basename, camera)
    
    def save_all(self, cfg:Config):
        pbar = tqdm(total=cfg.nb_faces, desc='saving all visages', unit='visage')
        for i in range(cfg.nb_faces//self.batch_size+(1 if cfg.nb_faces%self.batch_size>0 else 0)):
            self.generate_batch(i)
            self.save_batch(cfg, leave_pbar=False)
            pbar.update(self._vertices.shape[0])
        pbar.close()

def click_callback_strToList(ctx:click.Context, param:click.Parameter, value):
    return Config._str_to_list(value, param.metavar)

@click.command()
# General
@click.option('--nb-faces', type=int, default=1000, help='number faces generate')
@click.option('--not-texturing', 'texturing', type=bool,  default=True,  help='disable texture', is_flag=True)
@click.option('--device',  type=str,  default='cuda',  help='choice your device for generate face. ("cpu" or "cuda")')
@click.option('--view',  type=bool,  default=False,  help='enable view', is_flag=True)
@click.option('--batch-size', type=int, default=32, help='number of visage generate in the same time')
@click.option('--camera', type=str, metavar=float, default=[10.,0.,0.,-2.,0.,0.,0.], help='default camera for renderer [fov, tx, ty, tz, rx, ry, rz] (rotation in degree)', callback=click_callback_strToList)

# Generator parameter
@click.option('--input-folder', type=str, default=None, help='input folder for load parameter')
@click.option('--zeros-params', type=bool, default=False, help='zeros for all params not loaded', is_flag=True)
@click.option('--shape-params', type=str, metavar=float, default=[300,-2,2], help='Shape parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. default : sum(nX)==300', callback=click_callback_strToList)
@click.option('--expression-params', type=str, metavar=float, default=[100,-2,2], help='Expression parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. default : sum(nX)==100', callback=click_callback_strToList)
@click.option('--pose-params', type=str, metavar=float, default=[3,0,0, 1,0,30, 2,-10,10], help='Pose parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==6 (min, max in degree)', callback=click_callback_strToList)
@click.option('--texture-params', type=str, metavar=float, default=[50,-2,2], help='Texture parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==50', callback=click_callback_strToList)
@click.option('--neck-params', type=str, metavar=float, default=[3,-30,30], help='Neck parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==3 (min, max in degree)', callback=click_callback_strToList)
@click.option('--eye-params', type=str, metavar=float, default=[6,0,0], help='Eye parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==6', callback=click_callback_strToList)
@click.option('--camera-params', type=str, metavar=float, default=[1,8,12, 2,-0.05,0.05, 1,-2.1,-1.9, 3,-30,30], help='Camera parameter intervals. Format: [n1,min1,max1,n2,min2,max2,...]. sum(nX)==7, params order : [fov, tx, ty, tz, rx, ry, rz]. (rotation in degree)', callback=click_callback_strToList)
@click.option('--fixed-shape', type=bool, default=False, help='fixed the same shape for all visage generated', is_flag=True)
@click.option('--fixed-expression', type=bool, default=False, help='fixed the same expression for all visage generated', is_flag=True)
@click.option('--fixed-pose', type=bool, default=False, help='fixed the same jaw for all visage generated', is_flag=True)
@click.option('--fixed-texture', type=bool, default=False, help='fixed the same texture for all visage generated', is_flag=True)
@click.option('--fixed-neck', type=bool, default=False, help='fixed the same neck for all visage generated', is_flag=True)
@click.option('--fixed-eye', type=bool, default=False, help='fixed the same eye for all visage generated', is_flag=True)
@click.option('--fixed-cameras', type=bool, default=False, help='fixed the same cameras for all visage generated', is_flag=True)

# Flame parameter
@click.option('--not-use-face-contour', 'use_face_contour', type=bool, default=True, is_flag=True, help='not use face contour for generate visage')
@click.option('--not-use-3D-translation', 'use_3D_translation', type=bool, default=True, is_flag=True, help='not use 3D translation for generate visage')

# Saving
@click.option('--outdir', type=str, default='output', help='path directory for output')
@click.option('--lmk2D-format', 'lmk2D_format', type=str, default='npy', help='format used for save lmk2d. (npy and pts is supported)')
@click.option('--save-obj',  type=bool,  default=False,  help='enable save into file obj', is_flag=True)
@click.option('--save-png',  type=bool,  default=False,  help='enable save into file png', is_flag=True)
@click.option('--random-bg', type=bool, default=False, help='enable random background color for renderer', is_flag=True)
@click.option('--save-lmks3D-npy', 'save_lmks3D_npy', type=bool,  default=False,  help='enable save landmarks 3D into file npy', is_flag=True)
@click.option('--save-lmks3D-png', 'save_lmks3D_png', type=bool,  default=False,  help='enable save landmarks 3D with visage into file png', is_flag=True)
@click.option('--save-lmks2D', 'save_lmks2D',  type=bool,  default=False,  help='enable save landmarks 2D into file npy', is_flag=True)
@click.option('--save-markers', type=bool,  default=False,  help='enable save markers into png file', is_flag=True)
@click.option('--img-resolution', type=str, metavar=int, default=[512,512], help='resolution of image', callback=click_callback_strToList)
@click.option('--show-window', type=bool,  default=False,  help='show window during save png (enable if images is the screenshot or full black)', is_flag=True)
@click.option('--not-pts-in-alpha', 'pts_in_alpha', type=bool, default=True, help='not save landmarks/markers png version to channel alpha', is_flag=True)
@click.option('--save-camera-default', type=bool, default=False, help='save camera in default format', is_flag=True)
@click.option('--save-camera-matrices', type=bool, default=False, help='save camera in matrices format', is_flag=True)
@click.option('--save-camera-json', type=bool, default=False, help='save camera in json format', is_flag=True)

# Path
@click.option('--flame-model-path', type=str, default='./model/flame2023.pkl', help='path for acess flame model')
@click.option('--static-landmark-embedding-path', type=str, default='./model/flame_static_embedding.pkl', help='path for static landmark embedding file')
@click.option('--dynamic-landmark-embedding-path', type=str, default='./model/flame_dynamic_embedding.npy', help='path for dynamic landmark embedding file')

def main(**kwargs):
    cfg = Config(**kwargs)
    vg = VisageGenerator(cfg)
    vg.save_all(cfg)
    if cfg.view: vg.view(cfg)

if __name__ == "__main__":
    main()
