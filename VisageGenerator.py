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
from tqdm import trange
import util
from Params import *

radian = torch.pi / 180.0

class VisageGenerator():
    def __init__(self, cfg: Config):
        self.flame_layer = FLAME(cfg.flame_model_path, cfg.batch_size, cfg.use_face_contour, cfg.use_3D_translation, cfg.shape_params, cfg.expression_params, cfg.static_landmark_embedding_path, cfg.dynamic_landmark_embedding_path).to(cfg.device)
        self.device = cfg.device
        self.shape_params_generator = ParamsGenerator(cfg.shape_params, cfg.min_shape_param, cfg.max_shape_param, cfg.device)
        self.expression_params_generator = ParamsGenerator(cfg.expression_params, cfg.min_expression_param, cfg.max_expression_param, cfg.device)
        self.pose_params_generator = MultiParamsGenerator([3,1,2], [cfg.min_rotation_param * radian, cfg.min_jaw_param1 * radian, cfg.min_jaw_param2_3 * radian], [cfg.max_rotation_param * radian, cfg.max_jaw_param1 * radian, cfg.max_jaw_param2_3 * radian], cfg.device)
        self.texture_params_generator = ParamsGenerator(50, cfg.min_texture_param, cfg.max_texture_param, cfg.device) if cfg.texturing else BaseParamsGenerator(0,0,0,cfg.device)
        self.neck_params_generator = ParamsGenerator(3, cfg.min_neck_param*radian, cfg.max_neck_param*radian, cfg.device)
        self.eye_params_generator = ParamsGenerator(6,0,0,cfg.device)
        self.params_generators = [
            self.shape_params_generator,
            self.pose_params_generator,
            self.expression_params_generator,
            self.texture_params_generator,
            self.neck_params_generator,
            self.eye_params_generator
        ]
        self.batch_size = cfg.batch_size
        self.cameras = None
        self.filenames = None

    def save_obj(self, path: str, vertices: np.ndarray, faces: np.ndarray = None, texture: np.ndarray = None) -> None:
        """
        Save 3D object to the obj format
        Args:
            path (str): save path
            vertices (ndarray): array of all vertex
            faces (ndarray): array of all face
            texture (ndarray): texture

        Returns: None

        """
        basename = path.split(".obj")[0]
        if faces is None:
            faces = self._faces
        if texture is not None:
            img = PIL.Image.fromarray(texture)
            img.save(basename+"_texture.png")
            uvcoords = self.render.uvcoords.cpu().numpy().reshape((-1, 2))
            uvfaces = self.render.uvfaces.cpu().numpy()
            with open(basename+".mtl","w") as f:
                f.write(f'newmtl material_0\nmap_Kd {basename.split("/")[-1]}_texture.png\n')
        with open(path, 'w') as f:
            if texture is not None:
                f.write(f'mtllib {basename.split("/")[-1]}.mtl\n')
            np.savetxt(f, vertices, fmt='v %.6f %.6f %.6f')
            if texture is None:
                np.savetxt(f, faces + 1, fmt='f %d %d %d')
            else:
                np.savetxt(f, uvcoords, fmt='vt %.6f %.6f')
                f.write(f'usemtl material_0\n')
                np.savetxt(f, np.hstack((faces + 1, uvfaces + 1))[:,[0,3,1,4,2,5]], fmt='f %d/%d %d/%d %d/%d')


    def view(self, cfg: Config, other_objects=None) -> None:
        """
        View visage generate
        Args:
            other_objects: other object to display with faces

        Returns: None
        """
        Viewer(self._vertex, self._textures, self._landmark, self._faces, other_objects=other_objects, device=self.device, window_size=cfg.img_resolution, cameras=self.cameras)

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

    def generate(self, cfg: Config):
        shape_params, pose_params, expression_params, texture_params, neck_pose, eye_pose = self.genParams(cfg) if cfg.input_folder is None else self.load_params(cfg)
        if cfg.pose_for_camera:
            if self.cameras is None: self.cameras = torch.tensor(cfg.camera, device=self.device).repeat(cfg.nb_faces,1)
            self.cameras[:,4:] = pose_params[:,:3] / radian
            pose_params[:,:3] = 0

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


    def save(self, cfg:Config):
        out = cfg.outdir
        tmp = 'tmp'
        outObj = (out if cfg.save_obj else tmp)+"/obj"
        outLmk3D_npy = (out if cfg.save_lmks3D_npy else tmp)+"/lmks/3D"
        outLmk2D = (out if cfg.save_lmks2D else tmp)+"/lmks/2D"
        outVisagePNG = (out if cfg.save_png else tmp)+"/png/default"
        outLmks3D_PNG = (out if cfg.save_lmks3D_png else tmp)+"/png/lmks"
        outMarkersPNG = (out if cfg.save_markers else tmp)+"/png/markers"
        outCamera = (out if cfg.save_camera else tmp)+"/camera"
        for folder in [outObj, outLmk3D_npy, outLmk2D, outVisagePNG, outLmks3D_PNG, outMarkersPNG, outCamera]: os.makedirs(folder, exist_ok=True)
        if cfg.save_markers: markers = np.load("markers.npy")
        save_any_png = cfg.save_png or cfg.save_lmks3D_png or cfg.save_markers
        self.render = Renderer(cfg.img_resolution[0], cfg.img_resolution[1], device=self.device, show=cfg.show_window, camera=cfg.camera)
        if cfg.camera_format=='json': cameras_json = {'labels': []}
        for i in trange(len(self._vertex), desc='saving', unit='visage'):
            vertices = self._vertex[i].to(self.device)
            lmk = self._landmark[i].to(self.device)
            camera = None if self.cameras is None else self.cameras[i]
            if self._textures is None: texture=None
            else: 
                texture = self._textures[i].to(self.device)
                texture = texture * 255
                texture = texture.detach().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            basename=f"visage{str(i)}" if self.filenames is None else self.filenames[i]
            visage_path = f'{outObj}/{basename}.obj'
            lmks3Dnpy_path = f'{outLmk3D_npy}/{basename}.npy'
            lmks2D_path = f'{outLmk2D}/{basename}.{cfg.lmk2D_format}'
            if cfg.save_obj :
                self.save_obj(visage_path, vertices.cpu().numpy(), texture=texture)
            if cfg.save_lmks3D_npy :
                np.save(lmks3Dnpy_path, lmk.cpu().numpy())
            if cfg.save_lmks2D:
                lmks2D = []
                for p in lmk:
                    lmks2D.append(self.render.getCoord2D(p))
                if lmks2D_path.endswith('.npy'):
                    np.save(lmks2D_path, lmks2D[17:])
                elif lmks2D_path.endswith('.pts'):
                    with open(lmks2D_path, 'w') as f:
                        lmks2D = lmks2D[17:]
                        for i in range(len(lmks2D)):
                            f.write(f'{i + 1} {lmks2D[i][0]} {lmks2D[i][1]} False\n')
                else: raise TypeError("format for saving landmarks 2D is not supported !")
            if save_any_png:
                if cfg.random_bg: self.render.randomBackground()
                if cfg.save_png: self.render.save_to_image(f'{outVisagePNG}/{basename}.png', vertices, texture, camera=camera)
                if cfg.save_lmks3D_png: self.render.save_to_image(f'{outLmks3D_PNG}/{basename}.png', vertices, texture, pts=lmk, ptsInAlpha=cfg.pts_in_alpha, camera=camera)
                if cfg.save_markers:
                    mks = util.read_all_index_opti_tri(vertices, self._faces, markers)
                    self.render.save_to_image(f'{outMarkersPNG}/{basename}.png', vertices, texture, pts=torch.tensor(np.array(mks), device=self.device), ptsInAlpha=cfg.pts_in_alpha, camera=camera)
            if cfg.save_camera:
                match cfg.camera_format:
                    case 'default': torch.save(camera, f'{outCamera}/{basename}.pt')
                    case 'matrices': torch.save(self.render.getCameraMatrices(), f'{outCamera}/{basename}.pt')
                    case 'json': 
                        intrinsic, extrinsic = self.render.getCameraMatrices()
                        intrinsic_norm = intrinsic[:3, :3] / intrinsic[2, 2]
                        cameras_json['labels'].append([basename, torch.cat([extrinsic.flatten(), intrinsic_norm.flatten()]).tolist()])
                    case _: TypeError('format of camera save is unknow')
        
        if cfg.save_camera and cfg.camera_format=='json':
            with open(f'{outCamera}/cameras.json', 'w') as file:
                import json
                json.dump(cameras_json, file)

cfg = Config()
@click.command()
# General
@click.option('--nb-faces', type=int, default=cfg.nb_faces, help='number faces generate')
@click.option('--not-texturing', 'texturing', type=bool,  default=cfg.texturing,  help='disable texture', is_flag=True)
@click.option('--device',  type=str,  default=cfg.device,  help='choice your device for generate face. ("cpu" or "cuda")')
@click.option('--view',  type=bool,  default=cfg.view,  help='enable view', is_flag=True)
@click.option('--batch-size', type=int, default=cfg.batch_size, help='number of visage generate in the same time')
@click.option('--texture-batch-size', type=int, default=cfg.texture_batch_size, help='number of texture generate in same time')
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
@click.option('--save-camera', type=bool, default=cfg.save_camera, help='save camera', is_flag=True)
@click.option('--camera-format', type=click.Choice(['default', 'matrices', 'json']), default=cfg.camera_format, help='format for saving camera')

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
        vg.view(cfg)

if __name__ == "__main__":
    main()
