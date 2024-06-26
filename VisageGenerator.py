"""
Author: Carlier Maxime
Copyright (c) 2023, Carlier Maxime
All rights reserved.
"""

import cv2
import click
from tqdm import trange, tqdm

import util
from FLAME import FLAME
from Params import *
from Viewer import Viewer
from config import Config
from savers import *


class VisageGenerator:
    def __init__(self, cfg: Config):
        self.face_index = None
        self._latents = None
        self._lmks = None
        self._vertices = None
        self.flame_layer = FLAME(cfg.flame.model_path, cfg.general.batch_size, cfg.flame.face_contour, cfg.flame.translation_3D, int(sum(cfg.generator.shape.params[::3])), int(sum(cfg.generator.expression.params[::3])), cfg.flame.static_landmark_embedding_path, cfg.flame.dynamic_landmark_embedding_path).to(cfg.general.device)
        self.device = cfg.general.device
        self.params_generators = [  # shape, expression, pose, texture, neck, eye
            MultiParamsGenerator.from_params(cfg.generator.shape.params, self.device),
            MultiParamsGenerator.from_params(cfg.generator.expression.params, self.device),
            MultiParamsGenerator.from_params(cfg.generator.pose.params, self.device, deg2rad=True),
            MultiParamsGenerator.from_params(cfg.generator.texture.params, self.device) if cfg.general.texturing else BaseParamsGenerator(0, 0, 0, self.device),
            MultiParamsGenerator.from_params(cfg.generator.neck.params, self.device, deg2rad=True),
            MultiParamsGenerator.from_params(cfg.generator.eye.params, self.device, deg2rad=True),
            MultiParamsGenerator.from_params(cfg.generator.camera.params, self.device),
            MultiParamsGenerator.from_params(cfg.generator.ambient.params, self.device)
        ]
        self.batch_size = cfg.general.batch_size
        self.filenames = None
        self.shape_params = self.expression_params = self.pose_params = self.texture_params = self.neck_pose = self.eye_pose = self.cameras = self.ambient_lights = self._faces = None
        self._textures = None
        if cfg.general.texturing:
            print("Loading Texture... ", end="", flush=True)
            tex_space = np.load("model/FLAME_texture.npz")
            self.texture_mean = tex_space['mean'].reshape(1, -1)
            self.texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            self.texture_mean = torch.from_numpy(self.texture_mean).float()[None, ...].to(self.device)
            self.texture_basis = torch.from_numpy(self.texture_basis[:, :int(sum(cfg.generator.texture.params[::3]))]).float()[None, ...].to(self.device)
            print("Done")
        else:
            self.texture_mean = None
        self.default_camera = torch.tensor(cfg.generator.camera.default, device=self.device, dtype=torch.float32)
        self.batch_index = None
        self.coords_multiplier = cfg.generator.coords_multiplier
        if cfg.general.view:
            self.render = Viewer(self, other_objects=None, device=self.device, window_size=cfg.save.global_.img_resolution, cameras=self.cameras, camera_type=cfg.generator.camera.type)
        else:
            self.render = Renderer(cfg.save.global_.img_resolution[0], cfg.save.global_.img_resolution[1], device=self.device, show=cfg.save.global_.show_window, camera=self.default_camera, camera_type=cfg.generator.camera.type)
        self.savers = Saver.load(cfg.save, renderer=self.render, device=self.device)
        self.markers = torch.load("markers.pt").to(self.device) if cfg.save.markers2D.enable or cfg.save.markers3D.enable else None

    def view(self, cfg: Config, other_objects=None) -> None:
        print("Open Viewer...")
        if cfg.general.view:
            self.render.set_other_objects(other_objects)
            self.render.loop()
        else:
            raise NotImplemented

    def get_faces(self):
        return self._faces

    def get_visage(self, index: int):
        assert 0 <= index < self.nb_faces()
        batch_index = index // self.batch_size
        if self.batch_index is None or self.batch_index != batch_index:
            self.generate_batch(batch_index)
        i = index % self.batch_size
        return self._vertices[i].clone(), self._textures[i].clone(), self._lmks[i].clone()

    def nb_faces(self):
        return self.shape_params.shape[0]

    def gen_params(self, cfg: Config):
        self.shape_params, self.expression_params, self.pose_params, self.texture_params, self.neck_pose, self.eye_pose, self.cameras, self.ambient_lights = self.__gen_params(cfg) if cfg.generator.input_folder is None else self.__load_params(cfg)
        self.ambient_lights = self.ambient_lights[:, :3] * self.ambient_lights[:, 3].view(-1, 1)
        self._faces = torch.tensor(self.flame_layer.faces.astype('int32'), device=self.device)

    def __gen_params(self, cfg: Config):
        generators = [cfg.generator.shape, cfg.generator.expression, cfg.generator.pose, cfg.generator.texture, cfg.generator.neck, cfg.generator.eye, cfg.generator.camera, cfg.generator.ambient]
        return [generator.zeros(cfg.general.nb_faces) if cfg.generator.zeros else generator.get(cfg.generator.animated.nb_frames if cfg.generator.animated.enable else cfg.general.nb_faces, gen.fixed, keyframes=gen.animation.keyframes if cfg.generator.animated.enable else None) for generator, gen in zip(self.params_generators, generators)]

    def __load_params(self, cfg: Config):
        if cfg.general.nb_faces == -1:
            cfg.general.nb_faces = sum(len(files) for _, _, files in os.walk(cfg.save.input_folder))
        pbar = trange(cfg.general.nb_faces, desc='load params', unit='visage')
        all_params = [[] for _ in range(len(self.params_generators))]
        keys = ['shape', 'expression', 'pose', 'texture', 'neck_pose', 'eye_pose', 'cameras', 'ambient']
        self.filenames = []
        for root, _, filenames in os.walk(cfg.generator.input_folder):
            filenames.sort()
            for filename in filenames:
                file = os.path.join(root, filename)
                if filename.endswith('.npy'):
                    params = np.load(file, allow_pickle=True).item()
                    for i, gen in enumerate(self.params_generators):
                        if gen is None:
                            continue
                        all_params[i].append(params[keys[i]].clone().detach().to(self.device) if keys[i] in params else gen.zeros() if cfg.generator.zeros else gen.one())
                        assert all_params[i][-1].shape[0] == gen.get_nb_params(), f'{keys[i]} params not a good, expected {gen.get_nb_params()}, but got {all_params[i][-1].shape[0]} ! (file : {filename})'
                    self.filenames.append(filename.split('.')[0])
                pbar.update(1)
                if pbar.n >= cfg.general.nb_faces:
                    break
            if pbar.n >= cfg.general.nb_faces:
                break
        pbar.close()
        for i, gen in enumerate(self.params_generators):
            if all_params[i] is not None:
                all_params[i] = torch.cat(all_params[i]).reshape(cfg.general.nb_faces, gen.get_nb_params())
        return all_params

    def generate_batch(self, batch_index: int):
        sp = self.shape_params[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        ep = self.expression_params[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        pp = self.pose_params[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        neck = self.neck_pose[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        eye = self.eye_pose[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        self._vertices, self._lmks = self.flame_layer(sp, ep, pp, neck, eye)
        self._vertices = self._vertices * self.coords_multiplier
        self._lmks = self._lmks * self.coords_multiplier

        tp = None
        if self.texture_mean is not None:
            tp = self.texture_params[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            self._textures = self.texture_mean + (self.texture_basis * tp[:, None, :]).sum(-1)
            self._textures = self._textures.reshape(tp.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
            self._textures = self._textures[:, [2, 1, 0], :, :]
            self._textures /= 255
        self._latents = {'shape': sp, 'expression': ep, 'pose': pp, 'neck_pose': neck, 'eye_pose': eye, 'texture': tp}
        self.batch_index = batch_index

    def get_latents(self, index: int):
        latents = {}
        for key, value in self._latents.items():
            latents[key] = value[index]
        return latents

    def save_batch(self, cfg: Config, leave_pbar: bool = True):
        for i in trange(len(self._vertices), desc='saving batch', unit='visage', leave=leave_pbar):
            vertices = self._vertices[i].to(self.device)
            lmk = self._lmks[i].to(self.device)
            index = self.batch_index * self.batch_size + i
            camera = self.default_camera if self.cameras is None else self.cameras[index]
            self.render.change_camera(camera)
            self.render.set_ambient_color(self.ambient_lights[index])
            texture = None if self._textures is None else self._textures[i].clone().to(self.device).mul(255).detach().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            basename = format(index, '08d') if self.filenames is None else self.filenames[index]
            if cfg.save.global_.random_bg:
                self.render.random_background()
            markers = util.read_all_index_opti_tri(vertices, self._faces, self.markers)
            face_img, depth_img = self.render.get_image(vertices, texture, vertical_flip=Saver.vertical_flip, return_depth=True)
            seg_img = self.render.get_image(vertices, None, vertical_flip=Saver.vertical_flip)
            lmks_img = self.render.get_image(vertices, texture, pts=lmk, vertical_flip=Saver.vertical_flip, dark_obj=True)
            markers_img = self.render.get_image(vertices, texture, pts=markers, vertical_flip=Saver.vertical_flip, dark_obj=True)
            args = Config({"vertices": vertices, "texture": texture, "faces": self._faces, "markers": markers, "lmks": lmk, "camera": self.render.get_camera(), "latents": self.get_latents(i), "face_img": face_img, 'seg_img': seg_img, 'depth_img': depth_img, "lmks_img": lmks_img, "markers_img": markers_img})
            for _, saver in self.savers.items(): saver(format(index // 1000, "05d") if self.face_index is None else format(self.face_index, "05d"), basename, **args)
            self.render.void_events()

    def save_all(self, cfg: Config):
        counts = cfg.generator.animated.nb_frames if cfg.generator.animated.enable else cfg.general.nb_faces
        nb_repeat = cfg.general.nb_faces if cfg.generator.animated.enable else 1
        pbar = tqdm(total=counts * nb_repeat, desc='saving all visages', unit='visage')
        for face in range(nb_repeat):
            self.face_index = face if nb_repeat > 1 else None
            self.gen_params(cfg)
            for i in range(counts // self.batch_size + (1 if counts % self.batch_size > 0 else 0)):
                self.generate_batch(i)
                self.save_batch(cfg, leave_pbar=False)
                pbar.update(self._vertices.shape[0])
        pbar.close()


@click.command()
@click.option('--cfg', default='configs/default.yml', callback=lambda ctx, param, value: Config.fromYml(value))
def main(cfg, **_):
    vg = VisageGenerator(cfg)
    vg.save_all(cfg)
    if cfg.general.view:
        vg.view(cfg)


if __name__ == "__main__":
    main()
