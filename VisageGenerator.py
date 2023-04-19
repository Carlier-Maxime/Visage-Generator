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
import trimesh

import getLandmark2D
from FLAME import FLAME
from Viewer import Viewer
from renderer import Renderer
from config import Config
from tqdm import trange
import util

class VisageGenerator():
    def __init__(self, device:str = Config.device, min_shape_param:float = Config.min_shape_param, max_shape_param:float = Config.max_shape_param,
                 min_expression_param:float = Config.min_expression_param, max_expression_param:float = Config.max_expression_param,
                 global_pose_param1:float = Config.global_pose_param1, global_pose_param2:float = Config.global_pose_param2, global_pose_param3:float = Config.global_pose_param3,
                 flame_model_path=Config.flame_model_path, batch_size=Config.batch_size, use_face_contour=Config.use_face_contour,
                 use_3D_translation=Config.use_3D_translation, shape_params=Config.shape_params, expression_params=Config.expression_params, 
                 static_landmark_embedding_path=Config.static_landmark_embedding_path, dynamic_landmark_embedding_path=Config.dynamic_landmark_embedding_path
        ):
        self.flame_layer = FLAME(flame_model_path, batch_size, use_face_contour, use_3D_translation, shape_params, expression_params, static_landmark_embedding_path, dynamic_landmark_embedding_path).to(device)
        self.render = Renderer(512, "visage.obj", 512)
        self.device = device
        self.min_shape_param = min_shape_param
        self.max_shape_param = max_shape_param
        self.min_expression_param = min_expression_param
        self.max_expression_param = max_expression_param
        self.global_pose_param1 = global_pose_param1
        self.global_pose_param2 = global_pose_param2
        self.global_pose_param3 = global_pose_param3
        self.batch_size = batch_size

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
            texture_image = texture.detach().cpu().numpy().transpose(1, 2, 0).clip(0,1)
            texture_image = (texture_image * 255).astype('uint8')
            img = PIL.Image.fromarray(texture_image)
            img.save(basename+"_texture.png")
            uvcoords = self.render.raw_uvcoords[0].cpu().numpy().reshape((-1, 2))
            uvfaces=self.render.uvfaces[0]
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
        file_obj = []
        for i in range(len(self._vertex)):
            file_obj.append('output/visage' + str(i) + '.obj')
        Viewer(self._vertex, self._landmark, self._faces, file_obj, other_objects=other_objects, device=self.device)

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

    def genParams(self, nb_faces:int = Config.nb_faces, texturing:bool = Config.texturing):
        print('Generate random parameters')
        radian = np.pi / 180.0
        shape_params = torch.tensor(np.random.uniform(self.min_shape_param, self.max_shape_param, [nb_faces, 300]),dtype=torch.float32).to(self.device)
        pose_params_numpy = np.array(
            [[self.global_pose_param1 * radian, self.global_pose_param2 * radian, self.global_pose_param3 * radian, 0.0, 0.0, 0.0]] * nb_faces,
            dtype=np.float32)
        pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).to(self.device)
        expression_params = torch.tensor(np.random.uniform(self.min_expression_param, self.max_expression_param, [nb_faces, 100]),dtype=torch.float32).to(self.device)
        if texturing: texture_params = torch.tensor(np.random.uniform(-2, 2, [nb_faces, 50])).float().to(self.device)
        else: texture_params = None
        return shape_params, pose_params, expression_params, texture_params

    def generate(self, nb_faces:int = Config.nb_faces, texturing:bool = Config.texturing, optimize_eyeballpose=Config.optimize_eyeballpose, optimize_neckpose=Config.optimize_neckpose, texture_batch_size:int=Config.texture_batch_size):
        shape_params, pose_params, expression_params, texture_params = self.genParams(nb_faces, texturing)

        self._vertex = None
        self._landmark = None
        for i in trange(nb_faces//self.batch_size+(1 if nb_faces%self.batch_size>0 else 0), desc='generate visages', unit='step'):
            sp = shape_params[i*self.batch_size:(i+1)*self.batch_size]
            ep = expression_params[i*self.batch_size:(i+1)*self.batch_size]
            pp = pose_params[i*self.batch_size:(i+1)*self.batch_size]
            neck_pose=None
            eye_pose=None
            if optimize_eyeballpose and optimize_neckpose:
                neck_pose = torch.zeros(sp.shape[0], 3).to(self.device)
                eye_pose = torch.zeros(sp.shape[0], 6).to(self.device)
            vertices, lmks = self.flame_layer(sp, ep, pp, neck_pose, eye_pose)
            if self._vertex is None: self._vertex = vertices.cpu()
            else: self._vertex = torch.cat((self._vertex, vertices.cpu()))
            if self._landmark is None: self._landmark = lmks.cpu()
            else: self._landmark = torch.cat((self._landmark, lmks.cpu()))
        self._faces = self.flame_layer.faces

        self._textures = None
        if texturing:
            tex_space = np.load("model/FLAME_texture.npz")
            texture_mean = tex_space['mean'].reshape(1, -1)
            texture_basis = tex_space['tex_dir'].reshape(-1, 200)
            texture_mean = torch.from_numpy(texture_mean).float()[None, ...].to(self.device)
            texture_basis = torch.from_numpy(texture_basis[:, :50]).float()[None, ...].to(self.device)
            for i in trange(nb_faces//texture_batch_size+(1 if nb_faces%texture_batch_size>0 else 0), desc='texturing', unit='step'):
                tp = texture_params[i*texture_batch_size:(i+1)*texture_batch_size]
                texture = texture_mean + (texture_basis * tp[:, None, :]).sum(-1)
                texture = texture.reshape(tp.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
                texture = texture[:, [2, 1, 0], :, :]
                texture = texture / 255
                if self._textures is None: self._textures = texture.cpu()
                else: self._textures = torch.cat((self._textures, texture.cpu()))


    def save(self, save_obj:bool = Config.save_obj, save_png:bool = Config.save_png, save_lmks3D_png:bool=Config.save_lmks3D_png, save_lmks2D:bool = Config.save_lmks2D, save_lmks3D_npy:bool=Config.save_lmks3D_npy, lmk2D_format:str=Config.lmk2D_format, save_markers:bool=Config.save_markers, img_resolution:list=Config.img_resolution, show_window=Config.show_window):
        out = 'output'
        tmp = 'tmp'
        outObj = (out if save_obj else tmp)+"/obj"
        outLmk3D_npy = (out if save_lmks3D_npy else tmp)+"/lmks/3D"
        outLmk2D = (out if save_lmks2D else tmp)+"/lmks/2D"
        outVisagePNG = (out if save_png else tmp)+"/png/default"
        outLmks3D_PNG = (out if save_lmks3D_png else tmp)+"/png/lmks"
        outMarkersPNG = (out if save_markers else tmp)+"/png/markers"
        for folder in [out, tmp, outObj, outLmk3D_npy, outLmk2D, outVisagePNG, outLmks3D_PNG, outMarkersPNG]: os.makedirs(folder, exist_ok=True)
        if save_lmks2D:
            lmks_paths = ""
            visage_paths = ""
            save_paths = ""
        if save_markers: markers = np.load("markers.npy")
        save_any_png = save_png or save_lmks3D_png or save_markers
        for i in trange(len(self._vertex), desc='saving', unit='visage'):
            vertices = self._vertex[i].to(self.device)
            lmk = self._landmark[i].to(self.device)
            if self._textures is None: texture=None
            else: texture = self._textures[i].to(self.device)
            basename=f"visage{str(i)}"
            visage_path = f'{outObj}/{basename}.obj'
            lmks3Dnpy_path = f'{outLmk3D_npy}/{basename}.npy'
            if save_obj or save_any_png or save_lmks2D:
                self.save_obj(visage_path, vertices, texture=texture)
            if save_lmks3D_npy or save_lmks2D:
                np.save(lmks3Dnpy_path, lmk)
            if save_lmks2D:
                if i != 0:
                    lmks_paths += ";"
                    visage_paths += ";"
                    save_paths += ";"
                lmks_paths += lmks3Dnpy_path
                visage_paths += visage_path
                save_paths += f'{outLmk2D}/{basename}.{lmk2D_format}'
            if save_any_png:
                scene = trimesh.Scene()
                scene.camera_transform = [
                    [-0.11912993, -0.59791899,  0.79265437,  0.30183245],
                    [ 0.99086974, -0.12235528,  0.05662456, -0.13695809],
                    [ 0.06312855,  0.79216291,  0.60703601,  0.29561023],
                    [ 0.,          0.,          0.,          1.        ]
                ]
                mesh = trimesh.load(visage_path)
                scene.add_geometry(mesh)
                if save_png:
                    with open(f'{outVisagePNG}/{basename}.png',"wb") as f:
                        f.write(scene.save_image(img_resolution, visible=show_window))
                if save_lmks3D_png:
                    for index,p in enumerate(lmk.cpu().numpy()):
                        sm = trimesh.primitives.Sphere(radius=0.0019, center=p)
                        sm.visual.vertex_colors = [0.2, 1., 0.2, 1.]
                        scene.add_geometry(sm, geom_name=f"sm{index}")
                    with open(f'{outLmks3D_PNG}/{basename}.png',"wb") as f:
                        f.write(scene.save_image(img_resolution, visible=show_window))
                    scene.delete_geometry([f"sm{index}" for index in range(lmk.size()[0])])
                if save_markers:
                    for im in markers:
                        m = trimesh.primitives.Sphere(radius=0.0019, center=util.read_index_opti_tri(vertices, self._faces, im))
                        m.visual.vertex_colors = [0.2, 1., 0., 1.]
                        scene.add_geometry(m)
                    with open(f'{outMarkersPNG}/{basename}.png',"wb") as f:
                        f.write(scene.save_image(img_resolution, visible=show_window))
        if save_lmks2D:
            getLandmark2D.run(visage_paths, lmks_paths, save_paths, save_png)


@click.command()
@click.option('--nb-faces', type=int, default=Config.nb_faces, help='number faces generate')
@click.option('--lmk2D-format', type=str, default=Config.lmk2D_format, help='format used for save lmk2d. (npy and pts is supported)')
@click.option('--not-texturing', 'texturing', type=bool,  default=Config.texturing,  help='enable texture', is_flag=True)
@click.option('--save-obj',  type=bool,  default=Config.save_obj,  help='enable save into file obj', is_flag=True)
@click.option('--save-png',  type=bool,  default=Config.save_png,  help='enable save into file png', is_flag=True)
@click.option('--save-lmks3D-npy', 'save_lmks3D_npy', type=bool,  default=Config.save_lmks3D_npy,  help='enable save landmarks 3D into file npy', is_flag=True)
@click.option('--save-lmks3D-png', 'save_lmks3D_png', type=bool,  default=Config.save_lmks3D_png,  help='enable save landmarks 3D with visage into file png', is_flag=True)
@click.option('--save-lmks2D', 'save_lmks2D',  type=bool,  default=Config.save_lmks2D,  help='enable save landmarks 2D into file npy', is_flag=True)
@click.option('--min-shape-param',  type=float,  default=Config.min_shape_param,  help='minimum value for shape param')
@click.option('--max-shape-param',  type=float,  default=Config.max_shape_param,  help='maximum value for shape param')
@click.option('--min-expression-param',  type=float,  default=Config.min_expression_param,  help='minimum value for expression param')
@click.option('--max-expression-param',  type=float,  default=Config.min_expression_param,  help='maximum value for expression param')
@click.option('--global-pose-param1',  type=float,  default=Config.global_pose_param1,  help='value of first global pose param')
@click.option('--global-pose-param2',  type=float,  default=Config.global_pose_param2,  help='value of second global pose param')
@click.option('--global-pose-param3',  type=float,  default=Config.global_pose_param3,  help='value of third global pose param')
@click.option('--device',  type=str,  default=Config.device,  help='choice your device for generate face. ("cpu" or "cuda")')
@click.option('--view',  type=bool,  default=Config.view,  help='enable view', is_flag=True)
@click.option('--flame-model-path', type=str, default=Config.flame_model_path, help='path for acess flame model')
@click.option('--batch-size', type=int, default=Config.batch_size, help='number of visage generate in the same time')
@click.option('--not-use-face-contour', 'use_face_contour', type=bool, default=Config.use_face_contour, is_flag=True, help='not use face contour for generate visage')
@click.option('--not-use-3D-translation', 'use_3D_translation', type=bool, default=Config.use_3D_translation, is_flag=True, help='not use 3D translation for generate visage')
@click.option('--shape-params', type=int, default=Config.shape_params, help='a number of shape parameter used')
@click.option('--expression-params', type=int, default=Config.expression_params, help='a number of expression parameter used')
@click.option('--static-landmark-embedding-path', type=str, default=Config.static_landmark_embedding_path, help='path for static landmark embedding file')
@click.option('--dynamic-landmark-embedding-path', type=str, default=Config.dynamic_landmark_embedding_path, help='path for dynamic landmark embedding file')
@click.option('--not-optimize-eyeballpose', 'optimize_eyeballpose', type=bool, default=Config.optimize_eyeballpose, is_flag=True, help='not optimize eyeballpose for generate visage')
@click.option('--not-optimize-neckpose', 'optimize_neckpose', type=bool, default=Config.optimize_neckpose, is_flag=True, help='not optimise neckpoes for generate visage')
@click.option('--texture-batch-size', type=int, default=Config.texture_batch_size, help='number of texture generate in same time')
@click.option('--save-markers', type=bool,  default=Config.save_markers,  help='enable save markers into png file', is_flag=True)
@click.option('--img-resolution', type=str, default=Config.img_resolution, help='resolution of image')
@click.option('--show-window', type=bool,  default=Config.show_window,  help='show window during save png (enable if images is the screenshot)', is_flag=True)
def main(
    nb_faces,
    lmk2d_format,
    texturing,
    save_obj,
    save_png,
    save_lmks3D_npy,
    save_lmks3D_png,
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
    optimize_neckpose,
    texture_batch_size,
    save_markers,
    img_resolution,
    show_window
):
    img_resolution = img_resolution[1:-1].split(",")
    for i in range(len(img_resolution)): img_resolution[i] = int(img_resolution[i])
    vg = VisageGenerator(device, min_shape_param, max_shape_param, min_expression_param, max_expression_param, global_pose_param1, global_pose_param2, global_pose_param3, flame_model_path, batch_size, use_face_contour, use_3D_translation, shape_params, expression_params, static_landmark_embedding_path, dynamic_landmark_embedding_path)
    vg.generate(nb_faces, texturing, optimize_eyeballpose, optimize_neckpose, texture_batch_size)
    vg.save(save_obj, save_png, save_lmks3D_png, save_lmks2D, save_lmks3D_npy, lmk2d_format, save_markers, img_resolution, show_window)
    if view:
        vg.view()

if __name__ == "__main__":
    main()
