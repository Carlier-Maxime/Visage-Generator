import torch
import numpy as np
import torch.nn.functional as F
import util

"""
tex_space = np.load("model/FLAME_texture.npz")
texture_mean = tex_space['mean'].reshape(1, -1)
texture_basis = tex_space['tex_dir'].reshape(-1, 200)
num_components = texture_basis.shape[1]
texture_mean = torch.from_numpy(texture_mean).float()[None,...]
texture_basis = torch.from_numpy(texture_basis[:,:50]).float()[None,...]

texcode = torch.zeros(1, 50).float().cpu()
texture = texture_mean + (texture_basis*texcode[:,None,:]).sum(-1)
texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0,3,1,2)
texture = F.interpolate(texture, [256, 256])
texture = texture[:,[2,1,0], :,:]

albedos = texture / 255
print(len(albedos[0]))
print(len(albedos[0][0]))
print(albedos[0][0][0])
"""

"""
    def on_mouse_press(self, x, y, buttons, modifiers):
        pyrender.Viewer.on_mouse_press(self,x,y,buttons,modifiers)
        camera = self._scene.main_camera_node.camera
        size = self._viewport_size
        print(camera.get_projection_matrix(640,280))
"""
"""
while len(util.loadSommets())>=2000:
    util.deleteBalises(100)
while len(util.loadSommets())>=1000:
    util.deleteBalises(25)
while len(util.loadSommets())>=500:
    util.deleteBalises(10)
while len(util.loadSommets())>150:
    util.deleteBalises(1)
"""

"""
sm = trimesh.creation.uv_sphere(radius=0.013)
sm.visual.vertex_colors = [1.0, 1.0, 0.0, 1.0]
eye = pyrender.Node("eye",mesh=pyrender.Mesh.from_trimesh(sm))
self._scene.add_node(eye)
tfs = np.tile(np.eye(4), (1, 1, 1))[0]
tfs[:3,3] = [0.108, -0.178, 0.085]
self._scene.set_pose(eye,tfs)
"""

#[0.108, -0.1155, 0.085] eye right
#[0.108, -0.178, 0.085] eye left
#[0.143, -0.1464, 0.055] noise