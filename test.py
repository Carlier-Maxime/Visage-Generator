import torch
import numpy as np
import torch.nn.functional as F
from VisageGenerator import VisageGenerator
import util
from read3D import read

"""
        from renderer import Renderer
        import torch.nn.functional as F
        import torch.nn as nn
        import util2
        render = Renderer(224,"visage.obj").cpu()

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

        cam = torch.zeros(224, 3); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().cpu())
        trans_vertices = util2.batch_orth_proj(self._vertice[0], cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        render(self._vertice[0], trans_vertices, albedos)
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

"""
def meanPos(vertices):
    mean = [0,0,0]
    for v in vertices:
        mean[0] = mean[0]+v[0]
        mean[1] = mean[1]+v[1]
        mean[2] = mean[2]+v[2]
    mean[0] = mean[0]/len(vertices)
    mean[1] = mean[1]/len(vertices)
    mean[2] = mean[2]/len(vertices)
    return mean

vg = VisageGenerator(0,0,0,0,0,0,0)
vert = vg.getVertices(0)[np.load("directionnalMatrix.npy")[:,0]]
visagePos = meanPos(vert)
obj = np.load("masque.npy", allow_pickle=True)
obj[0] = np.array(obj[0])/1000

with open("../test2.txt","r") as f:
    points = []
    while True:
        p = f.readline().split(",")
        if p == ['']:
            break
        for i in range(len(p)):
            p[i] = float(p[i])
        points.append(p)

points = np.array(points)
points = points/1000

pointsPos = meanPos(points)
ecart = [visagePos[i].item()-pointsPos[i] for i in range(3)]
for i in range(3): points[:,i] = points[:,i]+ecart[i]
for i in range(3): obj[0][:,i] = obj[0][:,i]+ecart[i]

vg.view([obj,[points,[]]])
"""

import read3D
from Viewer import Viewer
masqueOV, masqueOF = read3D.readOBJ("nogit/masqueOriginelle.obj")
points = []
with open('nogit/marqueursMasqueOriginelle.txt',"r") as f:
    while True:
        line = f.readline()
        if line == "":
            break
        line = line.split(",")
        points.append([float(line[i]) for i in range(3)])
indexs = util.getIndexForMatchPoints(masqueOV,masqueOF,points)
masqueVertices, masqueFaces = read3D.readOBJ("nogit/scan_scaled.obj")
points = util.readAllIndexOptiTri(masqueVertices,masqueFaces,indexs)
visageVertices, visageFaces = read3D.readOBJ("nogit/fit_scan_result.obj")
indexs = util.getIndexForMatchPoints(visageVertices,visageFaces,points)
#np.save("balises.npy",indexs)
points2 = util.readAllIndexOptiTri(visageVertices,visageFaces,indexs)
viewer = Viewer(torch.tensor([visageVertices]),None,visageFaces,otherObjects=[[points,[]],[points2,[]]])
#[0.023051, -0.003374, -0.011894]
#[-0.024288, -0.005205, -0.00865]