import torch
import numpy as np
import torch.nn.functional as F
import util

""" generate sommets.txt
t=[]
for i in range(len(vertices)):
    t.append([i,vertices[i]])
for i in range(len(t)):
    point = t[i][1]
    distance = 0
    index = -1
    for j in range(len(t)):
        if i==j:
            continue
        p = t[j][1]
        dist = np.sqrt((p[0]-point[0])**2+(p[1]-point[1])**2+(p[2]-point[2])**2)
        if index==-1 or dist<distance:
            distance = dist
            index = t[j][0]
    t[i].append(index)
    t[i].append(distance)
    print(str(i+1)+"/"+str(len(t)))
t.sort(key=lambda t:t[3])

with open("sommets.txt","w") as f:
    f.write("[\n")
    for e in t:
        f.write("["+str(e[0])+", ["+str(e[1][0])+", "+str(e[1][1])+", "+str(e[1][2])+"], "+str(e[2])+", "+str(e[3])+"]\n")
    f.write("]")
"""
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

a = [0.456,0.345,0.008]
b = [0.234,0.344,0.156]

x = b[0]-a[0]
y = b[1]-a[1]
z = b[2]-a[2]

print(x,y,z)