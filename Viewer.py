from os.path import exists

import numpy as np
import pyrender
import torch
import trimesh

from config import get_config
from util import read_index_opti_tri


class Viewer(pyrender.Viewer):
    def __init__(self, vertex, landmark, faces, file_obj_for_color=None, show_joints=False, show_vertices=False,
                 show_balises=True, other_objects=None):
        config = get_config()
        self._nbFace = config.number_faces
        self._device = config.device

        self._vertex = vertex
        self._landmark = landmark
        self._faces = faces
        self._fileObjForColor = file_obj_for_color
        self._show_vertices = False
        self._show_joints = False
        self._show_balises = False
        self._editBalises = False
        self._ctrl = False
        self._directionnalMatrix = []
        self._scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0, 1.0])

        if other_objects is not None:
            for obj in other_objects:
                vertices = obj[0]
                triangles = obj[1]
                vertices = torch.tensor(vertices).detach().cpu().numpy().squeeze()
                triangles = np.array(triangles)
                if len(triangles) == 0:
                    sm = trimesh.creation.uv_sphere(radius=0.001)
                    sm.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
                    tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
                    tfs[:, :3, 3] = vertices
                    mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    self._scene.add(mesh)
                else:
                    tri_mesh = trimesh.Trimesh(vertices, faces=triangles)
                    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
                    self._scene.add(mesh)

        self._index = 0
        self._slcIndex = 0
        self._balisesIndex = self.load_balise()
        self._visage = pyrender.Node()
        self.set_visage(0)
        self._scene.add_node(self._visage)

        self._verticesNode, self._tfs_vertices = self.gen_vertices_node()
        if landmark is not None:
            self._jointsNode, self._tfs_joints = self.gen_joints_node()
        self._balisesNode, self._tfs_balises = self.gen_balises_node()

        # select balise node
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0, 1.0]
        self._selectNode = pyrender.Node("select", mesh=pyrender.Mesh.from_trimesh(sm))

        pyrender.Viewer.__init__(self, self._scene, run_in_thread=True)

        if show_joints:
            self.show_joints()
        if show_vertices:
            self.show_vertices()
        if show_balises:
            self.show_balises()

    def on_key_press(self, symbol, modifiers):
        if self._ctrl:  # Ctrl On
            if symbol == 115:  # save balises (S)
                self.save_balise()
            if symbol == 108:  # load balises (L)
                self.load_balise()
        else:
            pyrender.Viewer.on_key_press(self, symbol, modifiers)

        if symbol == 118:  # show vertices (V)
            self.show_vertices()
        if symbol == 98:  # show balises (B)
            self.show_balises()
        if symbol == 106:  # show joints (J)
            self.show_joints()
        if symbol == 101:  # edit balises (E)
            self.edit_balises()
        if symbol == 65507:  # Ctrl
            self._ctrl = not self._ctrl
            if self._ctrl:
                self._message_text = "Ctrl On"
            else:
                self._message_text = "Ctrl Off"

        if self._editBalises:
            if symbol == 65535:  # Suppr
                self.remove_balise()
            if symbol == 65293:  # Enter
                self.add_balise()
            if symbol == 65362:  # Up Arrow
                self.next_balise(6)
            if symbol == 65364:  # Down Arrow
                self.next_balise(5)
            if symbol == 65361:  # Left Arrow
                self.next_balise(3)
            if symbol == 65363:  # Right Arrow
                self.next_balise(4)
            if symbol == 65365:  # Up Page
                self.next_balise(2)
            if symbol == 65366:  # Down Page
                self.next_balise(1)

    def show_vertices(self):
        self.render_lock.acquire()
        if not self._show_vertices:
            self._scene.add_node(self._verticesNode)
            self._show_vertices = True
        else:
            self._scene.remove_node(self._verticesNode)
            self._show_vertices = False
        self.render_lock.release()

    def show_joints(self):
        self.render_lock.acquire()
        if not self._show_joints:
            self._scene.add_node(self._jointsNode)
            self._show_joints = True
        else:
            self._scene.remove_node(self._jointsNode)
            self._show_joints = False
        self.render_lock.release()

    def show_balises(self):
        self.render_lock.acquire()
        if len(self._tfs_balises) <= 0:
            self._show_balises = False
            self.render_lock.release()
            return
        if not self._show_balises:
            self._scene.add_node(self._balisesNode)
            self._show_balises = True
        else:
            self._scene.remove_node(self._balisesNode)
            self._show_balises = False
        self.render_lock.release()

    def set_visage(self, i):
        if self._fileObjForColor is not None:
            mesh = pyrender.Mesh.from_trimesh(trimesh.load(self._fileObjForColor[self._index]))
        else:
            vertices = self._vertex[i].detach().to(self._device).numpy().squeeze()
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.925, 0.72, 0.519, 1.0]
            tri_mesh = trimesh.Trimesh(vertices, self._faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self._visage = pyrender.Node("Visage", mesh=mesh)

    def on_close(self):
        if self._index < self._nbFace - 1:
            self._index = self._index + 1
            self.render_lock.acquire()
            self._scene.remove_node(self._visage)
            self.set_visage(self._index)
            self._scene.add_node(self._visage)
            self.update_tfs()
            self.render_lock.release()
        else:
            pyrender.Viewer.on_close(self)

    def edit_balises(self):
        if not self._editBalises:
            if len(self._directionnalMatrix) == 0:
                self._directionnalMatrix = np.load("directionnalMatrix.npy")
            vert = self._vertex[self._index][self._directionnalMatrix[self._slcIndex][0]]
            tfs = np.tile(np.eye(4), (1, 1, 1))[0]
            tfs[:3, 3] = vert
            self._scene.add_node(self._selectNode)
            self._scene.set_pose(self._selectNode, tfs)
            if not self._show_balises:
                self.show_balises()
            self._editBalises = True
            self._message_text = 'Enable edit balises'
        else:
            self._scene.remove_node(self._selectNode)
            self._editBalises = False
            self._message_text = 'Disable edit balises'

    def next_balise(self, direction):
        i = self._directionnalMatrix[self._slcIndex][direction]
        if i == -1:
            return
        self._slcIndex = i
        vert = self._vertex[self._index][self._directionnalMatrix[self._slcIndex][0]]
        tfs = np.tile(np.eye(4), (1, 1, 1))[0]
        tfs[:3, 3] = vert
        self._scene.set_pose(self._selectNode, tfs)

    def update_tfs(self):
        if self._show_vertices:
            self._scene.remove_node(self._verticesNode)
            self._verticesNode, self._tfs_vertices = self.gen_vertices_node()
            self._scene.add_node(self._verticesNode)
        else:
            self._verticesNode, self._tfs_vertices = self.gen_vertices_node()
        if self._show_joints:
            self._scene.remove_node(self._jointsNode)
            self._jointsNode, self._tfs_joints = self.gen_joints_node()
            self._scene.add_node(self._jointsNode)
        else:
            self._jointsNode, self._tfs_joints = self.gen_joints_node()
        if self._show_balises and len(self._tfs_balises) > 0:
            self._scene.remove_node(self._balisesNode)
            self.gen_balises_node()
            self._scene.add_node(self._balisesNode)
        else:
            self.gen_balises_node()
        if self._editBalises:
            tfs = self._tfs_vertices[self._slcIndex]
            self._scene.set_pose(self._selectNode, tfs)

    def gen_vertices_node(self):
        vertices = self._vertex[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.0019)
        sm.visual.vertex_colors = [0.7, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
        tfs[:, :3, 3] = vertices
        vertices_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        return pyrender.Node("vertices", mesh=vertices_pcl), tfs

    def gen_joints_node(self):
        joints = self._landmark[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.0019)
        sm.visual.vertex_colors = [0.0, 0.5, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        return pyrender.Node("joints", mesh=joints_pcl), tfs

    def gen_balises_node(self):
        sm = trimesh.creation.uv_sphere(radius=0.0005)
        sm.visual.vertex_colors = [0.8, 0.0, 0.5, 1.0]
        t = []
        for balise in self._balisesIndex:
            t.append(read_index_opti_tri(self._vertex[self._index], self._faces, balise))
        if len(t) > 0:
            tfs_balises = np.tile(np.eye(4), (len(t), 1, 1))
            for i in range(len(t)):
                tfs_balises[i, :3, 3] = t[i]
            balises_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs_balises)
        else:
            tfs_balises = []
            balises_pcl = pyrender.Mesh.from_trimesh(sm)
        return pyrender.Node("balises", mesh=balises_pcl), tfs_balises

    def add_balise(self):
        self._balisesIndex = np.append(self._balisesIndex, [[self._directionnalMatrix[self._slcIndex][0], -1, 0, 0]],
                                       axis=0)
        self.update_balise()

    def save_balise(self):
        np.save("balises.npy", self._balisesIndex)

    def load_balise(self):
        self._balisesIndex = np.array([], int)
        if not exists("balises.npy"):
            return
        return np.load("balises.npy")

    def remove_balise(self):
        ind = self._directionnalMatrix[self._slcIndex][0]
        for i in range(len(self._balisesIndex)):
            if self._balisesIndex[i][0] == ind:
                self._balisesIndex = np.delete(self._balisesIndex, [i * 4 + j for j in range(4)])
                self._balisesIndex = self._balisesIndex.reshape(int(len(self._balisesIndex) / 4), 4)
                break
        self.update_balise()

    def update_balise(self):
        if self._show_balises:
            self._scene.remove_node(self._balisesNode)
        self.gen_balises_node()
        if not self._show_balises:
            self.show_balises()
        else:
            self._scene.add_node(self._balisesNode)
        self._message_text = "NbBalises = " + str(len(self._balisesIndex))
