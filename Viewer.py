import pyrender
import numpy as np
import trimesh
from config import nbFace,device

class Viewer(pyrender.Viewer):
    def __init__(self, vertice, landmark, faces, show_joints=False, show_vertices=False, show_balises = True):
        self._vertice = vertice
        self._landmark = landmark
        self._faces = faces
        self._show_vertices = False
        self._show_joints = False
        self._show_balises = False
        self._editBalises = False
        self._scene = pyrender.Scene()
        self.setVisage(0)
        self._scene.add_node(self._visage)
        self._index = 0

        #vertices node
        vertices = self._vertice[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
        tfs[:, :3, 3] = vertices
        vertices_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self._verticesNode = pyrender.Node("vertices",mesh=vertices_pcl)
        self._tfs_vertices = tfs

        #joints (landmark) node
        joints = self._landmark[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [0.0, 0.5, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self._jointsNode = pyrender.Node("joints",mesh=joints_pcl)

        #balises node
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [1.0, 0.0, 1.0, 1.0]
        self._tfs_balises = []
        balises_pcl = pyrender.Mesh.from_trimesh(sm)
        self._balisesNode = pyrender.Node("balises",mesh=balises_pcl)

        #select balise node
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0, 1.0]
        self._selectNode = pyrender.Node("select",mesh=pyrender.Mesh.from_trimesh(sm))

        pyrender.Viewer.__init__(self, self._scene, use_raymond_lighting=True, run_in_thread=True)

        if show_joints:
            self.showJoints()
        if show_vertices:
            self.showVertices()
        if show_balises:
            self.showBalises()      

    def on_key_press(self, symbol, modifiers):
        pyrender.Viewer.on_key_press(self,symbol,modifiers)
        if symbol == 118: # show vertices
            self.showVertices()
        if symbol == 98: # show balises
            self.showBalises()
        if symbol == 106: # show joints
            self.showJoints()
        if symbol == 101: # edit balises
            self.editBalises()

        if self._editBalises:
            if symbol == 65362: # Up Arrow
                pass
            if symbol == 65364: # Down Arrow
                pass
            if symbol == 65361: # Left Arrow
                pass
            if symbol == 65363: # Right Arrow
                pass

    def showVertices(self):
        self.render_lock.acquire()
        if not self._show_vertices:
            self._scene.add_node(self._verticesNode)
            self._show_vertices = True
        else:
            self._scene.remove_node(self._verticesNode)
            self._show_vertices = False
        self.render_lock.release()

    def showJoints(self):
        self.render_lock.acquire()
        if not self._show_joints:
            self._scene.add_node(self._jointsNode)
            self._show_joints = True
        else:
            self._scene.remove_node(self._jointsNode)
            self._show_joints = False
        self.render_lock.release()

    def showBalises(self):
        self.render_lock.acquire()
        if not self._show_balises:
            if (len(self._tfs_balises)>0):
                self._scene.add_node(self._balisesNode)
            self._show_balises = True
        elif len(self._tfs_balises)>0:
            self._scene.remove_node(self._balisesNode)
            self._show_balises = False
        self.render_lock.release()

    def setVisage(self,i):
        vertices = self._vertice[i].detach().to(device).numpy().squeeze()
        joints = self._landmark[i].detach().to(device).numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.925, 0.72, 0.519, 1.0]

        tri_mesh = trimesh.Trimesh(vertices, self._faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self._visage = pyrender.Node("Visage",mesh=mesh)

    def on_close(self):
        if (self._index<nbFace-1):
            self._index = self._index+1
            self.render_lock.acquire()
            self._scene.remove_node(self._visage)
            self.setVisage(self._index)
            self._scene.add_node(self._visage)
            self.render_lock.release()
        else:
            pyrender.Viewer.on_close(self)
    
    def editBalises(self):
        if not self._editBalises:
            self._editBalises = True
            tfs, self._tfs_vertices = self._tfs_vertices[-1], self._tfs_vertices[:-1]
            self._scene.add_node(self._selectNode)
            self._scene.set_pose(self._selectNode,tfs)