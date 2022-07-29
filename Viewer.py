from os.path import exists

import numpy as np
import pyrender
import torch
import trimesh

from config import get_config
from util import read_index_opti_tri


class Viewer(pyrender.Viewer):
    def __init__(self, vertex: list, landmark: list, faces: list, file_obj_for_color: list = None,
                 show_joints: bool = False, show_vertices: bool = False, show_markers: bool = True,
                 other_objects: list = None):
        """
        Args:
            vertex (list): array of all vertex
            landmark (list): array of all landmark
            faces (list): array of all faces
            file_obj_for_color (list): array of all path file for 3D object color
            show_joints (bool): display joints (landmark)
            show_vertices (bool): display vertices
            show_markers (bool): display markers
            other_objects (list): list of all others objects. one object represented by : [vertices, faces]
                but the triangles may be empty : []
        """
        config = get_config()
        self._nbFace = config.number_faces
        self._device = config.device

        self._vertex = vertex
        self._landmark = landmark
        self._faces = faces
        self._fileObjForColor = file_obj_for_color
        self._show_vertices = False
        self._show_joints = False
        self._show_markers = False
        self._edit_markers = False
        self._ctrl = False
        self._directionalMatrix = []
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
        self._markersIndex = self.load_marker()
        self._visage = pyrender.Node()
        self.set_visage(0)
        self._scene.add_node(self._visage)

        self._verticesNode, self._tfs_vertices = self.gen_vertices_node()
        if landmark is not None:
            self._jointsNode, self._tfs_joints = self.gen_joints_node()
        self._markersNode, self._tfs_markers = self.gen_markers_node()

        # select marker node
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0, 1.0]
        self._selectNode = pyrender.Node("select", mesh=pyrender.Mesh.from_trimesh(sm))

        pyrender.Viewer.__init__(self, self._scene, run_in_thread=True)

        if show_joints:
            self.show_joints()
        if show_vertices:
            self.show_vertices()
        if show_markers:
            self.show_markers()

    def on_key_press(self, symbol: int, modifiers) -> None:
        """
        This function call when key is pressed.
        Args:
            symbol (int): int value for key pressed
            modifiers:

        Returns: None
        """
        if self._ctrl:  # Ctrl On
            if symbol == 115:  # save markers (S)
                self.save_marker()
            if symbol == 108:  # load markers (L)
                self.load_marker()
        else:
            pyrender.Viewer.on_key_press(self, symbol, modifiers)

        if symbol == 118:  # show vertices (V)
            self.show_vertices()
        if symbol == 98:  # show markers (B)
            self.show_markers()
        if symbol == 106:  # show joints (J)
            self.show_joints()
        if symbol == 101:  # edit markers (E)
            self.edit_markers()
        if symbol == 65507:  # Ctrl
            self._ctrl = not self._ctrl
            if self._ctrl:
                self._message_text = "Ctrl On"
            else:
                self._message_text = "Ctrl Off"

        if self._edit_markers:
            if symbol == 65535:  # Delete
                self.remove_marker()
            if symbol == 65293:  # Enter
                self.add_marker()
            if symbol == 65362:  # Up Arrow
                self.next_marker(6)
            if symbol == 65364:  # Down Arrow
                self.next_marker(5)
            if symbol == 65361:  # Left Arrow
                self.next_marker(3)
            if symbol == 65363:  # Right Arrow
                self.next_marker(4)
            if symbol == 65365:  # Up Page
                self.next_marker(2)
            if symbol == 65366:  # Down Page
                self.next_marker(1)

    def show_vertices(self) -> None:
        """
        enable / disable display vertices
        Returns: None
        """
        self.render_lock.acquire()
        if not self._show_vertices:
            self._scene.add_node(self._verticesNode)
            self._show_vertices = True
        else:
            self._scene.remove_node(self._verticesNode)
            self._show_vertices = False
        self.render_lock.release()

    def show_joints(self) -> None:
        """
        enable / disable display joints
        Returns: None
        """
        self.render_lock.acquire()
        if not self._show_joints:
            self._scene.add_node(self._jointsNode)
            self._show_joints = True
        else:
            self._scene.remove_node(self._jointsNode)
            self._show_joints = False
        self.render_lock.release()

    def show_markers(self) -> None:
        """
        enable / disable display markers
        Returns: None
        """
        self.render_lock.acquire()
        if len(self._tfs_markers) <= 0:
            self._show_markers = False
            self.render_lock.release()
            return
        if not self._show_markers:
            self._scene.add_node(self._markersNode)
            self._show_markers = True
        else:
            self._scene.remove_node(self._markersNode)
            self._show_markers = False
        self.render_lock.release()

    def set_visage(self, i) -> None:
        """
        Switch visage to the visage provided by index
        Args:
            i: index visage

        Returns: None
        """
        if self._fileObjForColor is not None:
            mesh = pyrender.Mesh.from_trimesh(trimesh.load(self._fileObjForColor[self._index]))
        else:
            vertices = self._vertex[i].detach().to(self._device).numpy().squeeze()
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.925, 0.72, 0.519, 1.0]
            tri_mesh = trimesh.Trimesh(vertices, self._faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self._visage = pyrender.Node("Visage", mesh=mesh)

    def on_close(self) -> None:
        """
        This function call when the app is closed.
        Switch to the next visage or close
        Returns: None
        """
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

    def edit_markers(self) -> None:
        """
        Enable / disable edit markers
        Returns: None
        """
        if not self._edit_markers:
            if len(self._directionalMatrix) == 0:
                self._directionalMatrix = np.load("directionalMatrix.npy")
            vert = self._vertex[self._index][self._directionalMatrix[self._slcIndex][0]]
            tfs = np.tile(np.eye(4), (1, 1, 1))[0]
            tfs[:3, 3] = vert
            self._scene.add_node(self._selectNode)
            self._scene.set_pose(self._selectNode, tfs)
            if not self._show_markers:
                self.show_markers()
            self._edit_markers = True
            self._message_text = 'Enable edit markers'
        else:
            self._scene.remove_node(self._selectNode)
            self._edit_markers = False
            self._message_text = 'Disable edit markers'

    def next_marker(self, direction: int) -> None:
        """
        Switch to the next marker in the direction provided
        Args:
            direction (int): index direction

        Returns: None
        """
        i = self._directionalMatrix[self._slcIndex][direction]
        if i == -1:
            return
        self._slcIndex = i
        vert = self._vertex[self._index][self._directionalMatrix[self._slcIndex][0]]
        tfs = np.tile(np.eye(4), (1, 1, 1))[0]
        tfs[:3, 3] = vert
        self._scene.set_pose(self._selectNode, tfs)

    def update_tfs(self) -> None:
        """
        Update all position for all points based on face
        Returns: None
        """
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
        if self._show_markers and len(self._tfs_markers) > 0:
            self._scene.remove_node(self._markersNode)
            self.gen_markers_node()
            self._scene.add_node(self._markersNode)
        else:
            self.gen_markers_node()
        if self._edit_markers:
            tfs = self._tfs_vertices[self._slcIndex]
            self._scene.set_pose(self._selectNode, tfs)

    def gen_vertices_node(self):
        """
        Generate vertices Node and return result and tfs
        Returns: vertices node, tfs
        """
        vertices = self._vertex[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.0019)
        sm.visual.vertex_colors = [0.7, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
        tfs[:, :3, 3] = vertices
        vertices_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        return pyrender.Node("vertices", mesh=vertices_pcl), tfs

    def gen_joints_node(self):
        """
        Generate joints Node and return result and tfs
        Returns: joints node, tfs
        """
        joints = self._landmark[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.0019)
        sm.visual.vertex_colors = [0.0, 0.5, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        return pyrender.Node("joints", mesh=joints_pcl), tfs

    def gen_markers_node(self):
        """
        Generate markers Node and return result and tfs
        Returns: markers node, tfs
        """
        sm = trimesh.creation.uv_sphere(radius=0.0005)
        sm.visual.vertex_colors = [0.8, 0.0, 0.5, 1.0]
        t = []
        for marker in self._markersIndex:
            t.append(read_index_opti_tri(self._vertex[self._index], self._faces, marker))
        if len(t) > 0:
            tfs_markers = np.tile(np.eye(4), (len(t), 1, 1))
            for i in range(len(t)):
                tfs_markers[i, :3, 3] = t[i]
            markers_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs_markers)
        else:
            tfs_markers = []
            markers_pcl = pyrender.Mesh.from_trimesh(sm)
        return pyrender.Node("markers", mesh=markers_pcl), tfs_markers

    def add_marker(self) -> None:
        """
        Add marker into all markers.
        The marker added is the selected marker
        Returns: None
        """
        self._markersIndex = np.append(self._markersIndex, [[self._directionalMatrix[self._slcIndex][0], -1, 0, 0]],
                                       axis=0)
        self.update_marker()

    def save_marker(self) -> None:
        """
        Save all markers in the numpy file.
        The program will keep these markers even at the next launch
        Returns: None
        """
        np.save("markers.npy", self._markersIndex)

    def load_marker(self):
        """
        Load marker file (markers.npy), if file not exists return the empty numpy array
        Returns: array of all markers
        """
        self._markersIndex = np.array([], int)
        if not exists("markers.npy"):
            return
        return np.load("markers.npy")

    def remove_marker(self) -> None:
        """
        Remove selected marker in the list of all markers
        Returns: None
        """
        ind = self._directionalMatrix[self._slcIndex][0]
        for i in range(len(self._markersIndex)):
            if self._markersIndex[i][0] == ind:
                self._markersIndex = np.delete(self._markersIndex, [i * 4 + j for j in range(4)])
                self._markersIndex = self._markersIndex.reshape(int(len(self._markersIndex) / 4), 4)
                break
        self.update_marker()

    def update_marker(self) -> None:
        """
        Update marker view and print number of markers
        Returns: None
        """
        if self._show_markers:
            self._scene.remove_node(self._markersNode)
        self.gen_markers_node()
        if not self._show_markers:
            self.show_markers()
        else:
            self._scene.add_node(self._markersNode)
        self._message_text = "Nb markers = " + str(len(self._markersIndex))
