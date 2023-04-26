from os.path import exists

import numpy as np
import torch
import pygame
from pygame.constants import *
from OpenGL.GL import glDeleteLists

from util import read_all_index_opti_tri
from renderer import Renderer

class Viewer(Renderer):
    def __init__(self, vertex: list, textures, landmark: list, faces: list, show_joints: bool = False, show_vertices: bool = False, show_markers: bool = True, other_objects: list = None, device="cuda"):
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
        Renderer.__init__(self, 1024, 1024, device)
        self._device = device
        self._vertex = vertex
        self._textures = textures
        self._landmark = landmark
        self._faces = faces
        self._show_vertices = show_vertices
        self._show_joints = show_joints
        self._show_markers = show_markers
        self._edit_markers = False
        self._ctrl = False
        self._directionalMatrix = []
        self._joints_glList = self._vertices_glList = self._markers_glList = self._select_glList = None

        self.oobj_gl_list = []
        if other_objects is not None:
            for obj in other_objects:
                vertices = obj[0]
                triangles = obj[1]
                vertices = torch.tensor(vertices, device=device)
                triangles = torch.tensor(triangles, device=device)
                if len(triangles) == 0:
                    self.oobj_gl_list.append(self.create_spheres_gl_list(vertices))
                else:
                    self._create_GL_List(vertices, triangles)

        self._index = 0
        self._slcIndex = 0
        self._markersIndex = self.load_marker()
        self.large_raw_sphere = self.create_sphere(0.03, 30, 30)
        self.small_raw_sphere = self.create_sphere(0.01, 30, 30)
        self.set_visage(self._index)
        self.loop()

    def __del__(self):
        if self._joints_glList is not None: glDeleteLists(self._joints_glList, 1)
        if self._markers_glList is not None: glDeleteLists(self._markers_glList, 1)
        if self._vertices_glList is not None: glDeleteLists(self._vertices_glList, 1)
        if self._select_glList is not None: glDeleteLists(self._select_glList, 1)
        Renderer.__del__(self)

    def loop(self):
        clock = pygame.time.Clock()
        while 1:
            clock.tick(60)
            lists = [self.gl_list_visage]
            if self._show_joints and self._landmark is not None: lists.append(self._joints_glList)
            if self._show_markers: lists.append(self._markers_glList)
            if self._show_vertices: lists.append(self._vertices_glList)
            if self._edit_markers and self._select_glList is not None: lists.append(self._select_glList)
            self._render(lists)
            if not self._poll_events(): break

    def _poll_event(self, e):
        if not Renderer._poll_event(self, e): return 0
        if e.type != KEYDOWN: return 1
        if e.key == K_v: self._show_vertices = not self._show_vertices
        if e.key == K_b: self._show_markers = not self._show_markers
        if e.key == K_j: self._show_joints = not self._show_joints
        if e.key == K_e: self.edit_markers()
        if self._edit_markers:
            if e.key == K_DELETE: self.remove_marker()
            if e.key == K_KP_ENTER: self.add_marker()
            if e.key == K_UP: self.next_marker(6)
            if e.key == K_DOWN: self.next_marker(5)
            if e.key == K_LEFT: self.next_marker(4)
            if e.key == K_RIGHT: self.next_marker(3)
            if e.key == K_PAGEUP: self.next_marker(2)
            if e.key == K_PAGEDOWN: self.next_marker(1)
        else:
            if e.key == K_UP: self.set_visage(self._index+1)
            if e.key == K_DOWN: self.set_visage(self._index-1)
        if e.key == K_s: self.save_marker()
        if e.key == K_l: self.load_marker()
        return 1

    def set_visage(self, i) -> None:
        """
        Switch visage to the visage provided by index
        Args:
            i: index visage

        Returns: None
        """
        if i >= self._vertex.shape[0]: i=0
        elif i < 0: i=self._vertex.shape[0]-1
        self._index = i
        if self._textures is not None:
            texture = self._textures[self._index].to(self.device)
            texture = texture * 255
            texture = texture.detach().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        else: texture = None
        self._edit_GL_List(self._vertex[i].to(self._device), texture)
        self.update_pts()

    def edit_markers(self) -> None:
        """
        Enable / disable edit markers
        Returns: None
        """
        if not self._edit_markers:
            if len(self._directionalMatrix) == 0: self._directionalMatrix = np.load("directionalMatrix.npy")
            self._select_glList = self.gen_select_glList()
            self._show_markers = True
            self._edit_markers = True
        else:
            self._edit_markers = False

    def next_marker(self, direction: int) -> None:
        """
        Switch to the next marker in the direction provided
        Args:
            direction (int): index direction

        Returns: None
        """
        i = self._directionalMatrix[self._slcIndex][direction]
        if i == -1: return
        self._slcIndex = i
        self._select_glList = self.gen_select_glList()

    def update_pts(self) -> None:
        """
        Update all position for all points based on face
        Returns: None
        """
        self._vertices_glList = self.gen_vertices_glList()
        if self._landmark is not None: self._joints_glList = self.gen_joints_glList()
        self._markers_glList = self.gen_markers_glList()

    def gen_select_glList(self):
        vert = self._vertex[self._index][self._directionalMatrix[self._slcIndex][0]]
        vert = vert.to(self.device)[None]
        vert = vert[:, [0, 2, 1]]
        vert *= 10
        self._select_glList = Renderer.create_spheres_gl_list(self.large_raw_sphere, vert, self._select_glList, [255, 255, 0.1])
        return self._select_glList

    def gen_vertices_glList(self):
        vts = self._vertex[self._index].to(self._device)
        vts = vts[:, [0, 2, 1]]
        vts *= 10
        return Renderer.create_spheres_gl_list(self.small_raw_sphere, vts, self._vertices_glList, [255, 0.2, 0.2])

    def gen_joints_glList(self):
        lmk = self._landmark[self._index].to(self._device)
        lmk = lmk[:, [0, 2, 1]]
        lmk *= 10
        return Renderer.create_spheres_gl_list(self.raw_sphere, lmk, self._joints_glList, [0.2, 0.2, 255])

    def gen_markers_glList(self):
        mks = torch.tensor(np.array(read_all_index_opti_tri(self._vertex[self._index], self._faces, self._markersIndex)), device=self._device)
        mks = mks[:, [0, 2, 1]]
        mks *= 10
        return Renderer.create_spheres_gl_list(self.raw_sphere, mks, self._markers_glList, [0.2, 255, 0.2])

    def add_marker(self) -> None:
        """
        Add marker into all markers.
        The marker added is the selected marker
        Returns: None
        """
        self._markersIndex = np.append(self._markersIndex, [[self._directionalMatrix[self._slcIndex][0], -1, 0, 0]], axis=0)
        self._markers_glList = self.gen_markers_glList()

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
        self._markers_glList = self.gen_markers_glList()
