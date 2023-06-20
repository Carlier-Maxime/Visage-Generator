import abc
from typing import Tuple

import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from torch import Tensor
from pygame.constants import *


def get_intrinsic_matrix(self):
    focal_length = (self.width / (2 * torch.pi)) / torch.tan(torch.deg2rad(self.fov / 2.0))

    # Calcul de la matrice intrinsèque normalisé
    intrinsic_matrix = torch.tensor([
        [focal_length / self.width, 0, 0.5],
        [0, focal_length / self.height, 0.5],
        [0, 0, 1]
    ], dtype=torch.float32, device=self.fov.device)
    return intrinsic_matrix


class BaseCamera:
    def __init__(self, camera: torch.Tensor, width: int, height: int):
        self.width = self.height = None
        self.set_size(width, height)

    @abc.abstractmethod
    def _update(self) -> None:
        pass

    @abc.abstractmethod
    def get_matrix(self) -> tuple[Tensor, Tensor]:
        pass

    @abc.abstractmethod
    def set_camera(self, camera: torch.Tensor) -> None:
        pass

    def set_size(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def poll_event(self, e):
        pass


class DefaultCamera(BaseCamera):

    def __init__(self, camera: torch.Tensor, width, height):
        super().__init__(camera, width, height)
        self.fov = self.tx = self.ty = self.tz = self.rx = self.ry = self.rz = None
        self.rotate = self.rotate_z = self.move = False
        self.set_camera(camera)

    def _update(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.width / float(self.height), 0.1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslate(self.tx, self.ty, self.tz)
        glRotate(self.rx, 1, 0, 0)
        glRotate(self.ry, 0, 1, 0)
        glRotate(self.rz, 0, 0, 1)

    def get_matrix(self) -> tuple[Tensor, Tensor]:
        intrinsic_matrix = get_intrinsic_matrix(self)

        extrinsic_matrix = torch.tensor(glGetFloatv(GL_MODELVIEW_MATRIX), device=self.fov.device)
        extrinsic_matrix[:, 3] = extrinsic_matrix[3]
        extrinsic_matrix[3] = torch.tensor([0, 0, 0, 1], device=self.fov.device)

        return intrinsic_matrix, extrinsic_matrix

    def set_camera(self, camera: torch.Tensor) -> None:
        if camera is None:
            return
        self.fov, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz = camera
        self._update()

    def __str__(self):
        return f'fov: {self.fov}, tx: {self.tx}, ty: {self.ty}, tz: {self.tz}, rx: {self.rx}, ry: {self.ry}, rz: {self.rz}'

    def poll_event(self, e):
        if e.type == MOUSEBUTTONDOWN:
            if e.button == 4:
                self.tz = max(1, self.tz - 0.1)
                self._update()
            elif e.button == 5:
                self.tz += 0.1
                self._update()
            elif e.button == 1:
                self.rotate = True
            elif e.button == 3:
                self.move = True
            elif e.button == 2:
                self.rotate_z = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1:
                self.rotate = False
            elif e.button == 3:
                self.move = False
            elif e.button == 2:
                self.rotate_z = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if self.rotate:
                self.rx += j
                self.ry += i
                self._update()
            if self.rotate_z:
                self.rz += i
                self._update()
            if self.move:
                self.tx += i / 256
                self.ty -= j / 256
                self._update()
        elif e.type == KEYDOWN:
            if e.key == K_c:
                print(self)
            if e.key == K_KP_MINUS:
                self.fov -= 0.1
                self._update()
            if e.key == K_KP_PLUS:
                self.fov += 0.1
                self._update()


class VectorCamera(BaseCamera):
    def __init__(self, camera: torch.Tensor, width, height):
        super().__init__(camera, width, height)
        self.fov = self.lookAt = self.radius = self.theta = self.phi = self.origin = self.forward_vector = self.up_vector = self.right_vector = None
        self.rotate = self.move_z = self.move = False
        self.set_camera(camera)

    def _update(self) -> None:
        if self.phi == 0:
            self.phi += 1e-5
        elif self.phi == torch.pi:
            self.phi -= 1e-5
        self.origin = torch.zeros(3, device=self.fov.device)
        self.origin[0] = self.radius * torch.sin(self.phi) * torch.cos(torch.pi - self.theta)
        self.origin[2] = self.radius * torch.sin(self.phi) * torch.sin(torch.pi - self.theta)
        self.origin[1] = self.radius * torch.cos(self.phi)
        self.forward_vector = self.lookAt - self.origin
        self.forward_vector = torch.nn.functional.normalize(self.forward_vector, p=2, dim=0)
        self.up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.origin.device)
        self.right_vector = -torch.nn.functional.normalize(torch.cross(self.up_vector, self.forward_vector, dim=-1), p=2, dim=0)
        self.up_vector = torch.nn.functional.normalize(torch.cross(self.forward_vector, self.right_vector, dim=-1), p=2, dim=0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.width / float(self.height), 0.1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        ox, oy, oz = self.origin
        ux, uy, uz = -self.up_vector
        ax, ay, az = self.lookAt
        gluLookAt(ox, oy, oz, ax, ay, az, ux, uy, uz)

    def get_matrix(self) -> tuple[Tensor, Tensor]:
        intrinsic_matrix = get_intrinsic_matrix(self)
        matrix = torch.eye(4, device=self.origin.device)
        matrix[:3, :3] = torch.stack((self.right_vector, self.up_vector, self.forward_vector), dim=-1)
        matrix[:3, 3] = self.origin
        return intrinsic_matrix, matrix

    def set_camera(self, camera: torch.Tensor) -> None:
        if camera is None:
            return
        self.fov, self.radius, self.phi, self.theta = camera[[0, 4, 5, 6]]
        self.lookAt = camera[1:4]
        self.theta = torch.deg2rad(self.theta)
        self.phi = torch.deg2rad(self.phi)
        self._update()

    def __str__(self):
        return f'fov: {self.fov}, lookAt : {self.lookAt}, radius : {self.radius}, phi : {self.phi}, theta : {self.theta}'

    def poll_event(self, e):
        if e.type == MOUSEBUTTONDOWN:
            if e.button == 4:
                self.radius = self.radius - 0.1
                self._update()
            elif e.button == 5:
                self.radius += 0.1
                self._update()
            elif e.button == 1:
                self.rotate = True
            elif e.button == 3:
                self.move = True
            elif e.button == 2:
                self.move_z = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1:
                self.rotate = False
            elif e.button == 3:
                self.move = False
            elif e.button == 2:
                self.move_z = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if self.rotate:
                self.phi += j/100
                self.theta -= i/100
                self._update()
            if self.move_z:
                self.lookAt[2] += i
                self._update()
            if self.move:
                self.lookAt[0] -= i / 256
                self.lookAt[1] += j / 256
                self._update()
        elif e.type == KEYDOWN:
            if e.key == K_c:
                print(self)
            if e.key == K_KP_MINUS:
                self.fov -= 0.1
                self._update()
            if e.key == K_KP_PLUS:
                self.fov += 0.1
                self._update()