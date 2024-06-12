import abc

import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.constants import *
from torch import Tensor


def get_intrinsic_matrix(self):
    focal_length = 1 / (torch.tan(torch.deg2rad(self.fov)) * 1.414)

    intrinsic_matrix = torch.tensor([
        [focal_length, 0, 0.5],
        [0, focal_length, 0.5],
        [0, 0, 1]
    ], dtype=torch.float32, device=self.fov.device)
    return intrinsic_matrix


class BaseCamera:
    def __init__(self, camera: torch.Tensor, width: int, height: int):
        self.aspect_ratio = None
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
        self.aspect_ratio = self.width / float(self.height)

    def poll_event(self, e):
        pass

    @abc.abstractmethod
    def to_tensor(self):
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
        gluPerspective(self.fov, self.aspect_ratio, 0.1, 100.0)
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

    def to_tensor(self):
        return torch.tensor([self.fov, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz], device=self.fov.device)


class VectorCamera(BaseCamera):
    def __init__(self, camera: torch.Tensor, width, height):
        super().__init__(camera, width, height)
        self.fov = self.lookAt = self.radius = self.theta = self.phi = self.eyePoint = self.forward_vector = self.up_vector = self.right_vector = None
        self.rotate = self.move_z = self.move = False
        self._default_up_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=camera.device)
        self.set_camera(camera)

    def _update(self) -> None:
        self.phi += 1e-5 if self.phi == 0 else -1e-5 if self.phi == torch.pi else 0
        sin_phi = torch.sin(self.phi)
        cos_theta = torch.cos(torch.pi - self.theta)
        sin_theta = torch.sin(torch.pi - self.theta)
        self.eyePoint = torch.tensor([self.radius * sin_phi * cos_theta, self.radius * torch.cos(self.phi), self.radius * sin_phi * sin_theta], device=self.fov.device)
        self.forward_vector = torch.nn.functional.normalize(self.lookAt - self.eyePoint, p=2, dim=0)
        self.right_vector = -torch.nn.functional.normalize(torch.cross(self._default_up_vector, self.forward_vector, dim=-1), p=2, dim=0)
        self.up_vector = torch.nn.functional.normalize(torch.cross(self.forward_vector, self.right_vector, dim=-1), p=2, dim=0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect_ratio, 0.01, 100)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.eyePoint, *self.lookAt, *-self.up_vector)

    def get_matrix(self) -> tuple[Tensor, Tensor]:
        intrinsic_matrix = get_intrinsic_matrix(self)
        matrix = torch.eye(4, device=self.eyePoint.device)
        matrix[:3, :3] = torch.stack((self.right_vector, self.up_vector, self.forward_vector), dim=-1)
        matrix[:3, 3] = self.eyePoint
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

    def to_tensor(self):
        return torch.tensor([self.fov, self.lookAt[0], self.lookAt[1], self.lookAt[2], self.radius, self.phi, self.theta], device=self.fov.device)

    def poll_event(self, e):
        if e.type == MOUSEBUTTONDOWN:
            if e.button == 4:
                self.radius -= 0.1
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
