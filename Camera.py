import abc
from typing import Tuple

import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from torch import Tensor


class BaseCamera:
    def __init__(self, camera: torch.Tensor, width: int, height: int):
        self.width = self.height = None
        self.set_size(width, height)
        self.tensor = camera

    @abc.abstractmethod
    def _update(self) -> None:
        pass

    @abc.abstractmethod
    def get_matrix(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def set_camera(self, camera: torch.Tensor) -> None:
        pass

    def set_size(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def get_tensor(self):
        return self.tensor


class DefaultCamera(BaseCamera):

    def __init__(self, camera: torch.Tensor, width, height):
        super().__init__(camera, width, height)
        self.fov = self.tx = self.ty = self.tz = self.rx = self.ry = self.rz = None
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
        focal_length = (self.width / (2 * torch.pi)) / torch.tan(torch.deg2rad(self.fov / 2.0))

        # Calcul de la matrice intrinsèque normalisé
        intrinsic_matrix = torch.tensor([
            [focal_length / self.width, 0, 0.5],
            [0, focal_length / self.height, 0.5],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.fov.device)

        extrinsic_matrix = torch.tensor(glGetFloatv(GL_MODELVIEW_MATRIX), device=self.fov.device)
        extrinsic_matrix[:, 3] = extrinsic_matrix[3]
        extrinsic_matrix[3] = torch.tensor([0, 0, 0, 1], device=self.fov.device)

        return intrinsic_matrix, extrinsic_matrix

    def set_camera(self, camera: torch.Tensor) -> None:
        self.fov, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz = camera
        self._update()
