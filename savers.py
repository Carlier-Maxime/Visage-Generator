import cv2
import numpy as np
import os
import torch
import mrcfile
from typing import Any

from renderer import Renderer


class Saver:
    def __init__(self, location, enable: bool = True) -> None:
        self.location = location
        self.enable = enable

    def __call__(self, index, filename, *args: Any, **kwargs: Any) -> Any:
        if not self.enable:
            return
        self.sub_folder = format(index // 1000, "05d")
        path = f'{self.location}/{self.sub_folder}'
        os.makedirs(path, exist_ok=True)
        self._saving(path + f'/{filename}', *args, **kwargs)

    def _saving(self, path, *args: Any, **kwargs: Any):
        pass

    def set_enable(self, enable: bool):
        self.enable = enable


class TorchSaver(Saver):
    def __init__(self, location, enable: bool = True) -> None:
        super().__init__(location, enable)

    def _saving(self, path, data, *args: Any, **kwargs: Any) -> Any:
        torch.save(data, path, *args, **kwargs)


class NumpySaver(Saver):
    def __init__(self, location, enable: bool = True) -> None:
        super().__init__(location, enable)

    def _saving(self, path, data, *args: Any, **kwargs: Any) -> Any:
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        np.save(path, data, *args, **kwargs)


class ObjSaver(Saver):
    def __init__(self, location, renderer: Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        self.render = renderer
        self.uv_coords = self.render.uv_coords.cpu().numpy().reshape((-1, 2))
        self.uv_faces = self.render.uv_faces.cpu().numpy()

    def _saving(self, path: str, vertices, faces: torch.Tensor, *args: Any, texture=None, **kwargs: Any):
        basename = path.split(".obj")[0]
        faces = faces.cpu().numpy()
        vertices = vertices.cpu().numpy()
        if texture is not None:
            cv2.imwrite(basename + "_texture.png", texture[:, :, [2, 1, 0]])
            with open(basename + ".mtl", "w") as f:
                f.write(f'newmtl material_0\nmap_Kd {basename.split("/")[-1]}_texture.png\n')
        with open(path, 'w') as f:
            if texture is not None:
                f.write(f'mtllib {basename.split("/")[-1]}.mtl\n')
            np.savetxt(f, vertices, fmt='v %.6g %.6g %.6g')
            if texture is None:
                np.savetxt(f, faces + 1, fmt='f %d %d %d')
            else:
                np.savetxt(f, self.uv_coords, fmt='vt %.6g %.6g')
                f.write(f'usemtl material_0\n')
                np.savetxt(f, np.hstack((faces, self.uv_faces))[:, [0, 3, 1, 4, 2, 5]] + 1, fmt='f %d/%d %d/%d %d/%d')


class Lmks2DSaver(Saver):
    def __init__(self, location, renderer: Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        self.render = renderer

    def _saving(self, path: str, lmks, *args: Any, vertical_flip: bool = True, **kwargs: Any):
        lmks_2d = []
        for p in lmks:
            lmks_2d.append(self.render.get_coord_2d(p, vertical_flip=vertical_flip))
        if path.endswith('.npy'):
            np.save(path, lmks_2d[17:])
        elif path.endswith('.pts'):
            with open(path, 'w') as f:
                lmks_2d = lmks_2d[17:]
                for i in range(len(lmks_2d)):
                    f.write(f'{i + 1} {lmks_2d[i][0]} {lmks_2d[i][1]} False\n')
        else:
            raise TypeError("format for saving landmarks 2D is not supported !")


class VisageImageSaver(Saver):
    def __init__(self, location, renderer: Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        self.render = renderer

    def _saving(self, path, vertices, texture, *args: Any, pts=None, pts_in_alpha=True, vertical_flip: bool = True, save_depth: bool = False, depth_in_alpha: bool = False, **kwargs: Any):
        self.render.save_to_image(path, vertices, texture, pts=pts, pts_in_alpha=pts_in_alpha, vertical_flip=vertical_flip, save_depth=save_depth, depth_in_alpha=depth_in_alpha)


class CameraJSONSaver(Saver):
    def __init__(self, location, renderer: Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)

        self.render = renderer
        self.first = True
        self.__close = False

    def __call__(self, index, filename, *args: Any, **kwargs: Any) -> Any:
        if self.enable:
            if self.first:
                os.makedirs(self.location, exist_ok=True)
                self.file = open(f'{self.location}/cameras.json', 'w')
                self.file.write('{"labels": {\n')
                self.first = False
            else:
                self.file.write(',\n')
            self._saving(format(index // 1000, "05d") + '/' + filename, *args, **kwargs)

    def _saving(self, basename, camera, *args: Any, **kwargs: Any):
        self.render.change_camera(camera)
        intrinsic_norm, extrinsic = self.render.get_camera().get_matrix()
        np.savetxt(self.file, torch.cat([extrinsic.flatten(), intrinsic_norm.flatten()]).view(1, -1).cpu().numpy(), '%f', delimiter=',', newline='', header=f'"{basename}": [', footer=f']', comments='')

    def close(self):
        if not self.first:
            self.file.write("\n}}")
            self.file.close()
        self.__close = True

    def __del__(self):
        if not self.__close:
            self.close()


class DensityCubeSaver(Saver):
    def __init__(self, location, size, device, enable: bool = True, method_pts_in_tri: str = 'barycentric') -> None:
        super().__init__(location, enable)
        self.epsilon = size * 5e-3
        self.point_in_triangle_method = self.point_in_triangle_barycentric if method_pts_in_tri == 'barycentric' else self.point_in_triangle_normal
        x = y = z = torch.arange(size, dtype=torch.int16, device=device)
        self.cube_indices = torch.stack(torch.meshgrid(x, y, z), dim=3).view(-1, 3)
        self.size = size

    @staticmethod
    def get_tri_nearest(mesh, pts, pts_batch_size: int = 10000):
        from tqdm import trange
        centers = mesh.mean(dim=1).to(torch.float16)
        tri_nearest = torch.empty((pts.shape[0], 3, 3), dtype=torch.float, device=mesh.device)
        for limit in trange(pts_batch_size, pts.shape[0] + pts_batch_size, pts_batch_size, desc='Density Cube : Search nearest triangle', unit='batch', leave=False):
            tri_nearest[limit - pts_batch_size:limit] = mesh[torch.norm(centers[:, None, :].sub(pts[limit - pts_batch_size:limit][None]), dim=2).min(dim=0).indices]
        return tri_nearest

    def point_in_triangle_normal(self, pts, triangle, tri_vecs, normal):
        pts_vecs = pts[:, None, :].sub(triangle).permute(1, 0, 2)
        return torch.cross(tri_vecs, pts_vecs, dim=-1).mul(normal).sum(dim=-1).ge(-self.epsilon).all(dim=0)

    def point_in_triangle_barycentric(self, pts, triangle, tri_vecs, _):
        v0 = -tri_vecs[2]
        v1 = tri_vecs[0]
        v2 = pts - triangle[:, 0]

        dot00 = v0.mul(v0).sum(dim=-1)
        dot01 = v0.mul(v1).sum(dim=-1)
        dot02 = v0.mul(v2).sum(dim=-1)
        dot11 = v1.mul(v1).sum(dim=-1)
        dot12 = v1.mul(v2).sum(dim=-1)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        return (u >= -self.epsilon) & (v >= -self.epsilon) & (u + v <= 1+self.epsilon)

    def _saving(self, path, vertices, faces, v_interval: int = 0, pts_batch_size: int = 10000, *args: Any, **kwargs: Any) -> Any:
        vertices -= vertices.min()
        if v_interval == 0: v_interval = vertices.max()
        vertices *= self.size / v_interval
        mesh = vertices[faces]
        tri_nearest = self.get_tri_nearest(mesh, self.cube_indices)
        tri_vecs = torch.stack([tri_nearest[:, (i + 1) % 3].sub(tri_nearest[:, i]) for i in range(3)])
        normal = torch.cross(tri_vecs[0], tri_vecs[1])
        normal = normal.divide(torch.norm(normal, dim=0))
        dist_to_plane = normal.mul(tri_nearest[:, 0].sub(self.cube_indices).mul(normal).sum(dim=-1).divide(normal.square().sum(dim=-1))[:, None]).norm(dim=-1)
        cube_in_volume_tri = self.point_in_triangle_method(self.cube_indices, tri_nearest, tri_vecs, normal)
        cube = torch.zeros((self.size ** 3), device=mesh.device)
        cube[cube_in_volume_tri] = dist_to_plane[cube_in_volume_tri]
        segment_factor = self.cube_indices[None].sub(tri_nearest.permute(1, 0, 2)[[1, 2, 0]]).mul(tri_vecs).sum(dim=-1).divide(tri_vecs.mul(tri_vecs).sum(dim=-1)).clamp(0, 1)
        closest_point = tri_nearest.permute(1, 0, 2)[[1, 2, 0]].add(segment_factor[:, :, None].mul(tri_vecs))
        dist_to_segment = torch.norm(self.cube_indices.sub(closest_point), dim=-1).abs().min(dim=0).values
        cube[~cube_in_volume_tri] = dist_to_segment[~cube_in_volume_tri]
        cube = cube.view(self.size, self.size, self.size).mul(-1).add(cube.max()).divide(self.size)
        # cube = torch.pow(torch.e, cube.mul(torch.e)).sub(1).divide(torch.e**torch.e-1)
        with mrcfile.new_mmap(path, overwrite=True, shape=cube.shape, mrc_mode=2) as mrc:
            mrc.data[:] = cube.cpu().numpy()
