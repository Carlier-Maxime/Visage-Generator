from typing import Any
import os, torch, numpy as np, PIL.Image
from renderer import Renderer

class Saver():
    def __init__(self, location, enable:bool=True) -> None:
        self.location = location
        self.enable = enable

    def __call__(self, index, filename, *args: Any, **kwds: Any) -> Any:
        if not self.enable: return
        self.sub_folder = format(index//1000,"05d")
        path = f'{self.location}/{self.sub_folder}'
        os.makedirs(path, exist_ok=True)
        self._saving(path+f'/{filename}', *args, **kwds)

    def _saving(self, path, *args: Any, **kwds: Any):
        pass

    def setEnable(self, enable:bool):
        self.enable=enable

class TorchSaver(Saver):
    def __init__(self, location, enable:bool=True) -> None:
        super().__init__(location, enable)

    def _saving(self, path, data, *args: Any, **kwds: Any) -> Any:
        torch.save(data, path, *args, **kwds)
    
class NumpySaver(Saver):
    def __init__(self, location, enable: bool = True) -> None:
        super().__init__(location, enable)

    def _saving(self, path, data, *args: Any, **kwds: Any) -> Any:
        np.save(path, data, *args, **kwds)
    
class ObjSaver(Saver):
    def __init__(self, location, renderer:Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        self.render = renderer

    def _saving(self, path:str, vertices, faces, *args: Any, texture=None, **kwds: Any):
        basename = path.split(".obj")[0]
        if texture is not None:
            img = PIL.Image.fromarray(texture)
            img.save(basename+"_texture.png")
            uvcoords = self.render.uvcoords.cpu().numpy().reshape((-1, 2))
            uvfaces = self.render.uvfaces.cpu().numpy()
            with open(basename+".mtl","w") as f:
                f.write(f'newmtl material_0\nmap_Kd {basename.split("/")[-1]}_texture.png\n')
        with open(path, 'w') as f:
            if texture is not None:
                f.write(f'mtllib {basename.split("/")[-1]}.mtl\n')
            np.savetxt(f, vertices, fmt='v %.6f %.6f %.6f')
            if texture is None:
                np.savetxt(f, faces + 1, fmt='f %d %d %d')
            else:
                np.savetxt(f, uvcoords, fmt='vt %.6f %.6f')
                f.write(f'usemtl material_0\n')
                np.savetxt(f, np.hstack((faces + 1, uvfaces + 1))[:,[0,3,1,4,2,5]], fmt='f %d/%d %d/%d %d/%d')

class Lmks2DSaver(Saver):
    def __init__(self, location, renderer:Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        self.render = renderer

    def _saving(self, path:str, lmks, *args: Any, **kwds: Any):
        lmks2D = []
        for p in lmks:
            lmks2D.append(self.render.getCoord2D(p))
        if path.endswith('.npy'):
            np.save(path, lmks2D[17:])
        elif path.endswith('.pts'):
            with open(path, 'w') as f:
                lmks2D = lmks2D[17:]
                for i in range(len(lmks2D)):
                    f.write(f'{i + 1} {lmks2D[i][0]} {lmks2D[i][1]} False\n')
        else: raise TypeError("format for saving landmarks 2D is not supported !")

class VisageImageSaver(Saver):
    def __init__(self, location, renderer:Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        self.render = renderer

    def _saving(self, path, vertices, texture, *args: Any, pts=None, ptsInAlpha=True, camera=None, **kwds: Any):
        self.render.save_to_image(path, vertices, texture, pts=pts, ptsInAlpha=ptsInAlpha, camera=camera)

class CameraJSONSaver(Saver):
    def __init__(self, location, renderer:Renderer, enable: bool = True) -> None:
        super().__init__(location, enable)
        
        self.render = renderer
        self.first=True
        self.__close=False

    def __call__(self, index, filename, *args: Any, **kwds: Any) -> Any:
        if self.enable: 
            if self.first: 
                os.makedirs(self.location, exist_ok=True)
                self.file = open(f'{self.location}/cameras.json','w')
                self.file.write('{"labels": {\n')
                self.first=False
            else: self.file.write(',\n')
            self._saving(filename, *args, **kwds)

    def _saving(self, basename, camera, *args: Any, **kwds: Any):
        intrinsic_norm, extrinsic = self.render.getCameraMatrices(camera)
        np.savetxt(self.file, torch.cat([extrinsic.flatten(), intrinsic_norm.flatten()]).view(1,-1).cpu().numpy(), '%f', delimiter=',', newline='', header=f'"{basename}": [', footer=f']', comments='')
    
    def close(self):
        if not self.first:
            self.file.write("\n}}")
            self.file.close()
        self.__close=True

    def __del__(self):
        if not self.__close: self.close()