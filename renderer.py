"""
Author: Yao Feng
Copyright (c) 2020, Yao Feng
All rights reserved.
"""
import warnings
import torch.nn.functional as F
from pytorch3d.io import load_obj

class Renderer():
    def __init__(self, image_size, obj_filename, uv_size=256):
        warnings.filterwarnings("ignore",'No mtl file provided')
        verts, faces, aux = load_obj(obj_filename)
        self.raw_uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        self.uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        self.faces = faces.verts_idx[None, ...]