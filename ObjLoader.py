import numpy as np
import pygame
from OpenGL.GL import *


class OBJ:
    def __init__(self, filename, swapYZ=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.textures_coords = []
        self.faces = []

        material = None
        path = filename.split('/')[:-1]
        path = '/'.join(map(str, path))
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = values[1:4]
                for i in range(len(v)):
                    v[i] = float(v[i])
                if swapYZ:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapYZ:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.textures_coords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl_data = OBJ.mtl(f'{path}/{values[1]}')
            elif values[0] == 'f':
                face = []
                textures_coords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        textures_coords.append(int(w[1]))
                    else:
                        textures_coords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, textures_coords, material))

        self.textures_coords = np.array(self.textures_coords)
        self.vertices = np.array(self.vertices)
        self.normals = np.array(self.normals)
        self.vertices *= 10
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face
            try:
                mtl = self.mtl_data[material]
                if 'texture_Kd' in mtl:
                    # use diffuse texture map
                    glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
                else:
                    # just use diffuse colour
                    glColor(*mtl['Kd'])
            except AttributeError:
                pass

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    tex_coo = np.array(list(self.textures_coords[texture_coords[i] - 1]))
                    glTexCoord2fv(tex_coo)
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()

    @staticmethod
    def mtl(filename):
        contents = {}
        mtl_data = None
        path = filename.split('/')[:-1]
        path = '/'.join(map(str, path))
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'newmtl':
                mtl_data = contents[values[1]] = {}
            elif mtl_data is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl_data[values[0]] = values[1]
                surf = pygame.image.load(path + "/" + mtl_data['map_Kd'])
                image = pygame.image.tostring(surf, 'RGBA', True)
                ix, iy = surf.get_rect().size
                texid = mtl_data['texture_Kd'] = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texid)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                             GL_UNSIGNED_BYTE, image)
            else:
                mtl_data[values[0]] = map(float, values[1:])
        return contents
