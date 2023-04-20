"""
Author: Yao Feng
Copyright (c) 2020, Yao Feng
All rights reserved.
"""
import warnings
from pytorch3d.io import load_obj
import pygame
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import PIL.Image

class Renderer():
    def __init__(self, obj_filename, width, height):
        warnings.filterwarnings("ignore",'No mtl file provided')
        verts, faces, aux = load_obj(obj_filename)
        self.uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        self.uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        self.faces = faces.verts_idx[None, ...]

        pygame.init()
        self.width = width
        self.height = height
        hx = width/2
        hy = height/2
        srf = pygame.display.set_mode([width, height], pygame.constants.OPENGL | pygame.constants.DOUBLEBUF)

        glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width/float(height), 1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        self.rx, self.ry = (-79,41)
        self.tx, self.ty = (-29,-14)
        self.zpos = 4

    def __create_GL_texture(self, texture):
        print("create texture.. ",end=None)
        size = texture.shape
        image = PIL.Image.fromarray(texture,'RGB').convert('RGBA').transpose(PIL.Image.FLIP_TOP_BOTTOM).tobytes()
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size[0], size[1], 0, GL_RGBA,GL_UNSIGNED_BYTE, image)
        print("Done")
        return texid

    def __create_GL_List(self, vertices, faces, texture):
        print("Create GL_List... ",end=None)
        vertices = vertices[:,[0,2,1]]
        vertices *= 10
        gl_list = glGenLists(1)
        glNewList(gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        glBindTexture(GL_TEXTURE_2D, self.__create_GL_texture(texture))
        for face, uvface in zip(faces, self.uvfaces[0]):
            glBegin(GL_POLYGON)
            for i in range(len(face)):
                #if normals[i] > 0:
                #    pass glNormal3fv(self.normals[normals[i] - 1])
                tex_coo = self.uvcoords[0][uvface[i]].detach().cpu().numpy()
                glTexCoord2fv(tex_coo)
                glVertex3fv(vertices[face[i]].detach().cpu().numpy())
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()
        print("Done")
        return gl_list

    def test(self, gl_list):
        run=1
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        clock = pygame.time.Clock()
        rotate = move = False
        while run:
            clock.tick(30)
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    PIL.Image.frombytes('RGB', (self.width, self.height), glReadPixels(0,0,self.width,self.height, GL_RGB, GL_UNSIGNED_BYTE)).transpose(PIL.Image.FLIP_TOP_BOTTOM).save("test.jpg")
                    run=0
                    break
                elif e.type == MOUSEBUTTONDOWN:
                    if e.button == 4: self.zpos = max(1, self.zpos-0.1)
                    elif e.button == 5: self.zpos += 0.1
                    elif e.button == 1: rotate = True
                    elif e.button == 3: move = True
                elif e.type == MOUSEBUTTONUP:
                    if e.button == 1: rotate = False
                    elif e.button == 3: move = False
                elif e.type == MOUSEMOTION:
                    i, j = e.rel
                    if rotate:
                        self.rx += i
                        self.ry += j
                    if move:
                        self.tx += i
                        self.ty -= j
                elif e.type == KEYDOWN and e.key == K_c:
                    print(f'rx: {self.rx}, ry: {self.ry}, tx: {self.tx}, ty: {self.ty}, zpos: {self.zpos}')
            self.__render(gl_list)

    def __render(self,gl_list):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # RENDER OBJECT
        glTranslate(self.tx/20., self.ty/20., - self.zpos)
        glRotate(self.ry, 1, 0, 0)
        glRotate(self.rx, 0, 1, 0)
        glCallList(gl_list)
        pygame.display.flip()

    def save_to_image(self, filename, vertices, faces, texture):
        gl_list = self.__create_GL_List(vertices, faces, texture)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        self.__render(gl_list)
        PIL.Image.frombytes('RGB', (self.width, self.height), glReadPixels(0,0,self.width,self.height, GL_RGB, GL_UNSIGNED_BYTE)).transpose(PIL.Image.FLIP_TOP_BOTTOM).save(filename)

if __name__ == '__main__':
    import ObjLoader, sys
    render = Renderer("visage.obj", 1024, 1024)
    obj = ObjLoader.OBJ(sys.argv[1], swapyz=True)
    render.test(obj.gl_list)