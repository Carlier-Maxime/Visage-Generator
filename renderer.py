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
import torch

class Renderer():
    def __init__(self, obj_filename:str, width:int, height:int, device:torch.device, show:bool=True):
        warnings.filterwarnings("ignore",'No mtl file provided')
        self.device = torch.device(device)
        _, faces, aux = load_obj(obj_filename, device=device)
        self.uvcoords = aux.verts_uvs # (N, V, 2)
        self.uvfaces = faces.textures_idx  # (N, F, 3)
        self.order_indexs = torch.cat([self.uvfaces, faces.verts_idx],dim=1)[:,[0,3,1,4,2,5]].reshape([self.uvfaces.shape[0]*3,2]).unique(dim=0)
        self.raw_sphere = self.create_sphere(0.02, 30, 30)

        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_mode([width, height], pygame.constants.OPENGL | pygame.constants.DOUBLEBUF | pygame.SHOWN if show else pygame.HIDDEN)

        glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
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

        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)

        self.buffers = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2])
        glBufferData(GL_ARRAY_BUFFER, self.uvcoords.cpu().numpy(), GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.uvfaces.cpu().numpy().astype('uint32'), GL_STATIC_DRAW)

        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2])
        glTexCoordPointer(2, GL_FLOAT, 0, None)

        self.texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR)

        self.gl_list_visage = glGenLists(1)

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslate(self.tx/20., self.ty/20., - self.zpos)
        glRotate(self.ry, 1, 0, 0)
        glRotate(self.rx, 0, 1, 0)

    def __del__(self):
        glDisable(GL_TEXTURE_2D)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDeleteBuffers(3, self.buffers)
        glDeleteLists(self.gl_list_visage,1)

    def __change_GL_texture(self, texture):
        size = texture.shape
        image = PIL.Image.fromarray(texture,'RGB').transpose(PIL.Image.FLIP_TOP_BOTTOM).tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size[0], size[1], 0, GL_RGB,GL_UNSIGNED_BYTE, image)
    
    def __edit_GL_List(self, vertices, texture):
        vertices = vertices[:, [0, 2, 1]]
        vertices *= 10
        vertices = vertices[self.order_indexs[:,1]]
        vertices = vertices.cpu().numpy()

        glNewList(self.gl_list_visage, GL_COMPILE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[1])
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STREAM_DRAW)
        glVertexPointer(3, GL_FLOAT, 0, None)
        self.__change_GL_texture(texture)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[0])
        glDrawElements(GL_TRIANGLES, self.uvfaces.numel(), GL_UNSIGNED_INT, None)
        glEndList()

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

    def __render(self,gl_lists):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glCallLists(gl_lists)
        pygame.display.flip()

    def save_to_image(self, filename, vertices, texture, pts=None, ptsInAlpha:bool=True):
        self.__edit_GL_List(vertices, texture)
        if pts is not None:
            pts = pts[:, [0, 2, 1]]
            pts *= 10
            pts_gl_list = Renderer.create_spheres_gl_list(self.raw_sphere, pts)
            if ptsInAlpha:
                self.__render([self.gl_list_visage])
                img_visage = torch.frombuffer(bytearray(glReadPixels(0,0,self.width,self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glColor(0.,0.,0.)
                glCallList(self.gl_list_visage)
                glColor(1.,1.,1.)
                glCallList(pts_gl_list)
                pygame.display.flip()
                img_pts = torch.frombuffer(bytearray(glReadPixels(0,0,self.width,self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
                green_mask = (img_pts == torch.tensor([0, 255, 0, 255], device=self.device)).all(dim=2)
                new_colors = img_visage[green_mask]
                new_colors[:,3] = 0
                img_visage[green_mask] = new_colors
            else: 
                self.__render([self.gl_list_visage, pts_gl_list])
                img_visage = torch.frombuffer(bytearray(glReadPixels(0,0,self.width,self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
        else:
            self.__render([self.gl_list_visage])
            img_visage = torch.frombuffer(bytearray(glReadPixels(0,0,self.width,self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
        PIL.Image.fromarray(img_visage.cpu().numpy()).transpose(PIL.Image.FLIP_TOP_BOTTOM).save(filename)

    def create_sphere(self, radius, slices, stacks):
        vertex_array = torch.zeros(((stacks + 1) * (slices + 1), 3), dtype=torch.float32, device=self.device)
        tpi = torch.tensor(torch.pi, device=self.device)
        for i in range(stacks + 1):
            theta = i * tpi / stacks
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)

            for j in range(slices + 1):
                phi = j * 2 * tpi / slices
                sin_phi = torch.sin(phi)
                cos_phi = torch.cos(phi)

                x = radius * cos_phi * sin_theta
                y = radius * cos_theta
                z = radius * sin_phi * sin_theta

                vertex_array[i * (slices + 1) + j] = torch.tensor([x, y, z], device=self.device)

        indices = torch.zeros((stacks, slices, 6), dtype=torch.int64, device=self.device)
        for i in range(stacks):
            for j in range(slices):
                p1 = i * (slices + 1) + j
                p2 = p1 + slices + 1
                indices[i, j] = torch.tensor([p1, p2, p1 + 1, p1 + 1, p2, p2 + 1], dtype=torch.int64, device=self.device)

        return vertex_array, indices


    def create_spheres_gl_list(raw_sphere, positions):
        vertex_array, indices = raw_sphere
        nb_spheres = positions.shape[0]
        indices = indices.clone().flatten()
        a=torch.arange(0,vertex_array.shape[0]*nb_spheres,vertex_array.shape[0], device=indices.device)
        indices  = (indices.reshape([indices.shape[0]//3,3]).repeat(nb_spheres,1,1).permute(1,2,0) + a).permute(2,0,1)
        vertex_array = (vertex_array.clone().repeat(nb_spheres,1,1).permute(1,0,2) + positions).permute(1,0,2)
        list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)
        glDisable(GL_TEXTURE_2D)
        glColor(0, 255., 0)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_array.cpu().numpy(), GL_STATIC_DRAW)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.cpu().numpy().astype('uint32'), GL_STATIC_DRAW)
        glDrawElements(GL_TRIANGLES, indices.numel(), GL_UNSIGNED_INT, None)
        glColor(1., 1., 1.)
        glEnable(GL_TEXTURE_2D)
        glEndList()
        glDeleteBuffers(2, [vbo, ibo])
        return list_id


if __name__ == '__main__':
    import ObjLoader, sys
    render = Renderer("visage.obj", 1024, 1024)
    obj = ObjLoader.OBJ(sys.argv[1], swapyz=True)
    render.test(obj.gl_list)