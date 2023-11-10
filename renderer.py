import numpy as np
import pygame
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import torch
from Camera import *


class Renderer:
    def __init__(self, width: int, height: int, device: torch.device, show: bool = True, camera: torch.Tensor | None = None, camera_type: str = 'default'):
        self.show = show
        if camera is None:
            camera = torch.tensor([10, 0, 0, -2, 0, 0, 0], device=device, dtype=torch.float32)
        if camera_type == 'default':
            self.camera = DefaultCamera(camera, width, height)
        elif camera_type == 'vector':
            self.camera = VectorCamera(camera, width, height)
        self.device = torch.device(device)
        render_data = torch.load('render_data.pt')
        self.uv_coords = render_data['uv_coords'].to(self.device)
        self.uv_faces = render_data['uv_faces'].to(self.device)
        self.order_indices = render_data['order_indices'].to(self.device)
        self.raw_sphere = self.create_sphere(0.002, 30, 30)

        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_icon(pygame.image.load('logo.png'))
        pygame.display.set_mode([width, height], pygame.constants.OPENGL | pygame.constants.DOUBLEBUF | pygame.SHOWN if show else pygame.HIDDEN)

        framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_texture, 0)

        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Erreur: Le framebuffer n'est pas complet.")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return
        self.frameBuffer = framebuffer

        glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnableClientState(GL_VERTEX_ARRAY)

        self.rotate = False
        self.rotate_z = False
        self.move = False

        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)

        self.buffers = glGenBuffers(5)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2])
        glBufferData(GL_ARRAY_BUFFER, self.uv_coords.cpu().numpy(), GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[0])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.uv_faces.cpu().numpy().astype('uint32'), GL_STATIC_DRAW)

        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2])
        glTexCoordPointer(2, GL_FLOAT, 0, None)

        self.texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.gl_list_visage = glGenLists(1)

        glEnable(GL_DEPTH_TEST)
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def __del__(self):
        glDisable(GL_TEXTURE_2D)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDeleteBuffers(5, self.buffers)
        glDeleteLists(self.gl_list_visage, 1)

    def _change_gl_texture(self, texture):
        size = texture.shape
        image = cv2.flip(texture[:, :, [2, 1, 0]], 0).tobytes()
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size[0], size[1], 0, GL_BGR, GL_UNSIGNED_BYTE, image)

    def _create_gl_list(self, vertices, triangles):
        vertices = vertices[self.order_indices[:, 1]]
        vertices = vertices.cpu().numpy()
        vbo, fbo = glCreateBuffers(2)
        gl_list = glGenLists(1)
        glNewList(gl_list, GL_COMPILE)
        glDisable(GL_TEXTURE_2D)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.cpu().numpy(), GL_STATIC_DRAW)
        glDrawElements(GL_TRIANGLES, triangles.numel(), GL_UNSIGNED_INT, None)
        glEnable(GL_TEXTURE_2D)
        glEndList()
        glDeleteBuffers(2, [vbo, fbo])
        return gl_list

    def _edit_gl_list(self, vertices, texture):
        vertices = vertices[self.order_indices[:, 1]]
        vertices = vertices.cpu().numpy()

        glNewList(self.gl_list_visage, GL_COMPILE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[1])
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STREAM_DRAW)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[0])
        if texture is not None:
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            self._change_gl_texture(texture)
            glBindBuffer(GL_ARRAY_BUFFER, self.buffers[2])
            glTexCoordPointer(2, GL_FLOAT, 0, None)
        else:
            glDisable(GL_TEXTURE_2D)
        glDrawElements(GL_TRIANGLES, self.uv_faces.numel(), GL_UNSIGNED_INT, None)
        if texture is None:
            glEnable(GL_TEXTURE_2D)
        glEndList()

    def test(self, gl_list):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        clock = pygame.time.Clock()
        while 1:
            clock.tick(60)
            if not self._poll_events():
                break
            self._render(gl_list)
        image = np.array(bytearray(glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE))).reshape([self.height, self.width, 3])
        cv2.imwrite('test.jpg', cv2.flip(image, 0))

    def _poll_event(self, e):
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            return 0
        self.camera.poll_event(e)
        return 1

    def _poll_events(self):
        for e in pygame.event.get():
            if not self._poll_event(e):
                return 0
        return 1

    def change_camera(self, camera: torch.Tensor) -> None:
        self.camera.set_camera(camera)

    def _render(self, gl_lists):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glCallLists(gl_lists)
        if self.show:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.frameBuffer)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)
        pygame.display.flip()

    def random_background(self):
        glClearColor(*torch.rand(3, device=self.device).tolist(), 1.)

    def getImageFromTextureColor(self):
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, img)
        img = np.frombuffer(img, np.uint8).reshape([self.height, self.width, -1]).copy()
        glBindTexture(GL_TEXTURE_2D, self.texid)
        return img

    def save_to_image(self, filename, vertices, texture, pts=None, pts_in_alpha: bool = True, vertical_flip: bool = True, save_depth: bool = False, depth_in_alpha: bool = False):
        self._edit_gl_list(vertices, texture)
        save_color: bool = not save_depth or depth_in_alpha
        if pts is not None and save_color:
            pts_gl_list = self.create_spheres_gl_list(self.raw_sphere, pts)
            if pts_in_alpha:
                assert not save_depth or not depth_in_alpha
                self._render([self.gl_list_visage])
                img_visage = torch.from_numpy(self.getImageFromTextureColor()).to(self.device)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glColor(0., 0., 0.)
                glCallList(self.gl_list_visage)
                glColor(1., 1., 1.)
                glCallList(pts_gl_list)
                pygame.display.flip()

                img_pts = np.zeros((self.height, self.width, 1), dtype=np.uint8)
                glBindTexture(GL_TEXTURE_2D, self.color_texture)
                glGetTexImage(GL_TEXTURE_2D, 0, GL_GREEN, GL_UNSIGNED_BYTE, img_pts)
                img_pts = torch.frombuffer(bytearray(img_pts), dtype=torch.uint8).view(self.height, self.width, -1)
                glBindTexture(GL_TEXTURE_2D, self.texid)

                green_mask = (img_pts == 255).all(dim=2)
                new_colors = img_visage[green_mask]
                new_colors[:, 3] = 0
                img_visage[green_mask] = new_colors
                img = img_visage.cpu().numpy()
            else:
                self._render([self.gl_list_visage, pts_gl_list])
                img = self.getImageFromTextureColor()
            glDeleteLists(pts_gl_list, 1)
        elif save_color:
            self._render([self.gl_list_visage])
            img = self.getImageFromTextureColor()
        else:
            self._render([self.gl_list_visage])
            img = None
        if save_depth or depth_in_alpha:
            depth_values = np.zeros((self.height, self.width), dtype=np.float32)
            glBindTexture(GL_TEXTURE_2D, self.depth_texture)
            glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth_values)
            depth_image = torch.frombuffer(bytearray(depth_values), dtype=torch.float32).to(self.device).view(self.height, self.width)
            mask = depth_image < 1
            tmp = depth_image[mask]
            depth_min, depth_max = tmp.min(), tmp.max()
            depth_image[mask] = ((tmp - depth_min) / (depth_max - depth_min)) * (1. - 0.2)
            depth_image = (depth_image - 1) * -1
            depth_image = (depth_image * 255.).to(torch.uint8).cpu().numpy()
            if depth_in_alpha:
                assert not pts_in_alpha or pts is None
                assert img is not None
                img[:, :, 3] = depth_image
            else:
                img = depth_image
        cv2.imwrite(filename, cv2.flip(img, 0) if vertical_flip else img)

    def create_sphere(self, radius, slices, stacks):
        theta = torch.linspace(0, torch.pi, stacks + 1, device=self.device).repeat(slices + 1).reshape(stacks + 1, slices + 1).permute(1, 0)[None]
        phi = torch.linspace(0, 2 * torch.pi, slices + 1, device=self.device)
        sin_theta = torch.sin(theta)
        x = radius * torch.cos(phi) * sin_theta
        y = radius * torch.cos(theta)
        z = radius * torch.sin(phi) * sin_theta
        vertex_array = torch.cat([x, y, z], dim=0).permute(1, 2, 0).flatten(end_dim=1)

        i = torch.arange(stacks, dtype=torch.int64, device=self.device)
        j = torch.arange(slices, dtype=torch.int64, device=self.device)
        p1 = ((i * (slices + 1)).repeat(j.shape).reshape(stacks, slices).permute(1, 0) + j)[None]
        p2 = p1 + slices + 1
        indices = torch.cat([p1, p2, p1 + 1, p1 + 1, p2, p2 + 1], dim=0).permute(1, 2, 0)
        return vertex_array, indices

    def create_spheres_gl_list(self, raw_sphere, positions, list_id=None, color=None):
        if color is None:
            color = [0., 255., 0.]
        vertex_array, indices = raw_sphere
        nb_spheres = positions.shape[0]
        indices = indices.flatten()
        a = torch.arange(0, vertex_array.shape[0] * nb_spheres, vertex_array.shape[0], device=indices.device)
        indices = (indices.reshape([indices.shape[0] // 3, 3]).repeat(nb_spheres, 1, 1).permute(1, 2, 0) + a).permute(2, 0, 1)
        vertex_array = (vertex_array.repeat(nb_spheres, 1, 1).permute(1, 0, 2) + positions).permute(1, 0, 2)
        if list_id is None:
            list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)
        glDisable(GL_TEXTURE_2D)
        glColor(color[0], color[1], color[2])
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[3])
        glBufferData(GL_ARRAY_BUFFER, vertex_array.cpu().numpy(), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[3])
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[4])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.cpu().numpy().astype('uint32'), GL_STATIC_DRAW)
        glDrawElements(GL_TRIANGLES, indices.numel(), GL_UNSIGNED_INT, None)
        glColor(1., 1., 1.)
        glEnable(GL_TEXTURE_2D)
        glEndList()
        return list_id

    def get_coord_2d(self, point3d, vertical_flip: bool = True):
        viewport = glGetIntegerv(GL_VIEWPORT)
        model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        win_x, win_y, win_z = gluProject(point3d[0], point3d[1], point3d[2], model_view, projection_matrix, viewport)
        if 0 <= win_z <= 1:
            return int(win_x), (self.height-int(win_y)-1) if vertical_flip else int(win_y)
        return None

    def get_camera(self):
        return self.camera

    @staticmethod
    def void_events():
        for _ in pygame.event.get():
            pass


if __name__ == '__main__':
    import ObjLoader
    import sys

    render = Renderer(1024, 1024, torch.device("cuda"))
    obj = ObjLoader.OBJ(sys.argv[1], swapYZ=True)
    render.test(obj.gl_list)
    del render
