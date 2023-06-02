import pygame
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import PIL.Image
import torch


class Renderer:
    def __init__(self, width: int, height: int, device: torch.device, show: bool = True, camera=None):
        if camera is None:
            camera = [10, 0, 0, -2, 0, 0, 0]
        self.device = torch.device(device)
        render_data = torch.load('render_data.pt')
        self.uvcoords = render_data['uvcoords'].to(self.device)
        self.uvfaces = render_data['uvfaces'].to(self.device)
        self.order_indexs = render_data['order_indexs'].to(self.device)
        self.raw_sphere = self.create_sphere(0.002, 30, 30)

        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_icon(pygame.image.load('logo.png'))
        pygame.display.set_mode([width, height], pygame.constants.OPENGL | pygame.constants.DOUBLEBUF | pygame.SHOWN if show else pygame.HIDDEN)

        glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)

        self.rotate = False
        self.rotate_z = False
        self.move = False
        self.fov, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz = camera

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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.gl_list_visage = glGenLists(1)

        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._update_camera()

    def __del__(self):
        glDisable(GL_TEXTURE_2D)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDeleteBuffers(3, self.buffers)
        glDeleteLists(self.gl_list_visage, 1)

    def _update_camera(self):
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

    @staticmethod
    def _change_gl_texture(texture):
        size = texture.shape
        image = PIL.Image.fromarray(texture, 'RGB').transpose(PIL.Image.FLIP_TOP_BOTTOM).tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size[0], size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    def _create_gl_list(self, vertices, triangles):
        vertices = vertices[self.order_indexs[:, 1]]
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
        vertices = vertices[self.order_indexs[:, 1]]
        vertices = vertices.cpu().numpy()

        glNewList(self.gl_list_visage, GL_COMPILE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[1])
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STREAM_DRAW)
        glVertexPointer(3, GL_FLOAT, 0, None)
        if texture is not None:
            self._change_gl_texture(texture)
        else:
            glDisable(GL_TEXTURE_2D)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[0])
        glDrawElements(GL_TRIANGLES, self.uvfaces.numel(), GL_UNSIGNED_INT, None)
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
        PIL.Image.frombytes('RGB', (self.width, self.height), glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)).transpose(PIL.Image.FLIP_TOP_BOTTOM).save("test.jpg")

    def _poll_event(self, e):
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            return 0
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4:
                self.tz = max(1, self.tz - 0.1)
                self._update_camera()
            elif e.button == 5:
                self.tz += 0.1
                self._update_camera()
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
                self._update_camera()
            if self.rotate_z:
                self.rz += i
                self._update_camera()
            if self.move:
                self.tx += i / 256
                self.ty -= j / 256
                self._update_camera()
        elif e.type == KEYDOWN:
            if e.key == K_c:
                print(f'fov: {self.fov}, tx: {self.tx}, ty: {self.ty}, tz: {self.tz}, rx: {self.rx}, ry: {self.ry}, rz: {self.rz}')
            if e.key == K_KP_MINUS:
                self.fov -= 0.1
                self._update_camera()
            if e.key == K_KP_PLUS:
                self.fov += 0.1
                self._update_camera()
        return 1

    def _poll_events(self):
        for e in pygame.event.get():
            if not self._poll_event(e):
                return 0
        return 1

    def _change_camera(self, camera):
        self.fov, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz = camera
        self._update_camera()

    def _render(self, gl_lists, camera=None):
        if camera is not None:
            self._change_camera(camera)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glCallLists(gl_lists)
        pygame.display.flip()

    def random_background(self):
        glClearColor(*torch.rand(3, device=self.device).tolist(), 1.)

    def save_to_image(self, filename, vertices, texture, pts=None, pts_in_alpha: bool = True, camera=None):
        self._edit_gl_list(vertices, texture)
        if pts is not None:
            pts_gl_list = Renderer.create_spheres_gl_list(self.raw_sphere, pts)
            if pts_in_alpha:
                self._render([self.gl_list_visage], camera)
                img_visage = torch.frombuffer(bytearray(glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glColor(0., 0., 0.)
                glCallList(self.gl_list_visage)
                glColor(1., 1., 1.)
                glCallList(pts_gl_list)
                pygame.display.flip()
                img_pts = torch.frombuffer(bytearray(glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
                green_mask = (img_pts == torch.tensor([0, 255, 0, 255], device=self.device)).all(dim=2)
                new_colors = img_visage[green_mask]
                new_colors[:, 3] = 0
                img_visage[green_mask] = new_colors
            else:
                self._render([self.gl_list_visage, pts_gl_list], camera)
                img_visage = torch.frombuffer(bytearray(glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
            glDeleteLists(pts_gl_list, 1)
        else:
            self._render([self.gl_list_visage], camera)
            img_visage = torch.frombuffer(bytearray(glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)), dtype=torch.uint8).to(self.device).view(self.height, self.width, 4)
        PIL.Image.fromarray(img_visage.cpu().numpy()).transpose(PIL.Image.FLIP_TOP_BOTTOM).save(filename)

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

    @staticmethod
    def create_spheres_gl_list(raw_sphere, positions, list_id=None, color=None):
        if color is None:
            color = [0., 255., 0.]
        vertex_array, indices = raw_sphere
        nb_spheres = positions.shape[0]
        indices = indices.clone().flatten()
        a = torch.arange(0, vertex_array.shape[0] * nb_spheres, vertex_array.shape[0], device=indices.device)
        indices = (indices.reshape([indices.shape[0] // 3, 3]).repeat(nb_spheres, 1, 1).permute(1, 2, 0) + a).permute(2, 0, 1)
        vertex_array = (vertex_array.clone().repeat(nb_spheres, 1, 1).permute(1, 0, 2) + positions).permute(1, 0, 2)
        if list_id is None:
            list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)
        glDisable(GL_TEXTURE_2D)
        glColor(color[0], color[1], color[2])
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

    @staticmethod
    def get_coord_2d(point3d):
        viewport = glGetIntegerv(GL_VIEWPORT)
        model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        win_x, win_y, win_z = gluProject(point3d[0], point3d[1], point3d[2], model_view, projection_matrix, viewport)
        if 0 <= win_z <= 1:
            return int(win_x), int(win_y)
        return None

    def get_rotation_matrix(self, rx, ry, rz):
        rotation_x = torch.deg2rad(rx)
        rotation_y = torch.deg2rad(ry)
        rotation_z = torch.deg2rad(rz)

        rotation_matrix_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(rotation_x), -torch.sin(rotation_x)],
            [0, torch.sin(rotation_x), torch.cos(rotation_x)]
        ], dtype=torch.float32, device=self.device)

        rotation_matrix_y = torch.tensor([
            [torch.cos(rotation_y), 0, torch.sin(rotation_y)],
            [0, 1, 0],
            [-torch.sin(rotation_y), 0, torch.cos(rotation_y)]
        ], dtype=torch.float32, device=self.device)

        rotation_matrix_z = torch.tensor([
            [torch.cos(rotation_z), -torch.sin(rotation_z), 0],
            [torch.sin(rotation_z), torch.cos(rotation_z), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        return torch.matmul(torch.matmul(rotation_matrix_z, rotation_matrix_y), rotation_matrix_x)

    def get_camera_matrices(self, camera: torch.Tensor):
        # Conversion de l'angle de champ (fov) en focale
        fov, tx, ty, tz, rx, ry, rz = camera
        focal_length = (self.width / 16.) / torch.tan(torch.deg2rad(fov / 2.0))

        # Calcul de la matrice intrinsèque normalisé
        intrinsic_matrix = torch.tensor([
            [focal_length / self.width, 0, 0.5],
            [0, focal_length / self.height, 0.5],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Calcul de la matrice d'extrinsèques
        rotation_matrix = self.get_rotation_matrix(rx, ry, rz)
        translation = torch.tensor([tx, ty, tz], dtype=torch.float32, device=self.device)

        extrinsic_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation

        return intrinsic_matrix, extrinsic_matrix

    def get_camera(self):
        return torch.tensor([self.fov, self.tx, self.ty, self.tz, self.rx, self.ry, self.rz], device=self.device)


if __name__ == '__main__':
    import ObjLoader
    import sys

    render = Renderer(1024, 1024, torch.device("cuda"))
    obj = ObjLoader.OBJ(sys.argv[1], swapyz=True)
    render.test(obj.gl_list)
