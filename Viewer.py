import pyrender
import numpy as np
import trimesh
from config import get_config
from os.path import exists
import torch

class Viewer(pyrender.Viewer):
    def __init__(self, vertice, landmark, faces, show_joints=False, show_vertices=False, show_balises = True, otherObjects = None):
        config = get_config()
        self._nbFace = config.number_faces
        self._device = config.device
        
        self._vertice = vertice
        self._landmark = landmark
        self._faces = faces
        self._show_vertices = False
        self._show_joints = False
        self._show_balises = False
        self._editBalises = False
        self._ctrl = False
        self._directionnalMatrix = []
        self._scene = pyrender.Scene()

        if otherObjects != None:
            for obj in otherObjects:
                vertices = obj[0]
                triangles = obj[1]
                vertices = torch.tensor(vertices).detach().cpu().numpy().squeeze()
                triangles = np.array(triangles)
                if len(triangles)==0:
                    sm = trimesh.creation.uv_sphere(radius=0.001)
                    sm.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
                    tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
                    tfs[:, :3, 3] = vertices
                    mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    self._scene.add(mesh)
                else:
                    tri_mesh = trimesh.Trimesh(vertices,faces=triangles)
                    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
                    self._scene.add(mesh)


        self._index = 0
        self._slcIndex = 0
        self.loadBalise()
        self.setVisage(0)
        self._scene.add_node(self._visage)

        self.genVerticesNode()
        if landmark != None : self.genJointsNode()
        self.genBalisesNode()
        

        #select balise node
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [1.0, 1.0, 0.0, 1.0]
        self._selectNode = pyrender.Node("select",mesh=pyrender.Mesh.from_trimesh(sm))

        pyrender.Viewer.__init__(self, self._scene, use_raymond_lighting=True, run_in_thread=True)

        if show_joints:
            self.showJoints()
        if show_vertices:
            self.showVertices()
        if show_balises:
            self.showBalises()      

    def on_key_press(self, symbol, modifiers):
        if self._ctrl: # Ctrl On
            if symbol == 115: # save balises (S)
                self.saveBalise()
            if symbol == 108: # load balises (L)
                self.loadBalise()
        else:
            pyrender.Viewer.on_key_press(self,symbol,modifiers)

        if symbol == 118: # show vertices (V)
            self.showVertices()
        if symbol == 98: # show balises (B)
            self.showBalises()
        if symbol == 106: # show joints (J)
            self.showJoints()
        if symbol == 101: # edit balises (E)
            self.editBalises()
        if symbol == 65507: # Ctrl
            self._ctrl = not self._ctrl
            if self._ctrl:
                self._message_text = "Ctrl On"
            else:
                self._message_text = "Ctrl Off"

        if self._editBalises:
            if symbol == 65535: # Suppr
                self.removeBalise()
            if symbol == 65293: # Enter
                self.addBalise()
            if symbol == 65362: # Up Arrow
                self.nextBalise(6)
            if symbol == 65364: # Down Arrow
                self.nextBalise(5)
            if symbol == 65361: # Left Arrow
                self.nextBalise(3)
            if symbol == 65363: # Right Arrow
                self.nextBalise(4)
            if symbol == 65365: # Up Page
                self.nextBalise(2)
            if symbol == 65366: # Down Page
                self.nextBalise(1)

    def showVertices(self):
        self.render_lock.acquire()
        if not self._show_vertices:
            self._scene.add_node(self._verticesNode)
            self._show_vertices = True
        else:
            self._scene.remove_node(self._verticesNode)
            self._show_vertices = False
        self.render_lock.release()

    def showJoints(self):
        self.render_lock.acquire()
        if not self._show_joints:
            self._scene.add_node(self._jointsNode)
            self._show_joints = True
        else:
            self._scene.remove_node(self._jointsNode)
            self._show_joints = False
        self.render_lock.release()

    def showBalises(self):
        self.render_lock.acquire()
        if len(self._tfs_balises)<=0:
            self._show_balises = False
            self.render_lock.release()
            return
        if not self._show_balises:
            self._scene.add_node(self._balisesNode)
            self._show_balises = True
        else:
            self._scene.remove_node(self._balisesNode)
            self._show_balises = False
        self.render_lock.release()

    def setVisage(self,i):
        vertices = self._vertice[i].detach().to(self._device).numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.925, 0.72, 0.519, 1.0]

        tri_mesh = trimesh.Trimesh(vertices, self._faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        self._visage = pyrender.Node("Visage",mesh=mesh)

    def on_close(self):
        if (self._index<self._nbFace-1):
            self._index = self._index+1
            self.render_lock.acquire()
            self._scene.remove_node(self._visage)
            self.setVisage(self._index)
            self._scene.add_node(self._visage)
            self.updateTfs()
            self.render_lock.release()
        else:
            pyrender.Viewer.on_close(self)
    
    def editBalises(self):
        if not self._editBalises:
            if len(self._directionnalMatrix)==0:
                self._directionnalMatrix = np.load("directionnalMatrix.npy")
            vert = self._vertice[self._index][self._directionnalMatrix[self._slcIndex][0]]
            tfs = np.tile(np.eye(4), (1, 1, 1))[0]
            tfs[:3,3] = vert
            self._scene.add_node(self._selectNode)
            self._scene.set_pose(self._selectNode,tfs)
            if not self._show_balises:
                self.showBalises()
            self._editBalises = True
            self._message_text = 'Enable edit balises'
        else:
            self._scene.remove_node(self._selectNode)
            self._editBalises = False
            self._message_text = 'Disable edit balises'

    def nextBalise(self,direction):
        i = self._directionnalMatrix[self._slcIndex][direction]
        if i==-1:
            return
        self._slcIndex = i
        vert = self._vertice[self._index][self._directionnalMatrix[self._slcIndex][0]]
        tfs = np.tile(np.eye(4), (1, 1, 1))[0]
        tfs[:3,3] = vert
        self._scene.set_pose(self._selectNode,tfs)

    def updateTfs(self):
        if self._show_vertices:
            self._scene.remove_node(self._verticesNode)
            self.genVerticesNode()
            self._scene.add_node(self._verticesNode)
        else:
            self.genVerticesNode()
        if self._show_joints:
            self._scene.remove_node(self._jointsNode)
            self.genJointsNode()
            self._scene.add_node(self._jointsNode)
        else:
            self.genJointsNode()
        if self._show_balises and len(self._tfs_balises)>0:
            self._scene.remove_node(self._balisesNode)
            self.genBalisesNode()
            self._scene.add_node(self._balisesNode)
        else:
            self.genBalisesNode()
        if self._editBalises:
            tfs = self._tfs_vertices[self._slcIndex]
            self._scene.set_pose(self._selectNode,tfs)

    def genVerticesNode(self):
        vertices = self._vertice[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.0019)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(vertices), 1, 1))
        tfs[:, :3, 3] = vertices
        vertices_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self._verticesNode = pyrender.Node("vertices",mesh=vertices_pcl)
        self._tfs_vertices = tfs

    def genJointsNode(self):
        joints = self._landmark[self._index]
        sm = trimesh.creation.uv_sphere(radius=0.0019)
        sm.visual.vertex_colors = [0.0, 0.5, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        self._jointsNode = pyrender.Node("joints",mesh=joints_pcl)
        self._tfs_joints = tfs

    def genBalisesNode(self):
        sm = trimesh.creation.uv_sphere(radius=0.002)
        sm.visual.vertex_colors = [1.0, 0.0, 1.0, 1.0]
        t = []
        vertices = self._vertice[self._index]
        for i in self._balisesIndex:
            t.append(vertices[int(i)])
        if len(t)>0:
            self._tfs_balises = np.tile(np.eye(4), (len(t), 1, 1))
            for i in range(len(t)):
                self._tfs_balises[i, :3, 3] = t[i]
            balises_pcl = pyrender.Mesh.from_trimesh(sm, poses=self._tfs_balises)
        else:
            self._tfs_balises = []
            balises_pcl = pyrender.Mesh.from_trimesh(sm)
        self._balisesNode = pyrender.Node("balises",mesh=balises_pcl)

    def addBalise(self):
        self._balisesIndex = np.append(self._balisesIndex,self._directionnalMatrix[self._slcIndex][0])
        self.updateBalise()

    def saveBalise(self):
        np.save("balises.npy",self._balisesIndex)

    def loadBalise(self):
        self._balisesIndex = np.array([],int)
        if not exists("balises.npy"):
            return
        self._balisesIndex = np.load("balises.npy")

    def removeBalise(self):
        ind=self._directionnalMatrix[self._slcIndex][0]
        for i in range(len(self._balisesIndex)) :
            if self._balisesIndex[i] == ind:
                self._balisesIndex = np.delete(self._balisesIndex,i)
                break
        self.updateBalise()

    def updateBalise(self):
        if self._show_balises:
            self._scene.remove_node(self._balisesNode)
        self.genBalisesNode()
        if not self._show_balises:
            self.showBalises()
        else:
            self._scene.add_node(self._balisesNode)
        self._message_text = "NbBalises = "+str(len(self._balisesIndex))