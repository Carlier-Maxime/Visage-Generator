import logging as log
import os
import sys

import cv2
import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import render2d, aspect2d
from panda3d.core import DirectionalLight, CollisionTraverser, \
    CollisionHandlerQueue, CollisionNode, CollisionRay, GeomNode, loadPrcFile, Point3, Point2, LVecBase2f

import util


class MyApp(ShowBase):

    def __init__(self, file_path, lmks3D_path, save_path="tmp/lmks2d.npy", pyv=""):
        ShowBase.__init__(self)
        print("préparation de la scéne")
        self.file_path = file_path
        self.lmk3d_path = lmks3D_path
        self.save_path = save_path
        self.pyv = pyv
        self.model = self.loader.load_model(file_path)
        self.model.reparentTo(render)
        self.dlight = DirectionalLight('my dlight')
        dlnp = render.attachNewNode(self.dlight)
        dlnp.setPosHpr(0, 0, 10, 90, -45, 0)
        self.model.setLight(dlnp)
        base.disableMouse()
        base.setBackgroundColor(1, 1, 1)
        base.camera.setPosHpr(0.77, -0.18, 0.86, 90, -45, 0)
        base.camLens.setFov(LVecBase2f(26.3201, 17))
        taskMgr.doMethodLater(0, self.screenshotTask, 'screenshot')

    def enable_mouve_camera(self):
        self.accept('q', self.move, [0, 0.01, 0, 0, 0, 0, 0])
        self.accept('d', self.move, [0, -0.01, 0, 0, 0, 0, 0])
        self.accept('s', self.move, [0.01, 0, 0, 0, 0, 0, 0])
        self.accept('z', self.move, [-0.01, 0, 0, 0, 0, 0, 0])
        self.accept('a', self.move, [0, 0, 0.01, 0, 0, 0, 0])
        self.accept('e', self.move, [0, 0, -0.01, 0, 0, 0, 0])
        self.accept('y', self.move, [0, 0, 0, 1, 0, 0, 0])
        self.accept('h', self.move, [0, 0, 0, -1, 0, 0, 0])
        self.accept('p', self.move, [0, 0, 0, 0, 1, 0, 0])
        self.accept('m', self.move, [0, 0, 0, 0, -1, 0, 0])
        self.accept('r', self.move, [0, 0, 0, 0, 0, 1, 0])
        self.accept('f', self.move, [0, 0, 0, 0, 0, -1, 0])
        self.accept('i', self.move, [0, 0, 0, 0, 0, 0, -1])
        self.accept('o', self.move, [0, 0, 0, 0, 0, 0, 1])
        self.accept('c', self.ShowCamPos)
        self.accept('c-up', self.HideCamPos)

    def move(self, x, y, z, h, p, r, fov):
        base.camera.setX(base.camera.getX() + x)
        base.camera.setY(base.camera.getY() + y)
        base.camera.setZ(base.camera.getZ() + z)
        base.camera.setH(base.camera.getH() + h)
        base.camera.setP(base.camera.getP() + p)
        base.camera.setR(base.camera.getR() + r)
        base.camLens.setFov(base.camLens.getFov()+fov)

    def ShowCamPos(self):
        x = base.camera.getX()
        y = base.camera.getY()
        z = base.camera.getZ()
        h = base.camera.getH()
        p = base.camera.getP()
        r = base.camera.getR()
        fov = base.camLens.getFov()
        self.title = OnscreenText(text=str(x) + " : " + str(y) + " : " + str(z) + "\n" + str(h) + " : " + str(p) + " : " + str(r) + " : " + str(fov),
                                  style=1, fg=(1, 1, 0, 1), pos=(0, 0), scale=0.07)

    def HideCamPos(self):
        self.title.destroy()

    def screenshotTask(self, task):
        base_name = self.file_path.split(".obj")[0].split(".stl")[0]
        print("screenshot de l'aperçu de la scene")
        base.screenshot(f"{base_name}.png", False)

        print("Transformation des landmark 3D en landmark 2D..")
        lmks3d = np.load(self.lmk3d_path)
        lmks2d = []
        for lmk3d in lmks3d:
            lmk2d = self.Coord3dIn2d(self.model, Point3(lmk3d[0], lmk3d[1], lmk3d[2]))
            x, y = lmk2d[0], lmk2d[1]
            halfX = base.win.getXSize() / 2
            halfY = base.win.getYSize() / 2
            x = x * halfX + halfX
            y = y * -1 * halfY + halfY
            lmks2d.append([int(x), int(y)])

        print("Sauvegarde des landmarks")
        np.save(self.save_path, lmks2d)
        self.finalizeExit()
        return task.done

    def Coord3dIn2d(self, nodePath, point=Point3(0, 0, 0)):
        """ Computes a 3-d point, relative to the indicated node, into a
        2-d point as seen by the camera.  The range of the returned value
        is based on the len's current film size and film offset, which is
        (-1 .. 1) by default. """

        # Convert the point into the camera's coordinate space
        p3d = base.cam.getRelativePoint(nodePath, point)

        # Ask the lens to project the 3-d point to 2-d.
        p2d = Point2()
        if base.camLens.project(p3d, p2d):
            # Got it!
            return p2d

        # If project() returns false, it means the point was behind the
        # lens.
        return None


if __name__ == '__main__':
    print("Configuration de panda3d")
    loadPrcFile("etc/Config.prc")
    args = sys.argv[1:]
    if len(args) < 4:
        if len(args) < 3:
            args.append("")
        args.append("")
    app = MyApp(str(args[0]), str(args[1]), str(args[2]), str(args[3]))
    app.run()
