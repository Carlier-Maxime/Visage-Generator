import logging as log
import os
import sys

import cv2
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import render2d, aspect2d
from panda3d.core import DirectionalLight, CollisionTraverser, \
    CollisionHandlerQueue, CollisionNode, CollisionRay, GeomNode, loadPrcFile, Point3, Point2

import util


class MyApp(ShowBase):

    def __init__(self, file_path, lmks3D_path, save_path="tmp/lmks2d.npy", pyv=""):
        ShowBase.__init__(self)
        print("préparation de la scéne")
        self.file_path = file_path
        self.lmk3d_path = lmks3D_path
        self.save_path = save_path
        self.pyv = pyv
        model = self.loader.load_model(file_path)
        model.reparentTo(render)
        self.dlight = DirectionalLight('my dlight')
        dlnp = render.attachNewNode(self.dlight)
        dlnp.setPosHpr(0, 0, 500, 0, -84, 0)
        model.setLight(dlnp)
        base.disableMouse()
        base.setBackgroundColor(1, 1, 1)
        base.camera.setPosHpr(0, 0, 450, 0, -84, 0)

        # CollisionTraverser  and a Collision Handler is set up
        print("Initialisation du laser pour revenir à la 3D")
        self.picker = CollisionTraverser()
        self.picker.showCollisions(render)
        self.pq = CollisionHandlerQueue()

        self.pickerNode = CollisionNode('mouseRay')
        self.pickerNP = camera.attachNewNode(self.pickerNode)
        self.pickerNode.setFromCollideMask(GeomNode.getDefaultCollideMask())
        self.pickerRay = CollisionRay()
        self.pickerNode.addSolid(self.pickerRay)
        self.picker.addCollider(self.pickerNP, self.pq)

        taskMgr.doMethodLater(0, self.screenshotTask, 'screenshot')

    def screenshotTask(self, task):
        base_name = self.file_path.split(".obj")[0].split(".stl")[0]
        print("screenshot de l'aperçu de la scene")
        base.screenshot(f"{base_name}.png", False)

        print("Transformation des landmark 3D en landmark 2D..")
        lmks3d = np.load(self.lmk3d_path)
        lmks2d = []
        for lmk3d in lmks3d:
            lmk2d = self.Coord3dIn2d(Point3(lmk3d[0], lmk3d[1], lmk3d[2]))
            lmk2d = [lmk2d[0], lmk2d[1]]
            lmks2d.append(lmk2d)

        print("Sauvegarde des landmarks")
        np.save(self.save_path, lmks2d)
        self.finalizeExit()
        return task.done

    def Coord3dIn2d(self, coord3D):
        coord2d = Point2()
        base.camLens.project(coord3d, coord2d)
        coordInRender2d = Point3(coord2d[0], 0, coord2d[1])
        coordInAspect2d = aspect2d.getRelativePoint(render2d,
                                                    coordInRender2d)
        return coordInAspect2d


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
