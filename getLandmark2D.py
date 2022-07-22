import logging as log
import os
import sys

import cv2
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, CollisionTraverser, \
    CollisionHandlerQueue, CollisionNode, CollisionRay, GeomNode, loadPrcFile

import util


class MyApp(ShowBase):

    def __init__(self, file_path, lmks3D_path, pyv):
        ShowBase.__init__(self)
        print("préparation de la scéne")
        self.file_path = file_path
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
        # A FAIRE !!!

        print("Sauvegarde des landmarks")
        # A FAIRE !!!
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
    if len(args) < 3:
        args.append("")
    app = MyApp(str(args[0]), str(args[1]), str(args[2]))
    app.run()
