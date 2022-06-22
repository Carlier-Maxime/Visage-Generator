import struct
import numpy as np
import pyrender
import trimesh
import torch
from datetime import datetime,timedelta

def readSTL(file=""):
    if file == "":
        file = input("Path file : ")
    with open(file,"rb") as f:
        print("Lecture du fichier..")
        print(f.read(80).decode("utf-8")) # Header
        nbTriangles = int.from_bytes(f.read(4),"little")

        triangles = []
        vertices = []
        perc = 0
        time = datetime.now()
        att = timedelta(seconds=5)
        for i in range(nbTriangles):
            #triangle
            triangle = []
            normalVector = [struct.unpack("f",f.read(4)),struct.unpack("f",f.read(4)),struct.unpack("f",f.read(4))]  
            for j in range(3):
                v = [struct.unpack("f",f.read(4)),struct.unpack("f",f.read(4)),struct.unpack("f",f.read(4))]
                if v in vertices:
                    triangle.append(vertices.index(v))
                else:
                    vertices.append(v)
                    triangle.append(len(vertices)-1)
            controle = f.read(2)
            triangles.append(triangle)

            if datetime.now()>time+att:
                print(str((i/nbTriangles)*100)+" % ("+str(i)+"/"+str(nbTriangles-1)+")")
                time = datetime.now()
        print("Lecture terminee.")
    return vertices, triangles

def readAndView(file=""):
    vertices, triangles = read(file)
    vertices = torch.tensor(vertices).detach().cpu().numpy().squeeze()
    triangles = np.array(triangles)
    print(triangles)
    scene = pyrender.Scene()
    tri_mesh = trimesh.Trimesh(vertices,faces=triangles)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

def readOBJ(file=""):
    """
    read file obj, no suppport texture and normal.
    """
    if file == "":
        file = input("Path file : ")
    with open(file,"r") as f:
        vertices = []
        faces = []
        minFace = 1000
        while True:
            line = f.readline()
            if "#" in line:
                continue
            if line.startswith("v "):
                line = line.split(" ")
                v = [float(line[i]) for i in range(1,4)]
                vertices.append(v)
            elif line.startswith("f "):
                line = line.split(" ")
                face = [int(line[i].split("/")[0]) for i in range(1,4)]
                if min(face)<minFace:
                    minFace=min(face)
                faces.append(face)
            elif line=="":
                break
    if minFace>0:
        for face in faces:
            for i in range(len(face)):
                face[i]-=1
    return vertices,faces

def read(file=""):
    if file == "":
        file = input("Path file : ")
    if file.endswith('.stl'):
        vertices, triangles = readSTL(file)
    elif file.endswith('.obj'):
        vertices, triangles = readOBJ(file)
    else:
        print("format unknow !")
        exit(1)
    return vertices, triangles