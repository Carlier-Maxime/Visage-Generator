import struct
import numpy as np
import pyrender
import trimesh
import torch
from datetime import datetime, timedelta


def read_stl(file=""):
    if file == "":
        file = input("Path file : ")
    with open(file, "rb") as f:
        print("Read file..")
        print(f.read(80).decode("utf-8"))  # Header
        nb_triangles = int.from_bytes(f.read(4), "little")

        triangles = []
        vertices = []
        time = datetime.now()
        att = timedelta(seconds=5)
        for i in range(nb_triangles):
            # triangle
            triangle = []
            normal_vector = [struct.unpack("f", f.read(4)), struct.unpack("f", f.read(4)),
                             struct.unpack("f", f.read(4))]
            for j in range(3):
                v = [struct.unpack("f", f.read(4)), struct.unpack("f", f.read(4)), struct.unpack("f", f.read(4))]
                if v in vertices:
                    triangle.append(vertices.index(v))
                else:
                    vertices.append(v)
                    triangle.append(len(vertices) - 1)
            f.read(2)  # control
            triangles.append(triangle)

            if datetime.now() > time + att:
                print(str((i / nb_triangles) * 100) + " % (" + str(i) + "/" + str(nb_triangles - 1) + ")")
                time = datetime.now()
        print("Reading finish.")
    return vertices, triangles


def read_and_view(file=""):
    vertices, triangles = read(file)
    vertices = torch.tensor(vertices).detach().cpu().numpy().squeeze()
    triangles = np.array(triangles)
    print(triangles)
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0, 1.0])
    tri_mesh = trimesh.Trimesh(vertices, faces=triangles)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene.add(mesh)
    pyrender.Viewer(scene, run_in_thread=True)


def read_obj(file=""):
    """
    read file obj, no support texture and normal.
    """
    if file == "":
        file = input("Path file : ")
    with open(file, "r") as f:
        vertices = []
        faces = []
        minFace = 1000
        while True:
            line = f.readline()
            if "#" in line:
                continue
            if line.startswith("v "):
                line = line.split(" ")
                v = [float(line[i]) for i in range(1, 4)]
                vertices.append(v)
            elif line.startswith("f "):
                line = line.split(" ")
                face = [int(line[i].split("/")[0]) for i in range(1, 4)]
                if min(face) < minFace:
                    minFace = min(face)
                faces.append(face)
            elif line == "":
                break
    if minFace > 0:
        for face in faces:
            for i in range(len(face)):
                face[i] -= 1
    return vertices, faces


def read(file=""):
    if file == "":
        file = input("Path file : ")
    if file.endswith('.stl'):
        vertices, triangles = read_stl(file)
    elif file.endswith('.obj'):
        vertices, triangles = read_obj(file)
    else:
        print("format unknown !")
        exit(1)
        vertices, triangles = None, None  # remove warning
    return vertices, triangles
