import random
from typing import Tuple, List, Any

import numpy as np
import torch
import os
from skimage.io import imsave


def get_vertex_markers(vertex: list, min_x: float = 0.05, min_z: float = -0.03, max_z: float = 0.15,
                       min_eye_distance: float = 0.0139, min_noise_distance: float = 0.0130) \
        -> tuple[list[list[Any]], list[list[int]]]:
    """
    This function allows to remove the vertices present in the eyes and noise and 
    what are not included in the default zone of the markers.
    Args:
        vertex (list): array of all vertices for all face
        min_x (float): minimum value for X coordinates of vertices
        min_z (float): minimum value for Z coordinates of vertices
        max_z (float): maximum value for Z coordinates of vertices
        min_eye_distance (float): minimum eye distance
        min_noise_distance (float): minimum noise distance
    Return:
        vertex array respecting the conditions, index correspondence
    """
    eye_l = [0.108, -0.178, 0.085]
    eye_r = [0.108, -0.1155, 0.085]
    noise = [0.143, -0.1464, 0.055]

    t = []
    m = []
    for vertices in vertex:
        line = []
        li = []
        for i in range(len(vertices)):
            p = vertices[i]
            in_zone = vertices[i][0] > min_x and min_z < vertices[i][2] < max_z
            dist_eye_l = np.sqrt((p[0] - eye_l[0]) ** 2 + (p[1] - eye_l[1]) ** 2 + (p[2] - eye_l[2]) ** 2)
            dist_eye_r = np.sqrt((p[0] - eye_r[0]) ** 2 + (p[1] - eye_r[1]) ** 2 + (p[2] - eye_r[2]) ** 2)
            dist_noise = np.sqrt((p[0] - noise[0]) ** 2 + (p[1] - noise[1]) ** 2 + (p[2] - noise[2]) ** 2)
            if in_zone and dist_eye_l > min_eye_distance and dist_eye_r > min_eye_distance \
                    and dist_noise > min_noise_distance:
                line.append(vertices[i])
                li.append(i)
        t.append(line)
        m.append(li)
    return t, m


def gen_vertices_file(vertices: list) -> None:
    """
    Generate vertices file for by given vertices.
    The vertices file is numpy file and contains array : [vertex1, vertex2, ..., vertexN].
    One vertex is represented by [index in the original array, x, y, z, index of nearest vertex, distance]
    Args:
        vertices (list): array of all vertices
    Returns: None
    """
    t = []
    for i in range(len(vertices)):
        t.append([i, float(vertices[i][0]), float(vertices[i][1]), float(vertices[i][2]), -1, -1])
    t = calc_distance_vertices(t, True)
    np.save("vertices.npy", t)


def full_random_markers() -> None:
    """
    generate a numpy array containing 150 random marker indexes and save it to a numpy file.
    Args: None
    Return: None
    """
    t = []
    for i in range(5023):
        t.append(i)
    random.shuffle(t)
    markers = []
    for i in range(150):
        markers.append(t[i])

    np.save("markers.npy", markers)


def delete_markers(n) -> None:
    """
    Delete the N markers having the nearest neighbors
    Args:
        n: the number of markers to delete
    Returns: None
    """
    t = load_vertices()
    li = []
    t2 = []
    for i in range(n):
        if t[i][4] not in li:
            li.append(i)
        else:
            t2.append(t[i])
    for i in range(n, len(t)):
        t2.append(t[i])
    t = calc_distance_vertices(t2)
    save_vertices(t)


def load_vertices() -> np.array:
    """
    Load numpy file containing vertices
    Returns: numpy array
    """
    return np.load("vertices.npy")


def calc_distance_vertices(t: list, verbose: bool = False) -> list:
    """
    Calcul distance of all vertices provided
    Args:
        t: vertices distance array : [vertex1, vertex2, ..., vertexN].
            One vertex is represented by [index in the original array, x, y, z, index of nearest vertex, distance]
        verbose (bool): prompt progress
    Returns: vertices distance array : [vertex1, vertex2, ..., vertexN].
    """
    for i in range(len(t)):
        distance = 0
        index = -1
        for j in range(len(t)):
            if i == j:
                continue
            dist = np.sqrt((t[j][1] - t[i][1]) ** 2 + (t[j][2] - t[i][2]) ** 2 + (t[j][3] - t[i][3]) ** 2)
            if index == -1 or dist < distance:
                distance = dist
                index = t[j][0]
        t[i][4] = index
        t[i][5] = distance
        if verbose:
            print(str(i + 1) + "/" + str(len(t)))
    t.sort(key=lambda key: t[5])
    return t


def save_vertices(t) -> None:
    """
    Save vertices in numpy file
    Args:
        t: numpy array containing the vertices
    Returns: None
    """
    np.save("vertices.npy", t)


def select_random_markers() -> None:
    """
    Selection random markers and save in numpy array.
    Returns: None
    """
    t = load_vertices()
    markers = []
    for e in t:
        markers.append(e[0])
    random.shuffle(markers)
    markers = markers[0:150]

    np.save('markers.npy', markers)


def gen_directional_matrix(vertices: list, index_list: list) -> None:
    """
    Generate directional matrix and save in numpy file.
    Args:
        vertices: array of all vertex : [vertex1, vertex2, vertexN], one vertex is array : [x, y , z]
        index_list: the list of veritable index for the vertices
            useful when, for example, you provided an array with fewer vertices than the total number of vertices
    Returns: None
    """
    m = []
    for i in range(len(vertices)):
        index_d = [-1, -1, -1, -1, -1, -1]
        dist_d = [-1, -1, -1, -1, -1, -1]
        coo = vertices[i]
        for j in range(len(vertices)):
            if i == j:
                continue
            coo2 = vertices[j]
            dist = np.sqrt((coo2[0] - coo[0]) ** 2 + (coo2[1] - coo[1]) ** 2 + (coo2[2] - coo[2]) ** 2)
            ind = get_index_for_d(index_d, dist_d, dist, coo, coo2)
            if ind != -1:
                index_d[ind] = j
                dist_d[ind] = dist
        index_d.insert(0, index_list[i])
        m.append(index_d)
        print(str(i) + "/" + str(len(vertices) - 1))
    np.save("directionalMatrix.npy", m)


def get_index_for_d(index_d: list, dist_d: list, dist: float, coo: list, coo2: list) -> int:
    """
    get the index of the corresponding direction,
    starting from the point of coordinates coo towards the point of coordinates coo2.
    Args:
        index_d (list): list of all index direction
        dist_d (list): list of distances from all direction
        dist (float): distance between point coo and point coo2
        coo (list): coordinate for start point
        coo2 (list): coordinate for destination point
    Returns (int): index of direction
    """
    assert len(index_d) == len(dist_d)
    diff = [coo2[0] - coo[0],
            coo2[1] - coo[1],
            coo2[2] - coo[2]]

    for i in range(len(index_d)):
        n = diff[int(i / 2)]
        if max(np.abs(diff)) == np.abs(n) and (index_d[i] == -1 or dist < dist_d[i]):
            if i % 2 == 0 and n < 0:
                return i
            elif i % 2 != 0 and n > 0:
                return i
    return -1


def save_faces(vertex: list) -> None:
    """
    save all points corresponding to markers for all faces provided by vertex
    Args:
        vertex (list): array of all vertices for all faces
    Returns: None
    """
    markers = np.load("markers.npy")
    data = np.zeros([len(vertex), len(markers), 3])
    for i in range(len(vertex)):
        for j in range(len(markers)):
            data[i, j, :] = vertex[i][markers[j]]
    np.save("data.npy", data)


def get_index_for_match_points(vertices: list, faces: list, points: list, verbose: bool = False,
                               triangle_optimize: bool = True, pas_tri: int = 1000):
    """
    Obtain the clues that best correspond to the points provided.
    it can be vertex indexes or more complex indexes depending on the triangle_optimize value.

    Optimization with triangles returns indexes which are arrays containing different information:
    the format of index : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]
    format detail :
        - index_vertex: index of vertex
        - index_triangle: index of triangle
        - percentage_vector_1: percentage of vector one
        - percentage_vector_2: percentage of vector two
    vector 1 and 2 are calculated using as origin the vertex of index index_vertex and as
    destination one of the other points of the triangle.
    the destination point is obtained with respect to the order of the vertices in the triangle,
    by not tacking the original vertices. the list of possibility :
    - [origin, dest1, dest2]
    - [dest1, origin, dest2]
    - [dest1, dest2, origin]

    Args:
        vertices (list): array of all vertex
        faces (list): array of all face
        points (list): array of all points
        verbose (bool): enable prompt progress
        triangle_optimize (bool): enable optimize by triangle (index to place the point in the triangle)
        pas_tri (int): pas used for precision in placement in triangle
    Returns: list index of index matching points.
    """
    assert len(vertices) > 0

    list_index = []
    no_tri = 0
    for ind in range(len(points)):
        if verbose:
            print(ind, "/", len(points) - 1, "points")
        p = points[ind]
        index = -1
        dist = -1
        v = []
        for i in range(len(vertices)):
            v = vertices[i]
            d = np.sqrt((v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2 + (v[2] - p[2]) ** 2)
            if dist == -1 or d < dist:
                index = i
                dist = d
        if triangle_optimize:
            index_triangles = get_index_triangles_match_vertex(faces, index)
            triangles = np.array(faces)[index_triangles]
            triangles = np.array(vertices)[triangles]
            vectors = get_vector_for_point(triangles, vertices[index])
            ind_vect = -1
            percentage = [0, 0]
            dist2 = dist
            for i in range(len(vectors)):
                vect = np.array(vectors[i]) / pas_tri
                perc = []
                distance = dist
                vert = vertices[index]
                for j in range(len(vect)):
                    pas = 0
                    for k in range(1, pas_tri):
                        v = vert + vect[j] * k
                        d = np.sqrt((v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2 + (v[2] - p[2]) ** 2)
                        if d <= distance:
                            pas = k
                            distance = d
                        else:
                            break
                    perc.append(pas / pas_tri)
                    vert = v
                if distance < dist2:
                    dist2 = distance
                    ind_vect = i
                    percentage = perc
            if ind_vect == -1:
                ind_tri = -1
                no_tri += 1
            else:
                ind_tri = index_triangles[ind_vect]
            list_index.append([index, ind_tri, percentage[0], percentage[1]])
        else:
            list_index.append(index)
    print(no_tri, "/", len(points), " points no need triangles !")
    return list_index


def get_index_triangles_match_vertex(triangles: list, index_point: int) -> list:
    """
    Obtain all index of triangles contain the vertex given by index point
    Args:
        triangles: array of all faces.
        index_point: index point for the vertex

    Returns: all index of triangles contain vertex
    """
    faces = []
    for i in range(len(triangles)):
        if index_point in triangles[i]:
            faces.append(i)
    return faces


def get_vector_for_point(triangles: list, p: list) -> list:
    """
    Obtain all vector for all triangles.
    Args:
        triangles (list): triangle array => triangle = coo triangle [x,y,z]
        p (list): coo point
    Returns: array of all vector for all triangles
    """
    vectors = []
    for triangle in triangles:
        t = [0, 0, 0]
        ind = []
        for i in range(3):
            if p[0] == triangle[i][0] and p[1] == triangle[i][1] and p[2] == triangle[i][2]:
                t[0] = triangle[i]
            else:
                ind.append(i)
        if len(ind) > 2:
            print("Error in getVectoForPoint in util.py ! (verify your point is vertex of triangles)")
            exit(1)
        t[1] = triangle[ind[0]]
        t[2] = triangle[ind[1]]

        v = []
        for i in range(1, 3):
            v.append([t[i][0] - p[0], t[i][1] - p[1], t[i][2] - p[2]])
        vectors.append(v)
    return vectors


def read_index_opti_tri(vertices: list, faces: list, index_opti_tri: list) -> list:
    """
    retrieve the index point of the type : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]
    Args:
        vertices (list): the array of all vertex (one vertex is a point represented in an array : [x, y, z])
        faces (list): the array of all face / triangle (one face is an array containing 3 index of vertex : [i, j, k])
        index_opti_tri (list): A index of type : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]

    Returns: coordinate point for index provided
    """
    p = vertices[int(index_opti_tri[0])].cpu().numpy()
    if index_opti_tri[1] != -1:
        tri = vertices.cpu().numpy()[faces[int(index_opti_tri[1])]]
        tmp = get_vector_for_point([tri], p)[0]
        vectors = torch.tensor(tmp).cpu().numpy()
        p = p + vectors[0] * index_opti_tri[2]
        p = p + vectors[1] * index_opti_tri[3]
    return p


def read_all_index_opti_tri(vertices: list, faces: list, indexs_opti_tri: list) -> list:
    """
    Read all index of the type : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]
    Args:
        vertices (list): the array of all vertex (one vertex is a point represented in an array : [x, y, z])
        faces (list): the array of all face / triangle (one face is an array containing 3 index of vertex : [i, j, k])
        indexs_opti_tri (list): array of all indexs. type index is :
            [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]

    Returns: array of all points. point represented by coordinate : [X, Y, Z]
    """
    points = []
    for indexOptiTri in indexs_opti_tri:
        points.append(read_index_opti_tri(vertices, faces, indexOptiTri))
    return points


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = face_vertices(vertices, faces)

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = torch.nn.functional.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def gen_lst() -> None:
    """
    Generate list from used for training SAN (https://github.com/D-X-Y/landmark-detection/tree/master/SAN)
    Returns: None
    """
    with open('tmp/result.lst', 'w') as f:
        os.chdir('output')
        for (folder, sub_folder, files) in os.walk('.'):
            for i in range(0, len(files), 2):
                f.write(f'{os.path.abspath(files[i])} {os.path.abspath(files[i + 1])} 200 100 600 500\n')
    os.chdir('..')
