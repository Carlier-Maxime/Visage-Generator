import os
import random
from typing import Any

import numpy as np
import torch


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
    generate a torch array containing 150 random marker indexes and save it to a torch file.
    Args: None
    Return: None
    """
    torch.save(torch.randperm(5023)[:150], "markers.pt")


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


def load_vertices() -> torch.Tensor:
    """
    Load torch file containing vertices
    Returns: torch tensor
    """
    return torch.load("vertices.pt")


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
    Selection random markers and save in torch array.
    Returns: None
    """
    markers = load_vertices()[:, 0]
    markers = markers[torch.randperm(len(markers))][:150]
    torch.save(markers, 'markers.pt')


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


def save_faces(vertex: torch.Tensor) -> None:
    """
    save all points corresponding to markers for all faces provided by vertex
    Args:
        vertex (list): array of all vertices for all faces
    Returns: None
    """
    markers = torch.load("markers.pt").to(vertex.device)
    data = vertex[torch.arange(len(vertex))][markers[torch.arange(len(markers))]]
    torch.save(data, "data.pt")


def get_index_for_match_points(vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor, verbose: bool = False, triangle_optimize: bool = True, nb_step: int = 1000, lr: float = 0.1):
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
        vertices (Tensor): array of all vertex
        faces (Tensor): array of all face
        points (Tensor): array of all points
        verbose (bool): enable prompt progress
        triangle_optimize (bool): enable optimize by triangle (index to place the point in the triangle)
        nb_step (int): number step used for optimize percentage of vector
        lr (float): learning rate used for optimize percentage of vector
    Returns: list index of index matching points.
    """
    assert len(vertices) > 0

    ind = (vertices.repeat(len(points), 1, 1).permute(1, 0, 2) - points).square().sum(dim=2).permute(1, 0).min(dim=1).indices
    if not triangle_optimize:
        return ind
    faces = faces.repeat(len(points), 1, 1)
    tmp = (faces.permute(1, 2, 0) == ind).permute(2, 0, 1)
    mask = tmp.any(dim=2)
    tri = torch.arange(faces.shape[1], device=faces.device).repeat(105, 1)[mask]
    vec = vertices[faces[mask][~tmp[mask]]]
    vec = vec.reshape(vec.shape[0] // 2, 2, 3).permute(1, 0, 2)
    ind_repeat = torch.arange(len(points), device=mask.device).repeat_interleave(mask.sum(dim=1))
    pts = vertices[ind][ind_repeat]
    vec = (vec - pts).permute(1, 0, 2)
    points = points[ind_repeat]
    coeff = torch.zeros(len(vec), 2, dtype=torch.float32, requires_grad=True, device=pts.device)
    optimizer = torch.optim.Adam([coeff], lr=lr)
    pts = pts.permute(1, 0)
    points = points.permute(1, 0)
    vec = vec.permute(1, 2, 0)

    def calc_dist(coeff, pts, vec, points):
        coeff = coeff.clamp(0, 1)
        pts = pts + vec[0] * coeff[:, 0] + vec[1] * coeff[:, 1]
        return (pts - points).square().permute(1, 0).sum(dim=1).sqrt()

    from tqdm import trange
    for _ in trange(nb_step, desc='Opti Coeff Vector', unit='step', disable=not verbose):
        optimizer.zero_grad()
        loss = calc_dist(coeff, pts, vec, points).sum()
        loss.backward()
        optimizer.step()
    del optimizer
    coeff = coeff.requires_grad_(False).clamp(0, 1)

    dist = calc_dist(coeff, pts, vec, points)
    i = 0
    index = []
    for group_size in mask.sum(dim=1):
        index.append(int(dist[i:i + group_size].argmin()) + i)
        i += group_size
    index = torch.tensor(index)
    coeff = coeff[index]
    tri = tri[index]
    no_need_tri = int((coeff == torch.tensor([0, 0], device=coeff.device)).all(dim=1).sum())
    print(no_need_tri, "/", len(tri), " points no need triangles !")
    return torch.stack([ind, tri, coeff[:, 0], coeff[:, 1]]).permute(1, 0)


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


def get_vector_for_point(triangles: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Obtain all vector for all triangles.
    Args:
        triangles (list): triangle array => triangle = coo triangle [x,y,z]
        p (list): coo point
    Returns: array of all vector for all triangles
    """
    mask = (triangles == p).all(dim=2)
    assert mask.any(dim=1).all(), "Error in getVectoForPoint in util.py ! (verify your point is vertex of triangles)"
    pts = triangles[mask]
    assert pts.shape[0] == triangles.shape[0], "Error in getVectoForPoint in util.py ! (point must be equal one vertex per triangle)"
    vec = triangles[~mask].reshape(triangles.shape[0], 2, 3)
    return vec - pts


def read_index_opti_tri(vertices: torch.Tensor, faces: torch.Tensor, index_opti_tri: list) -> torch.Tensor:
    """
    retrieve the index point of the type : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]
    Args:
        vertices (list): the array of all vertex (one vertex is a point represented in an array : [x, y, z])
        faces (list): the array of all face / triangle (one face is an array containing 3 index of vertex : [i, j, k])
        index_opti_tri (list): A index of type : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]

    Returns: coordinate point for index provided
    """
    p = vertices[int(index_opti_tri[0])]
    if index_opti_tri[1] != -1:
        tri = vertices[faces[int(index_opti_tri[1])]]
        vectors = get_vector_for_point(tri[None], p)[0]
        p = p + vectors[0] * index_opti_tri[2] + vectors[1] * index_opti_tri[3]
    return p


def read_all_index_opti_tri(vertices: torch.Tensor, faces: torch.Tensor, indices_opti_tri: torch.Tensor | None) -> torch.Tensor | None:
    """
    Read all index of the type : [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]
    Args:
        vertices (list): the array of all vertex (one vertex is a point represented in an array : [x, y, z])
        faces (list): the array of all face / triangle (one face is an array containing 3 index of vertex : [i, j, k])
        indices_opti_tri (list): array of all indexes. type index is :
            [index_vertex, index_triangle, percentage_vector_1, percentage_vector_2]

    Returns: array of all points. point represented by coordinate : [X, Y, Z]
    """
    if indices_opti_tri is None:
        return None
    pts = vertices[indices_opti_tri[:, 0].to(torch.int32)]
    tri = vertices[faces[indices_opti_tri[:, 1].to(torch.int32)]]
    mask = ~(tri.permute(1, 0, 2) == pts).all(dim=2).permute(1, 0)
    vec = (tri[mask].reshape(tri.shape[0], 2, 3).permute(1, 0, 2) - pts).permute(2, 1, 0)
    vec[:, :, 0] *= indices_opti_tri[:, 2]
    vec[:, :, 1] *= indices_opti_tri[:, 3]
    vec = vec.permute(1, 2, 0)
    return pts + vec[:, 0] + vec[:, 1]


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


def transformJSON(input_path, output_path, extension):
    import json
    f = open(input_path, 'r')
    a = json.load(f)
    f.close()
    data = {'labels': {}}
    keys = a['labels'].keys()
    for key in keys:
        data['labels'][f'{key}.{extension}'] = a['labels'][key]
    f = open(output_path, 'w')
    json.dump(data, f)
    f.close()
