import random
import numpy as np
import torch
import os
from skimage.io import imsave


def get_vertex_markers(vertex, min_x=0.05, min_z=-0.03, max_z=0.15, min_eye_distance=0.0139, min_noise_distance=0.0130):
    """
    This function allows to remove the vertices present in the eyes and noise and 
    what are not included in the default zone of the markers.
    Input:
        - vertex : array of all vertices for all face
        - min_x : minimum value for X coordinates of vertices
        - min_z : minimum value for Z coordinates of vertices
        - max_z : maximum value for Z coordinates of vertices
        - min_eye_distance : minimum eye distance
        - min_noise_distance : minimum noise distance
    Return:
        vertex array respecting the conditions, index correpondance
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


def gen_vertices_file(vertices):
    t = []
    for i in range(len(vertices)):
        t.append([i, float(vertices[i][0]), float(vertices[i][1]), float(vertices[i][2]), -1, -1])
    t = calc_distance_vertices(t, True)
    np.save("sommets.npy", t)


def full_random_markers():
    """
    generate a numpy array containing 150 random marker indexes and save it to a numpy file.
    Input: None
    Return: None
    """
    t = []
    for i in range(5023):
        t.append(i)
    random.shuffle(t)
    balises = []
    for i in range(150):
        balises.append(t[i])

    np.save("balises.npy", balises)


def delete_markers(n):
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


def load_vertices():
    return np.load("sommets.npy")


def calc_distance_vertices(t, display=False):
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
        if display:
            print(str(i + 1) + "/" + str(len(t)))
    t.sort(key=lambda key: t[5])
    return t


def save_vertices(t):
    np.save("sommets.npy", t)


def select_random_markers():
    t = load_vertices()
    balises = []
    for e in t:
        balises.append(e[0])
    random.shuffle(balises)
    balises = balises[0:150]

    np.save('balises.npy', balises)


def gen_directional_matrix(vertices, index_list):
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
    np.save("directionnalMatrix.npy", m)


def get_index_for_d(index_d, dist_d, dist, coo, coo2):
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


def save_faces(vertice):
    balises = np.load("balises.npy")
    data = np.zeros([len(vertice), len(balises), 3])
    for i in range(len(vertice)):
        for j in range(len(balises)):
            data[i, j, :] = vertice[i][balises[j]]
    np.save("data.npy", data)


def get_index_for_match_points(vertices, faces, points, verbose=False, triangle_optimize=True, pas_tri=1000):
    """
    return: list index of index matching points.
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
    print(no_tri, "/", len(points), " points non pas eu besoin de triangles !")
    return list_index


def get_index_triangles_match_vertex(triangles, index_point):
    faces = []
    for i in range(len(triangles)):
        if index_point in triangles[i]:
            faces.append(i)
    return faces


def get_vector_for_point(triangles, p):
    """
    input:
        - triangles : triangle array => triangle = coo triangle [x,y,z]
        - p : coo point
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


def read_index_opti_tri(vertices, faces, index_opti_tri):
    p = vertices[int(index_opti_tri[0])]
    if index_opti_tri[1] != -1:
        tri = np.array(vertices)[faces[int(index_opti_tri[1])]]
        vectors = np.array(get_vector_for_point([tri], p)[0])
        p = p + vectors[0] * index_opti_tri[2]
        p = p + vectors[1] * index_opti_tri[3]
    return p


def read_all_index_opti_tri(vertices, faces, indexs_opti_tri):
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


def save_obj(filename, vertices, faces, textures=None, uvcoords=None, uvfaces=None, texture_type='surface'):
    assert vertices.ndimension() == 2
    assert faces.ndimension() == 2
    assert texture_type in ['surface', 'vertex']
    # assert texture_res >= 2

    if textures is not None and texture_type == 'surface':
        textures = textures.detach().cpu().numpy().transpose(1, 2, 0)
        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        # texture_image, vertices_textures = create_texture_image(textures, texture_res)
        texture_image = textures
        texture_image = texture_image.clip(0, 1)
        texture_image = (texture_image * 255).astype('uint8')
        imsave(filename_texture, texture_image)
    else:
        # remove weak warnings
        filename_mtl = ""
        material_name = ""
        filename_texture = ""

    faces = faces.detach().cpu().numpy()

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        if textures is not None and texture_type == 'vertex':
            for vertex, color in zip(vertices, textures):
                f.write('v %.8f %.8f %.8f %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2],
                                                               color[0], color[1], color[2]))
            f.write('\n')
        else:
            for vertex in vertices:
                f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
            f.write('\n')

        if textures is not None and texture_type == 'surface':
            for vertex in uvcoords.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, uvfaces[i, 0] + 1, face[1] + 1, uvfaces[i, 1] + 1, face[2] + 1, uvfaces[i, 2] + 1))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

    if textures is not None and texture_type == 'surface':
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))
