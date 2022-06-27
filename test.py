"""
def meanPos(vertices):
    mean = [0,0,0]
    for v in vertices:
        mean[0] = mean[0]+v[0]
        mean[1] = mean[1]+v[1]
        mean[2] = mean[2]+v[2]
    mean[0] = mean[0]/len(vertices)
    mean[1] = mean[1]/len(vertices)
    mean[2] = mean[2]/len(vertices)
    return mean

vg = VisageGenerator(0,0,0,0,0,0,0)
vert = vg.get_vertices(0)[np.load("directionalMatrix.npy")[:,0]]
visagePos = meanPos(vert)
obj = np.load("masque.npy", allow_pickle=True)
obj[0] = np.array(obj[0])/1000

with open("../test2.txt","r") as f:
    points = []
    while True:
        p = f.readline().split(",")
        if p == ['']:
            break
        for i in range(len(p)):
            p[i] = float(p[i])
        points.append(p)

points = np.array(points)
points = points/1000

pointsPos = meanPos(points)
ecart = [visagePos[i].item()-pointsPos[i] for i in range(3)]
for i in range(3): points[:,i] = points[:,i]+ecart[i]
for i in range(3): obj[0][:,i] = obj[0][:,i]+ecart[i]

vg.view([obj,[points,[]]])
"""

"""
import read3D
from Viewer import Viewer
masqueOV, masqueOF = read3D.read_obj("nogit/masqueOriginelle.obj")
points = []
with open('nogit/marqueursMasqueOriginelle.txt',"r") as f:
    while True:
        line = f.readline()
        if line == "":
            break
        line = line.split(",")
        points.append([float(line[i]) for i in range(3)])
indexs = util.get_index_for_match_points(masqueOV,masqueOF,points)
masqueVertices, masqueFaces = read3D.read_obj("nogit/scan_scaled.obj")
points = util.read_all_index_opti_tri(masqueVertices,masqueFaces,indexs)
visageVertices, visageFaces = read3D.read_obj("nogit/fit_scan_result.obj")
indexs = util.get_index_for_match_points(visageVertices,visageFaces,points)
#np.save("markers.npy",indexs)
points2 = util.read_all_index_opti_tri(visageVertices,visageFaces,indexs)
np.save("markers.npy",indexs)
#viewer = Viewer(torch.tensor([visageVertices]),None,visageFaces,other_objects=[[points,[]],[points2,[]]])
#[0.023051, -0.003374, -0.011894]
#[-0.024288, -0.005205, -0.00865]
"""