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
vert = vg.getVertices(0)[np.load("directionnalMatrix.npy")[:,0]]
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
masqueOV, masqueOF = read3D.readOBJ("nogit/masqueOriginelle.obj")
points = []
with open('nogit/marqueursMasqueOriginelle.txt',"r") as f:
    while True:
        line = f.readline()
        if line == "":
            break
        line = line.split(",")
        points.append([float(line[i]) for i in range(3)])
indexs = util.getIndexForMatchPoints(masqueOV,masqueOF,points)
masqueVertices, masqueFaces = read3D.readOBJ("nogit/scan_scaled.obj")
points = util.readAllIndexOptiTri(masqueVertices,masqueFaces,indexs)
visageVertices, visageFaces = read3D.readOBJ("nogit/fit_scan_result.obj")
indexs = util.getIndexForMatchPoints(visageVertices,visageFaces,points)
#np.save("balises.npy",indexs)
points2 = util.readAllIndexOptiTri(visageVertices,visageFaces,indexs)
np.save("balises.npy",indexs)
#viewer = Viewer(torch.tensor([visageVertices]),None,visageFaces,otherObjects=[[points,[]],[points2,[]]])
#[0.023051, -0.003374, -0.011894]
#[-0.024288, -0.005205, -0.00865]
"""