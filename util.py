import random
import numpy as np

def getVerticeBalises(vertice, minX=0.05, minZ=-0.03, maxZ=0.15, minEyeDistance=0.0139, minNoiseDistance=0.0130):
    """
    This function allows to remove the vertices present in the eyes and noise and 
    what are not included in the default zone of the markers.
    Input:
        - vertice : array of all vertices for all face
        - minX : minimum value for X coordinates of vertices
        - minZ : minimum value for Z coordinates of vertices
        - maxZ : maximum value for Z coordinates of vertices
        - minEyeDistance : minimum eye distance
        - minNoiseDistance : minimum noise distance
    Return:
        vertice array respecting the conditions, index correpondance
    """
    eyeL = [0.108, -0.178, 0.085]
    eyeR = [0.108, -0.1155, 0.085]
    noise = [0.143, -0.1464, 0.055]
    
    t=[]
    m=[]
    for vertices in vertice:
        l = []
        li = []
        for i in range(len(vertices)):
            p = vertices[i]
            inZone = vertices[i][0]>minX and vertices[i][2]>minZ and vertices[i][2]<maxZ
            distEyeL = np.sqrt((p[0]-eyeL[0])**2+(p[1]-eyeL[1])**2+(p[2]-eyeL[2])**2)
            distEyeR = np.sqrt((p[0]-eyeR[0])**2+(p[1]-eyeR[1])**2+(p[2]-eyeR[2])**2)
            distNoise = np.sqrt((p[0]-noise[0])**2+(p[1]-noise[1])**2+(p[2]-noise[2])**2)
            if inZone and distEyeL>minEyeDistance and distEyeR>minEyeDistance and distNoise>minNoiseDistance:
                l.append(vertices[i])
                li.append(i)
        t.append(l)
        m.append(li)
    return t,m

def genSommetsFile(vertices):
    t = []
    for i in range(len(vertices)):
        t.append([i,float(vertices[i][0]),float(vertices[i][1]),float(vertices[i][2]),-1,-1])
    t = calculDistanceSommets(t,True)
    np.save("sommets.npy",t)

def fullRandomBalises():
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

    np.save("balises.npy",balises)

def deleteBalises(n):
    t = loadSommets()
    l = []
    t2 = []
    for i in range(n):
        if t[i][4] not in l:
            l.append(i)
        else:
            t2.append(t[i])
    for i in range(n,len(t)):
        t2.append(t[i])
    t = calculDistanceSommets(t2)
    saveSommets(t)


def loadSommets():
    return np.load("sommets.npy")

def calculDistanceSommets(t,display=False):
    for i in range(len(t)):
        distance = 0
        index = -1
        for j in range(len(t)):
            if i==j:
                continue
            dist = np.sqrt((t[j][1]-t[i][1])**2+(t[j][2]-t[i][2])**2+(t[j][3]-t[i][3])**2)
            if index==-1 or dist<distance:
                distance = dist
                index = t[j][0]
        t[i][4] = index
        t[i][5] = distance
        if display:
            print(str(i+1)+"/"+str(len(t)))
    t.sort(key=lambda t:t[5])
    return t

def saveSommets(t):
    np.save("sommets.npy",t)

def selectRandomBalises():
    t = loadSommets()
    balises = []
    for e in t:
        balises.append(e[0])
    random.shuffle(balises)
    balises = balises[0:150]    

    np.save('balises.npy',balises)

def genDirectionnalMatrix(vertices,indexList):
    m = []
    for i in range(len(vertices)):
        indexD = [-1,-1,-1,-1,-1,-1]
        distD = [-1,-1,-1,-1,-1,-1]
        coo = vertices[i]
        for j in range(len(vertices)):
            if i==j:
                continue
            coo2 = vertices[j]
            dist = np.sqrt((coo2[0]-coo[0])**2+(coo2[1]-coo[1])**2+(coo2[2]-coo[2])**2)
            ind = getIndexForD(indexD,distD,dist,coo,coo2)
            if ind!=-1:
                indexD[ind] = j
                distD[ind] = dist
        indexD.insert(0,indexList[i])
        m.append(indexD)
        print(str(i)+"/"+str(len(vertices)-1))
    np.save("directionnalMatrix.npy",m)


def getIndexForD(indexD,distD,dist,coo,coo2):
    assert len(indexD)==len(distD)
    diff = [coo2[0]-coo[0],
            coo2[1]-coo[1],
            coo2[2]-coo[2]]

    for i in range(len(indexD)):
        n = diff[int(i/2)]
        if max(np.abs(diff))==np.abs(n) and (indexD[i]==-1 or dist<distD[i]):
            if i%2==0 and n<0:
                return i
            elif i%2!=0 and n>0:
                return i
    return -1

def saveFaces(vertice):
    balises = np.load("balises.npy")
    data = np.zeros([len(vertice),len(balises),3])
    for i in range(len(vertice)):
        for j in range(len(balises)):
            data[i,j,:] = vertice[i][balises[j]]
    np.save("data.npy",data)

def getIndexForMatchPoints(vertices,faces,points,verbose=False,triangleOptimize=True,pasTri=1000):
    """
    return: list of index matching points.
    """
    l = []
    noTri = 0
    for ind in range(len(points)):
        if verbose: print(ind,"/",len(points)-1,"points")
        p = points[ind]
        index = -1
        dist = -1
        for i in range(len(vertices)):
            v = vertices[i]
            d = np.sqrt((v[0]-p[0])**2+(v[1]-p[1])**2+(v[2]-p[2])**2)
            if dist==-1 or d<dist:
                index=i
                dist = d
        if triangleOptimize:
            indexTriangles = getIndexTrianglesMatchVertice(vertices,faces,index)
            triangles = np.array(faces)[indexTriangles]
            triangles = np.array(vertices)[triangles]
            vectors = getVectorForPoint(triangles,vertices[index])
            indVect = -1
            percentage = [0,0]
            dist2 = dist
            for i in range(len(vectors)):
                vect = np.array(vectors[i])/pasTri
                perc = []
                distance = dist
                vert = vertices[index]
                for j in range(len(vect)):
                    pas = 0
                    for k in range(1,pasTri):
                        v = vert+vect[j]*k
                        d = np.sqrt((v[0]-p[0])**2+(v[1]-p[1])**2+(v[2]-p[2])**2)
                        if d<=distance:
                            pas=k
                            distance = d
                        else:
                            break
                    perc.append(pas/pasTri)
                    vert = v
                if distance < dist2:
                    dist2 = distance
                    indVect = i
                    percentage = perc
            if indVect==-1: 
                indTri=-1
                noTri+=1
            else: indTri = indexTriangles[indVect]
            l.append([index,indTri,percentage[0],percentage[1]])
        else: l.append(index)
    print(noTri,"/",len(points)," points non pas eu besoin de triangles !")
    return l

def getIndexTrianglesMatchVertice(vertices,triangles,indexPoint):
    faces = []
    for i in range(len(triangles)):
        if indexPoint in triangles[i]:
            faces.append(i)
    return faces

def getVectorForPoint(triangles,p):
    """
    input:
        - triangles : triangle array => triangle = coo triangle [x,y,z]
        - p : coo point
    """
    vectors = []
    for triangle in triangles:
        t = [0,0,0]
        ind = []
        for i in range(3):
            if p[0] == triangle[i][0] and p[1] == triangle[i][1] and p[2] == triangle[i][2]:
                t[0]=triangle[i]
            else:
                ind.append(i)
        if len(ind)>2:
            print("Error in getVectoForPoint in util.py ! (verify your point is vertice of triangles)")
            exit(1)
        t[1] = triangle[ind[0]]
        t[2] = triangle[ind[1]]

        v = []
        for i in range(1,3):
            v.append([t[i][0]-p[0],t[i][1]-p[1],t[i][2]-p[2]])
        vectors.append(v)
    return vectors
        
def readIndexOptiTri(vertices,faces,indexOptiTri):
    p = vertices[int(indexOptiTri[0])]
    if indexOptiTri[1]!=-1:
        tri = np.array(vertices)[faces[int(indexOptiTri[1])]]
        vectors = np.array(getVectorForPoint([tri],p)[0])
        p = p+vectors[0]*indexOptiTri[2]
        p = p+vectors[1]*indexOptiTri[3]
    return p

def readAllIndexOptiTri(vertices,faces,indexsOptiTri):
    points = []
    for indexOptiTri in indexsOptiTri:
        points.append(readIndexOptiTri(vertices,faces,indexOptiTri))
    return points