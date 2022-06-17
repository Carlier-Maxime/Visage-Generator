import random
import numpy as np

def getVerticeBalises(vertice, minX=0.05, minZ=-0.03, maxZ=0.15, minEyeDistance=0.0139):
    """
    This function allows to remove the vertices present in the eyes and 
    what are not included in the default zone of the markers.
    Input:
        - vertice : array of all vertices for all face
        - minX : minimum value for X coordinates of vertices
        - minZ : minimum value for Z coordinates of vertices
        - maxZ : maximum value for Z coordinates of vertices
        - minEyeDistance : minimum eye distance
    Return:
        vertice array respecting the conditions, index correpondance
    """
    eyeL = [0.108, -0.178, 0.085]
    eyeR = [0.108, -0.1155, 0.085]
    
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
            if inZone and distEyeL>minEyeDistance and distEyeR>minEyeDistance:
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