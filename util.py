import random
import numpy as np

def saveVertices(vertices,all=False, minX=0.05, minZ=-0.03, maxZ=0.15):
    t=[]
    for i in range(len(vertices)):
        if vertices[i][0]>minX and vertices[i][2]>minZ and vertices[i][2]<maxZ:
            t.append([i,vertices[i]])
    for i in range(len(t)):
        point = t[i][1]
        distance = 0
        index = -1
        for j in range(len(t)):
            if i==j:
                continue
            p = t[j][1]
            dist = np.sqrt((p[0]-point[0])**2+(p[1]-point[1])**2+(p[2]-point[2])**2)
            if index==-1 or dist<distance:
                distance = dist
                index = t[j][0]
        t[i].append(index)
        t[i].append(distance)
        print(str(i+1)+"/"+str(len(t)))
    t.sort(key=lambda t:t[3])
    np.save("sommets.npy",t)

def fullRandomBalises():
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
        if t[i][2] not in l:
            l.append(i)
        else:
            t2.append(t[i])
    for i in range(n,len(t)):
        t2.append(t[i])
    t = calculDistanceSommets(t2)
    saveSommets(t)


def loadSommets():
    return np.load("sommets.npy")

def calculDistanceSommets(t):
    for i in range(len(t)):
        point = t[i][1]
        distance = 0
        index = -1
        for j in range(len(t)):
            if i==j:
                continue
            p = t[j][1]
            dist = np.sqrt((p[0]-point[0])**2+(p[1]-point[1])**2+(p[2]-point[2])**2)
            if index==-1 or dist<distance:
                distance = dist
                index = t[j][0]
        t[i][2]=index
        t[i][3]=distance
    t.sort(key=lambda t:t[3])
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