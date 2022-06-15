import random
import numpy as np

def saveVertices(vertices,all=False, minX=0.05, minZ=-0.03, maxZ=0.15):
    t=[]
    for i in range(len(vertices)):
        if vertices[i][0]>minX and vertices[i][2]>minZ and vertices[i][2]<maxZ:
            t.append([i,float(vertices[i][0]),float(vertices[i][1]),float(vertices[i][2]),-1,-1])
    t = calculDistanceSommets(t,display=True)
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

def genDirectionnalMatrix(vertices):
    m = []
    for i in range(len(vertices)):
        indexD = [-1,-1,-1,-1,-1,-1]
        distD = [-1,-1,-1,-1,-1,-1]
        dist1D = [-1,-1,-1,-1,-1,-1]
        coo = vertices[i]
        for j in range(len(vertices)):
            if i==j:
                continue
            coo2 = vertices[j]
            dist = np.sqrt((coo2[0]-coo[0])**2+(coo2[1]-coo[1])**2+(coo2[2]-coo[2])**2)
            ind = getIndexForD(indexD,distD,dist1D,dist,coo,coo2)
            if ind[0]!=-1:
                indexD[ind[0]] = j
                distD[ind[0]] = dist
                dist1D[ind[0]] = ind[1]
        m.append(indexD)
        print(str(i)+"/"+str(len(vertices)-1))
    np.save("directionnalMatrix.npy",m)


def getIndexForD(indexD,distD,dist1D,dist,coo,coo2):
    for i in range(len(indexD)):
        if indexD[i]==-1:
            n = coo2[int(i/2)]-coo[int(i/2)]
            if i%2==0 and n<=0:
                return [i,n]
            elif i%2!=0 and n>=0:
                return [i,n]
    best = [-1,-1]
    for i in range(len(distD)):
        if dist<distD[i]:
            n = coo2[0]-coo[0]
            if i==0 and (n)>dist1D[i] and (n)<=0:
                best = [i,n]
            if i==0 and (n)<dist1D[i] and (n)>=0:
                best = [i,n]
            n = coo2[1]-coo[1]
            if i==0 and (n)>dist1D[i] and (n)<=0:
                best = [i,n]
            if i==0 and (n)<dist1D[i] and (n)>=0:
                best = [i,n]
            n = coo2[2]-coo[2]
            if i==0 and (n)>dist1D[i] and (n)<=0:
                best = [i,n]
            if i==0 and (n)<dist1D[i] and (n)>=0:
                best = [i,n]
    return best
