def saveVertices(vertices,all=False,):
    t=[]
    for i in range(len(vertices),minX=0.05,minZ=-0.03,maxZ=0.15):
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

    with open("sommets.txt","w") as f:
        f.write("[\n")
        for e in t:
            f.write("["+str(e[0])+", ["+str(e[1][0])+", "+str(e[1][1])+", "+str(e[1][2])+"], "+str(e[2])+", "+str(e[3])+"]\n")
        f.write("]")

def fullRandomBalises():
    t = []
    for i in range(5023):
        t.append(i)
    random.shuffle(t)
    balises = []
    for i in range(150):
        balises.append(t[i])

    with open("balises.txt","w") as f:
        f.write("[")
        for i in range(len(balises)):
            f.write(str(balises[i]))
            if i<len(balises)-1:
                f.write(", ")
        f.write("]")

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
    t = []
    with open("sommets.txt","r") as f:
        line = f.readline()
        while True:
            line = f.readline()
            if line=="]":
                break
            line = line[1:len(line)-2]
            line = line.split(", ")
            e = [int(line[0])]
            coo = [float(line[1][1:len(line[1])])]
            coo.append(float(line[2]))
            coo.append(float(line[3][0:len(line[3])-1]))
            e.append(coo)
            e.append(int(line[4]))
            e.append(float(line[5]))
            t.append(e)
    return t

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
    with open("sommets.txt","w") as f:
        f.write("[\n")
        for e in t:
            f.write("["+str(e[0])+", ["+str(e[1][0])+", "+str(e[1][1])+", "+str(e[1][2])+"], "+str(e[2])+", "+str(e[3])+"]\n")
        f.write("]")

def selectRandomBalises():
    t = loadSommets()
    balises = []
    for e in t:
        balises.append(e[0])
    random.shuffle(balises)
    balises = balises[0:150]    

    with open("balises.txt","w") as f:
        f.write("[")
        for i in range(len(balises)):
            f.write(str(balises[i]))
            if i<len(balises)-1:
                f.write(", ")
        f.write("]")