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