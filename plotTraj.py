
import cv2
import numpy as np

def readFile(filename):
    f = open(filename,"r")
    f1 =f.readlines()

    translation = []

    for x in f1:
        temp = (x.split())
        tlist = [float(temp[3]),float(temp[7]),float(temp[11])]
        translation.append(tlist)
    
    return translation

trans = readFile("./dataset/poses/00.txt")


nptrans = np.asarray(trans)

print(nptrans.shape)

mean = np.mean(nptrans,axis = 0)


canvasH = 1200
canvasW = 1200
traj = np.zeros((canvasH,canvasW,3), dtype=np.uint8)


for t in trans:
    t[0] = int((t[0] - mean[0] + canvasH/2))
    t[2] = int((t[2] - mean[2] + canvasW/2))
    print(t)
    cv2.circle(traj, (t[0],t[2]), 1, (0,0,255), 2)

cv2.imshow('Trajectory', traj)
cv2.waitKey(0)
cv2.destroyAllWindows()


