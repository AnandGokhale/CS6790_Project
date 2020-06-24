
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

trans = readFile("./tryingrecur.txt")

gt = readFile("../01/01/01.txt")

nptrans = np.asarray(trans)
gt = np.asarray(gt)

print(nptrans.shape)


canvasH = 1200
canvasW = 1200
traj = np.zeros((canvasH,canvasW,3), dtype=np.uint8)

for t in gt:
    t[0] = int((t[0] + 20))
    t[2] = int((t[2]  + 900))
    print(t)
    cv2.circle(traj, (int(t[0]),int(t[2])), 1, (0,255,0), 2)


for t in trans:
    t[0] = int((t[0] + 20))
    t[2] = int((t[2]  + 900))
    print(t)
    cv2.circle(traj, (t[0],t[2]), 1, (0,0,255), 2)


cv2.imshow('Trajectory', traj)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Trajectory2.0.png",traj)


