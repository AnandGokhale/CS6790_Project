import cv2
import numpy as np
import argparse
import os

def readFile(filename):
    f = open(filename,"r")
    f1 =f.readlines()

    translation = []

    for x in f1:
        temp = (x.split())
        tlist = [float(temp[3]),float(temp[7]),float(temp[11])]
        translation.append(tlist)
    
    return translation


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gt",required = True,type = str,help='path to folder containing ground truth for kitti, containing all 11 ground truths')
    parser.add_argument("-p", "--pred",required = True,type = str,help='path to folder where predicted results are stored')
    parser.add_argument("-o", "--output",required = True,type = str,help='path to folder where plotted results are to be stored')
    args = parser.parse_args()

    for i in range(0,11):
        s = str(i).zfill(2) + ".txt"
        print("Reading " + os.path.join(args.pred,s) )
        trans   = readFile(os.path.join(args.pred,s))
        print("Reading " + os.path.join(args.gt,s) )
        gt      = readFile(os.path.join(args.gt,s))

        nptrans = np.asarray(trans)
        gt = np.asarray(gt)

        canvasH = 1200
        canvasW = 1200
        traj = np.zeros((canvasH,canvasW,3), dtype=np.uint8)

        for t in gt:
            t[0] = int((t[0]) + 500)
            t[2] = int((t[2]) + 200)
            #print(t)
            cv2.circle(traj, (int(t[0]),int(t[2])), 1, (0,255,0), 2)


        for t in trans:
            t[0] = int((t[0]) + 500)
            t[2] = int((t[2]) + 200)
            #print(t)
            cv2.circle(traj, (t[0],t[2]), 1, (0,0,255), 2)


        cv2.imshow('Trajectory', traj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(args.output,str(i).zfill(2) + ".png"),traj)


