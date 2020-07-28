import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, help='ground truth file')
parser.add_argument('--pred', type=str, help='prediction file',nargs='+')
parser.add_argument('--labels', type=str, nargs='+', help='labels for the file')
parser.add_argument('--loc', type=str, help='location of legend', default='ur')
# example usage : python plotTraj.py --gt ./dataset/poses/00.txt --pred gftt-brief/00_p.txt Howard/fast/final\ output/results/00.txt
# ./StereoDSO/kitti/00.txt --labels sptam_gftt_brief howard_fast stereo_dso 
args = parser.parse_args()


loc_dict  = {'ur':'upper right', 'lr':'lower right', 'ul':'upper left', 'll':'lower left'}
def readFile(filename):
    f = open(filename, "r")
    f1 = f.readlines()

    translation = []

    for x in f1:
        temp = x.split()
        tlist = [float(temp[3]), float(temp[7]), float(temp[11])]
        translation.append(tlist)

    return translation


#trans = readFile(
#    "/media/sumanth/a2790194-52bb-459d-aea3-c04d3783dd52/SixthSem/GPCV/cgarg-stereo-visual-odometry/src/svoPoseOut_Clique.txt"
#)
trans_list = []
for filename in args.pred:
	trans_list.append(np.asarray(readFile(filename)))
gt = readFile(args.gt)
gt = np.asarray(gt)


canvasH = 500
canvasW = 800
traj = np.zeros((canvasH, canvasW, 3), dtype=np.uint8)

#for t in gt:
#    t[0] = int((t[0] + 20))
#    t[2] = int((t[2] + 400))
#    print(t)
#    cv2.circle(traj, (int(t[0]), int(t[2])), 1, (0, 255, 0), 2)

plt.plot(gt[:,0], gt[:,2], label='Ground truth')

for i in range(len(trans_list)):
	trans = trans_list[i]
	label = args.labels[i]
	plt.plot(trans[:,0], trans[:,2], label=label)
plt.legend(loc=loc_dict[args.loc.lower()])
plt.xlabel(r'x$\rightarrow$')
plt.ylabel(r'z$\rightarrow$')
plt.savefig("Trajectory.png")
plt.show()

