import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from utils import get3Dpoints
from stereo import *
from utils import *
### projection matrices for left and right cameras for sequence 00
P0L = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
                [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]) 
P0R  = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02], 
                [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
assert (P0L[:3,:3]==P0R[:3,:3]).all() # just checking if matrices are for rectified images


P1L = P0L
P1R = P0R 



R = np.eye(3) # rotation matrix between left and right camera views
T = P0R[:,3] # translation between the views
K = P0L[:3, :3] # calibration matrix / Camera matrix 
print('Camera calibration matrix : ', K)

FBSIZE = (30,15)  # feature bucketing size in x and y directions (nimp)
IMSIZE = (1241, 376)  # (width, height) 




img1L = cv2.imread('000001.png', 0)
img1R = cv2.imread('1_000001.png', 0)
img1L = cv2.resize(img1L, IMSIZE) 
img1R = cv2.resize(img1R, IMSIZE)

# load images at T + 2
img2L = cv2.imread('000003.png', 0)
img2R = cv2.imread('1_000003.png', 0) 
img2L = cv2.resize(img2L, IMSIZE) 
img2R = cv2.resize(img2R, IMSIZE)


# A1

# Bilateral filtering step is not required as
#   openCV's built-in does pre-filtering

stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15) # numDisparities with 32 gives a pleasant depth map (nv)
disparity1 = stereo.compute(img1L.astype(np.uint8), img1R.astype(np.uint8))
disparity2 = stereo.compute(img2L.astype(np.uint8), img2R.astype(np.uint8))
# rescale maps to get true disparity values 
disparity1 = disparity1.astype(np.float32)/16.0
disparity2 = disparity2.astype(np.float32)/16.0

orb = cv2.ORB_create()
# using orb for now
kps1, des1 = orb.detectAndCompute(img1L, None)
print('Number of keypoints in 1 :', len(kps1))


# get valid keypoints ie. those with known depth values. 
valInds = filterKps(kps1, disparity1)
valDes1 = des1[valInds]
valKps1 = np.array(kps1)[valInds]
print('Number of valid keypoints in 1 :', len(valKps1))


kps2, des2 = orb.detectAndCompute(img2L, None)
valInds = filterKps(kps2, disparity2)
valDes2 = des2[valInds]
valKps2 = np.array(kps2)[valInds]


del kps1, kps2, des1, des2 # not required anymore



kps1_m, kps2_m = getMatches(valKps1, valDes1, valKps2, valDes2)
# visualize matched keypoints
img1L = cv2.drawKeypoints(img1L, kps1_m, None, color=(0,255,0))
img2L = cv2.drawKeypoints(img2L, kps2_m, None, color=(0,255,0))

kps1_m = np.array(kps1_m)
kps2_m = np.array(kps2_m)

pts1L = np.array([kp.pt for kp in  kps1_m]).astype(int)
pts2L = np.array([kp.pt for kp in kps2_m]).astype(int)
# get the corresponding positions in the right image using disparity map
pts1R = getCorrespondence(pts1L, disparity1)
pts2R = getCorrespondence(pts2L, disparity2)

pts3d1 = triangulate3D(pts1L, pts1R, len(pts1L), P0L, P0R)
pts3d2 = triangulate3D(pts2L, pts2R, len(pts2L), P1L, P1R)

# del valDes1,  valDes2, valKps1, valKps2

# A4 
W = np.zeros((len(kps1_m), len(kps1_m)))
delta = 0.01 # nv

for i in range(len(pts1L)):
    # first set of matches
    pt11 = pts1L[i]
    pt12 = pts2L[i]
    w1 = pts3d1[i]
    w2 = pts3d2[i]
    dist1 = np.linalg.norm(w1-w2, ord=2)
    for j in range(len(pts1L)):
        pt21 = pts1L[j]
        pt22 = pts2L[j]
        w1_ = pts3d1[j]
        w2_ = pts3d2[j]
        dist2 = np.linalg.norm(w1_-w2_, ord=2)
        dist = abs(dist1-dist2)
        if dist < delta:
            W[i][j] = 1

clique = findMaxClique(W)

# pick up clique point 3D coords and features for optimization
cliqued3dPointT1 = pts3d1[clique]
cliqued3dPointT2 = pts3d2[clique]

# points = features
pts1L = pts1L[clique]
pts2L = pts2L[clique]

dSeed = np.zeros(6)

# find optimal rotation and translation params
optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=2000,
                args=(pts1L, pts2L, cliqued3dPointT1, cliqued3dPointT2, P0L))


error = optRes.fun
pointsInClique = len(clique)
e = error.reshape((pointsInClique*2, 3))
errorThreshold = 1.0
# prune points with error greater than errorThreshold
xRes1 = np.where(e[0:pointsInClique, 0] >= errorThreshold)
yRes1 = np.where(e[0:pointsInClique, 1] >= errorThreshold)
zRes1 = np.where(e[0:pointsInClique, 2] >= errorThreshold)
xRes2 = np.where(e[pointsInClique:2*pointsInClique, 0] >= errorThreshold)
yRes2 = np.where(e[pointsInClique:2*pointsInClique, 1] >= errorThreshold)
zRes2 = np.where(e[pointsInClique:2*pointsInClique, 2] >= errorThreshold)

pruneIdx = xRes1[0].tolist() + yRes1[0].tolist() + zRes1[0].tolist() + (xRes2[0] - pointsInClique).tolist() + (yRes2[0] - pointsInClique).tolist() +  (zRes2[0] - pointsInClique).tolist()
if (len(pruneIdx) > 0):
        uPruneIdx = list(set(pruneIdx))
        pts1L = np.delete(pts1L, uPruneIdx, axis=0)
        pts2L = np.delete(pts2L, uPruneIdx, axis=0)
        cliqued3dPointT1 = np.delete(cliqued3dPointT1, uPruneIdx, axis=0)
        cliqued3dPointT2 = np.delete(cliqued3dPointT2, uPruneIdx, axis=0)
        print('number pruned :', len(uPruneIdx))
        optRes = least_squares(minimizeReprojection, optRes.x, method='lm', max_nfev=2000,
                        args=(pts1L, pts2L, cliqued3dPointT1, cliqued3dPointT2, P0L))

print('Residual error:, ', optRes.cost)

inds = [0,2] # indices of frames
poses = getPose('./dataset/poses/00.txt', inds)
dofs = optRes.x
R = genEulerZXZMatrix(dofs[0], dofs[1], dofs[2])
t = np.array(dofs[3:])

R1 = np.linalg.inv(K)@poses[0][:,:-1]
t1 = np.linalg.inv(K)@poses[0][:,-1]

R2 = np.linalg.inv(K)@poses[1][:,:-1]
t2 = np.linalg.inv(K)@poses[1][:,-1]


t_err, theta = getErrors(R, R1.T@R2, t, t2-t1)
# P1L_est = K@np.c_[R, t]
# P1R_est = K@np.c_[R, t+P1R[:,:-1]] 


print('Translation error :', t_err)
print('Rotation error :', theta)

plt.subplot(2,1,1)
plt.imshow(img1L)
plt.subplot(2,1,2)
plt.imshow(img2L)
plt.show()
