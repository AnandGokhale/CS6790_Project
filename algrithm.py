import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.approximation import max_clique,large_clique_size
import time

import data


P0,P1,R,T,K,distCoeffs = data.Camera_params()

IMSIZE = (1241, 376)  # (width, height) 


#Prepare Disparity Images

def findDisparity(img1,img2):
    assert img1.shape == img2.shape
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15) # numDisparities with 32 gives a pleasant depth map (nv)
    disparity = stereo.compute(img1.astype(np.uint8), img2.astype(np.uint8))

    return disparity

def feature_detector_orb(img1,img2):
    orb = cv2.ORB_create()
    # using orb for now
    kps1, des1 = orb.detectAndCompute(img1, None)
    kps2 , des2 = orb.detectAndCompute(img2, None)

    return kps1,des1,kps2,des2

def reproject(disparity):
    Q = cv2.stereoRectify(cameraMatrix1=K, distCoeffs1=distCoeffs, cameraMatrix2=K, distCoeffs2=distCoeffs, R=R, T=T, imageSize=IMSIZE)[4]
    img3d = cv2.reprojectImageTo3D(disparity, Q=Q) 
    return img3d


def process_frame(imgLeft,imgRight):
    imgLeft   = cv2.resize(imgLeft, IMSIZE) 
    imgRight  = cv2.resize(imgRight, IMSIZE)

    disparity = findDisparity(imgLeft,imgRight)
    kpsLeft, desLeft,_,_ = feature_detector_orb(imgLeft,imgRight)
    img3d = reproject(disparity)
    valKps = [i for i in range(len(kpsLeft)) if disparity[int(kpsLeft[i].pt[1]), int(kpsLeft[i].pt[0])]!=-1]
    
    valDes = desLeft[valKps]
    valKps = np.array(kpsLeft)[valKps]
    return valKps,valDes,img3d


# Image from timeframe 1
imgLeft_1   = cv2.imread('000001.png', 0)
imgRight_1  = cv2.imread('1_000001.png', 0)

#Image from timeframe 3
imgLeft_3   = cv2.imread('000003.png', 0)
imgRight_3  = cv2.imread('1_000003.png', 0)

valKps1,valDes1,img3d1 =  process_frame(imgLeft_1,imgRight_1)
valKps2,valDes2,img3d2 =  process_frame(imgLeft_3,imgRight_3)


# list of matched keypoints
kps1_m = []
kps2_m = []

# flann matching / A3
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(valDes1.astype(np.float32), valDes2.astype(np.float32), k=2)

for (m,_) in matches:
        kps2_m.append(valKps2[m.trainIdx])
        kps1_m.append(valKps1[m.queryIdx])

# visualize matched keypoints
img1L = cv2.drawKeypoints(imgLeft_1, kps1_m, None, color=(0,255,0))
img2L = cv2.drawKeypoints(imgLeft_3, kps2_m, None, color=(0,255,0))

kps1_m = np.array(kps1_m)
kps2_m = np.array(kps2_m)


# A4 
W = np.zeros((len(kps1_m), len(kps1_m)))
delta = 0.2 # nv

for i in range(len(kps1_m)):
    pt11 = kps1_m[i].pt
    pt12 = kps2_m[i].pt 
    w1 = img3d1[int(pt11[1]), int(pt11[0])]
    w2 = img3d2[int(pt12[1]), int(pt12[0])]
    dist1 = np.linalg.norm(w1-w2, ord=2)
    for j in range(len(kps1_m)):
        pt21 = kps1_m[j].pt
        pt22 = kps2_m[j].pt 
        w1_ = img3d1[int(pt21[1]), int(pt21[0])]
        w2_ = img3d2[int(pt22[1]), int(pt22[0])]
        dist2 = np.linalg.norm(w1_-w2_, ord=2)
        dist = abs(dist1-dist2)
        if dist < delta:
            W[i,j] = 1

# print(np.sum(W==1))

# build clique

G = nx.from_numpy_matrix(W)
inlier_set = max_clique(G)
inlier_set = list(inlier_set)
kps1_inliers = kps1_m[inlier_set]
kps2_inliers = kps2_m[inlier_set]

plt.subplot(2,1,1)
plt.imshow(img1L)
plt.subplot(2,1,2)
plt.imshow(img2L)
plt.show()






