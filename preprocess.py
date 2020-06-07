import numpy as np 
import cv2 
import matplotlib.pyplot as plt



### projection matrices for left and right cameras for sequence 00
P0 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
                [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]) 
P1  = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02], 
                [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
assert (P0[:3,:3]==P1[:3,:3]).all() # just checking if matrices are for rectified images
R = np.eye(3) # rotation matrix between left and right camera views
T = P1[:,3] # translation between the views
K = P0[:3, :3] # calibration matrix / Camera matrix 
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
distCoeffs = np.zeros(5) # distortion coefficients. Zero for kitti
# get Q matrix for reprojection
Q = cv2.stereoRectify(cameraMatrix1=K, distCoeffs1=distCoeffs, cameraMatrix2=K, distCoeffs2=distCoeffs, R=R, T=T, imageSize=IMSIZE)[4]
print('Q Matrix :', Q)
img3d = cv2.reprojectImageTo3D(disparity1, Q=Q) 
# get valid keypoints ie. those with known depth values. 
valKps1 = [i for i in range(len(kps1)) if disparity1[int(kps1[i].pt[1]), int(kps1[i].pt[0])]!=-1]
valDes1 = des1[valKps1]
valKps1 = np.array(kps1)[valKps1]


kps2, des2 = orb.detectAndCompute(img2L, None)
valKps2 = [i for i in range(len(kps2)) if disparity2[int(kps2[i].pt[1]), int(kps2[i].pt[0])]!=-1]
valDes2 = des2[valKps2]
valKps2 = np.array(kps2)[valKps2]
# 3D coordinates for T +2
img3d_2 = cv2.reprojectImageTo3D(disparity2, Q=Q)


del kps1, kps2, des1, des2 # not required anymore

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
img1L = cv2.drawKeypoints(img1L, kps1_m, None, color=(0,255,0))
img2L = cv2.drawKeypoints(img2L, kps2_m, None, color=(0,255,0))

kps1_m = np.array(kps1_m)
kps2_m = np.array(kps2_m)

# del valDes1,  valDes2, valKps1, valKps2

# A4 
W = np.zeros((len(kps1_m), len(kps1_m)))
delta = 0.2 # nv


for i in range(len(kps1_m)):
    pt11 = kps1_m[i].pt
    pt12 = kps2_m[i].pt 
    w1 = img3d[int(pt11[1]), int(pt11[0])]
    w2 = img3d_2[int(pt12[1]), int(pt12[0])]
    dist1 = np.linalg.norm(w1-w2, ord=2)
    for j in range(len(kps1_m)):
        pt21 = kps1_m[j].pt
        pt22 = kps2_m[j].pt 
        w1_ = img3d[int(pt21[1]), int(pt21[0])]
        w2_ = img3d_2[int(pt22[1]), int(pt22[0])]
        dist2 = np.linalg.norm(w1_-w2_, ord=2)
        dist = abs(dist1-dist2)
        if dist < delta:
            W[i][j] = 1

# print(np.sum(W==1))



plt.subplot(2,1,1)
plt.imshow(img1L)
plt.subplot(2,1,2)
plt.imshow(img2L)
plt.show()
