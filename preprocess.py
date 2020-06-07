import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from rectify import getMatches


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


img1 = cv2.imread('000001.png', 0)
img2 = cv2.imread('1_000001.png', 0)
img1 = cv2.resize(img1, IMSIZE) 
img2 = cv2.resize(img2, IMSIZE)

# A1

# Bilateral filtering step is not required as
#   openCV's built-in does pre-filtering
# img1_bf = cv2.bilateralFilter(img1, 5, 30, 30)
# img2_bf = cv2.bilateralFilter(img2, 5, 30, 30)
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15) # numDisparities with 32 gives a pleasant depth map (nv)
disparity = stereo.compute(img1.astype(np.uint8), img2.astype(np.uint8))
orb = cv2.ORB_create()
# using orb for now
kps1, des1 = orb.detectAndCompute(img1, None)
kps2 , des2 = orb.detectAndCompute(img2, None)
print('Number of keypoints in 1 :', len(kps1))
distCoeffs = np.zeros(5) # distortion coefficients. Zero for kitti
# get Q matrix for reprojection
Q = cv2.stereoRectify(cameraMatrix1=K, distCoeffs1=distCoeffs, cameraMatrix2=K, distCoeffs2=distCoeffs, R=R, T=T, imageSize=IMSIZE)[4]
print('Q Matrix :', Q)
img3d = cv2.reprojectImageTo3D(disparity, Q=Q) 
# get valid keypoints ie. those with known depth values. 
valKps1 = [kp for kp in kps1 if disparity[int(kp.pt[1]), int(kp.pt[0])]!=-16]

# load images at T + 2
img21 = cv2.imread('000003.png', 0)
img22 = cv2.imread('1_000003.png', 0) 
disparity2 = stereo.compute(img21.astype(np.uint8), img22.astype(np.uint8))
kps21, des21 = orb.detectAndCompute(img21, None)
kps22 , des22 = orb.detectAndCompute(img22, None)
valKps2 = [kp for kp in kps21 if disparity[int(kp.pt[1]), int(kp.pt[0])]!=-16]
# 3D coordinates for T +2
img3d_2 = cv2.reprojectImageTo3D(disparity2, Q=Q)

# list of matched keypoints
kps1_m = []
kps21_m = []

# flann matching / A3
ratio = 0.9
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(des1.astype(np.float32), des21.astype(np.float32), k=2)

for i,(m,n) in enumerate(matches):
    if m.distance < ratio*n.distance:
        # good.append(m)
        kps21_m.append(kps21[m.trainIdx])
        kps1_m.append(kps1[m.queryIdx])

# visualize matched keypoints
img1 = cv2.drawKeypoints(img1, kps1_m, None, color=(0,255,0))
img21 = cv2.drawKeypoints(img21, kps21_m, None, color=(0,255,0))
# valid matches which have known disparity values
valMatches = [i for i in range(len(kps1_m)) if disparity[int(kps1_m[i].pt[1]), int(kps1_m[i].pt[0])]!=-16
                 and disparity2[int(kps21_m[i].pt[1]), int(kps21_m[i].pt[0])]!= -16]

kps1_m = np.array(kps1_m)[valMatches]
kps21_m = np.array(kps21_m)[valMatches]


# A4 
# W = np.zeros((len(valMatches), len(valMatches)))
# delta = 10 # nv


# for i in range(len(valMatches)):
#     pt11 = kps1_m[i].pt
#     pt12 = kps21_m[i].pt 
#     w1 = img3d[int(pt11[1]), int(pt11[0])]
#     w2 = img3d_2[int(pt12[1]), int(pt12[0])]
#     dist1 = np.linalg.norm(w1-w2, ord=2)
#     for j in range(len(valMatches)):
#         pt21 = kps1_m[j].pt
#         pt22 = kps21_m[j].pt 
#         w1_ = img3d[int(pt21[1]), int(pt21[0])]
#         w2_ = img3d_2[int(pt22[1]), int(pt22[0])]
#         dist2 = np.linalg.norm(w1_-w2_, ord=2)
#         dist = abs(dist1-dist2)
#         if dist < delta:
#             W[i][j] = 1

# print(np.sum(W==1))



plt.subplot(2,1,1)
plt.imshow(img1)
plt.subplot(2,1,2)
plt.imshow(img21)
plt.show()
