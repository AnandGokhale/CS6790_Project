import preprocessing

import numpy as np 
import cv2

import os


def LoG(img):
    blur        = cv2.GaussianBlur(img,(3,3),0)
    laplacian   = cv2.Laplacian(blur,cv2.CV_64F)
    return laplacian

def inverse_Projection_matrix(x,y,d):
    wc = [0,0,0,1]
    return wc 


# First import ALL images into memory

imgR  =     []
imgL  =     []

#INSERT LOOP TO POPULATE THESE LISTS HERE

num_images = len(imgR)


J = []
D = []

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Preprocessing Steps
for i in range(num_images):
    J.append([LoG(imgL[i]),LoG(imgR[i])])
    D.append(stereo.compute(imgL[i],imgR[i]))


#Actual Algorithm
fast = cv2.FastFeatureDetector_create()

F = []
#Step 1 : Feature detection:

m = 7 

# P = Projection matrix : Generate that later
# P_ = inverse of that
P_ = np.zeros((4,3))

for i in range(num_images):
    features =fast.detect(imgL[i],None) 
    for kp in features:
        x,y = kp
        w   = inverse_Projection_matrix(x,y,D[i][x,y])

        s   = imgL[i][x-((m-1)/2):x+((m-1)/2),y-((m-1)/2):y+((m-1)/2)]
        fa = [[x,y],w,s] 
    F.append(fa)

for fa in F:
    for fb in F:
        





