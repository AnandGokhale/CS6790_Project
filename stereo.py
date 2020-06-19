import numpy as np
import cv2
import matplotlib.pyplot as pyplot

from scipy.optimize import least_squares
from math import cos, sin

from data import *


'''
In the code, there is 
1: Disparity
2: Feature extraction
3: Feature Matching
4: 3D triangulation so far.

'''
#Global constants for left and Right
L = 0

R = 1



def estDisparity(imgL,imgR,engine = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)):
    return np.divide(engine.compute(imgL,imgR).astype(np.float32),16.0)


def debugDisparity(img1,img2,img1_disparity,img2_disparity):
    cv2.imshow("Img1 left  ",img1[0])
    cv2.imshow("Img1 Right ",img1[1])
    cv2.imshow("Img2 left " ,img2[0])
    cv2.imshow("Img2 Right ",img2[1])

    cv2.imshow("Img1 Disparity",img1_disparity)

    cv2.imshow("Img2 Disparity",img2_disparity)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def orbfeatureDetector(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    return kp,des


def debugOrb(img,kp):
    cv2.imshow("Debugging Orb" , cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debugMatcher(img1,kp1_matched,img2,kp2_matched):
    imgDebug1 = cv2.drawKeypoints(img1, kp1_matched, None, color=(0,255,0))
    imgDebug2 = cv2.drawKeypoints(img2, kp2_matched, None, color=(0,255,0))


    cv2.imshow('frame 1', imgDebug1)
    cv2.imshow('frame 2', imgDebug2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Matcher(img1,img2,kp1,des1,kp2,des2):
    kps1_m = []
    kps2_m = []

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
    # Need to draw only good matches, so create a mask

    for (m,_) in matches:
            kps2_m.append(kp2[m.trainIdx])
            kps1_m.append(kp1[m.queryIdx])


    debugMatcher(img1,kps1_m,img2,kps2_m)


    # Converting Keypoints to Pixel values
    kp1_matched = []
    kp2_matched = []

    for kp in kps1_m:
        kp1_matched.append(kp.pt)

    for kp in kps2_m:
        kp2_matched.append(kp.pt)

    kps1_m = np.asarray(kp1_matched)
    kps2_m = np.asarray(kp2_matched)

    return kps1_m,kps2_m
    
def featureEliminator(img1_disparity,img2_disparity,kp1_matched,kp2_matched,disparityMinThres = 0.0, disparityMaxThres = 100.0):
    kps1L = kp1_matched
    kps2L = kp2_matched

    kps1R = np.copy(kp1_matched)
    kps2R = np.copy(kp2_matched)

    selectedPointMap = np.zeros(kps1L.shape[0])

    for i in range(kps1L.shape[0]):
        T1Disparity = img1_disparity[int(kps1L[i,1]), int(kps1L[i,0])]
        T2Disparity = img2_disparity[int(kps2L[i,1]), int(kps2L[i,0])]
        
        if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres 
            and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
            kps1R[i, 0] = kps1L[i, 0] - T1Disparity
            kps2R[i, 0] = kps2L[i, 0] - T2Disparity
            selectedPointMap[i] = 1
            
    selectedPointMap = selectedPointMap.astype(bool)
    trackPoints1L_3d = kps1L[selectedPointMap, ...]
    trackPoints1R_3d = kps1R[selectedPointMap, ...]
    trackPoints2L_3d = kps2L[selectedPointMap, ...]
    trackPoints2R_3d = kps2R[selectedPointMap, ...]

    return trackPoints1L_3d,trackPoints1R_3d,trackPoints2L_3d,trackPoints2R_3d

def triangulate3D(trackPointsL_3d,trackPointsR_3d,numPoints,Proj1,Proj2):
    d3dPoints = np.ones((numPoints,3))

    for i in range(numPoints):
        #for i in range(1):
        pLeft = trackPointsL_3d[i,:]
        pRight = trackPointsR_3d[i,:]
        
        X = np.zeros((4,4))
        X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
        X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
        X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
        X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]
        
        _,_,V = np.linalg.svd(X)

        d3dPoints[i, :] = (V[-1]/V[-1,-1]).T[:-1]

    return d3dPoints

def generateAdjMatrix(d3dPointsT1,d3dPointsT2,distDifference=0.01):
    numPoints = d3dPointsT1.shape[0]
    W = np.zeros((numPoints, numPoints))

    # diff of pairwise euclidean distance between same points in T1 and T2
    for i in range(numPoints):
        for j in range(numPoints):
            T2Dist = np.linalg.norm(d3dPointsT2[i,:] - d3dPointsT2[j,:])
            T1Dist = np.linalg.norm(d3dPointsT1[i,:] - d3dPointsT1[j,:])
            if (abs(T2Dist - T1Dist) < distDifference):
                W[i, j] = 1

    return W

def findMaxClique(W):
    #Sum along Axis = 1???
    maxn = np.argmax(np.sum(W,axis = 1))
    maxc = np.sum(W,axis = 1)[maxn]
    clique = [maxn]
    isin = True

    numPoints = W.shape[0]
    while True:
        potentialnodes = list()
        # Find potential nodes which are connected to all nodes in the clique
        for i in range(numPoints):
            for j in range(len(clique)):
                isin = isin & bool(W[i, clique[j]])
            if isin == True and i not in clique:
                potentialnodes.append(i)
            isin=True

        count = 0
        maxn = 0
        maxc = 0
        # Find the node which is connected to the maximum number of potential nodes and store in maxn
        for i in range(len(potentialnodes)):
            for j in range(len(potentialnodes)):
                if W[potentialnodes[i], potentialnodes[j]] == 1:
                    count = count+1
            if count > maxc:
                maxc = count
                maxn = potentialnodes[i]
            count = 0
        if maxc == 0:
            break
        clique.append(maxn)


    return clique

def genEulerZXZMatrix(psi, theta, sigma):
    # ref http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2008-A.pdf 
    mat = np.zeros((3,3))
    mat[0,0] = cos(psi) * cos(sigma) - sin(psi) * cos(theta) * sin(sigma)
    mat[0,1] = -cos(psi) * sin(sigma) - sin(psi) * cos(theta) * cos(sigma)
    mat[0,2] = sin(psi) * sin(theta)
    
    mat[1,0] = sin(psi) * cos(sigma) + cos(psi) * cos(theta) * sin(sigma)
    mat[1,1] = -sin(psi) * sin(sigma) + cos(psi) * cos(theta) * cos(sigma)
    mat[1,2] = -cos(psi) * sin(theta)
    
    mat[2,0] = sin(theta) * sin(sigma)
    mat[2,1] = sin(theta) * cos(sigma)
    mat[2,2] = cos(theta)
    
    return mat

def minimizeReprojection(dof,d2dPoints1, d2dPoints2, d3dPoints1, d3dPoints2, w2cMatrix):
    
    perspectiveProj = np.eye(4)
    Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    perspectiveProj[:3,:3] = Rmat
    perspectiveProj[0,-1] = dof[3]
    perspectiveProj[1,-1] = dof[4]
    perspectiveProj[2,-1] = dof[5]

    print (perspectiveProj)

    numPoints = d2dPoints1.shape[0]
    errorA = np.zeros((numPoints,3))
    errorB = np.zeros((numPoints,3))
    
    forwardProjection = np.matmul(w2cMatrix, perspectiveProj)
    backwardProjection = np.matmul(w2cMatrix, np.linalg.inv(perspectiveProj))
    for i in range(numPoints):
        Ja = np.ones((3))
        Jb = np.ones((3))
        Wa = np.ones((4))
        Wb = np.ones((4))
        
        Ja[0:2] = d2dPoints1[i,:]
        Jb[0:2] = d2dPoints2[i,:]
        Wa[0:3] = d3dPoints1[i,:]
        Wb[0:3] = d3dPoints2[i,:]
        
        JaPred = np.matmul(forwardProjection, Wb)
        JaPred /= JaPred[-1]
        e1 = Ja - JaPred
        
        JbPred = np.matmul(backwardProjection, Wa)
        JbPred /= JbPred[-1]
        e2 = Jb - JbPred
        
        errorA[i,:] = e1
        errorB[i,:] = e2
    
    residual = np.vstack((errorA,errorB))
    return residual.flatten()


def estimateOdometry(img1,img2):
    #img1 = [img1L,img1R]
    #img2 = [img2L,img2R]

    #Prepocessing the image
    img1[L]  = cv2.resize(img1[L], IMSIZE) 
    img1[R]  = cv2.resize(img1[R], IMSIZE) 
    img2[L]  = cv2.resize(img2[L], IMSIZE) 
    img2[R]  = cv2.resize(img2[R], IMSIZE) 

    Proj1,Proj2,_,_,_,_ = Camera_params()

    img1_disparity = estDisparity(img1[L],img1[R])
    img2_disparity = estDisparity(img2[L],img2[R])


    #debugDisparity(img1,img2,img1_disparity,img2_disparity)

    
    #Calculating img1 twice for now, can be made more efficient in the future
    kp1,des1 = orbfeatureDetector(img1[L])
    kp2,des2 = orbfeatureDetector(img2[L])

    #debugOrb(img1[L],kp1)
    #debugOrb(img2[L],kp2)

    
    #Feature Matching

    kp1_matched,kp2_matched = Matcher(img1[L],img2[L],kp1,des1,kp2,des2)

    #Feature Selection based on Disparity
    trackPoints1L_3d,trackPoints1R_3d,trackPoints2L_3d,trackPoints2R_3d = featureEliminator(img1_disparity,img2_disparity,kp1_matched,kp2_matched)

    
    # 3d triangulation
    numPoints = trackPoints1L_3d.shape[0]
    d3dPointsT1 = triangulate3D(trackPoints1L_3d,trackPoints1R_3d,numPoints,Proj1,Proj2)
    d3dPointsT2 = triangulate3D(trackPoints2L_3d,trackPoints2R_3d,numPoints,Proj1,Proj2)


    # Eliminate inliers
    W = generateAdjMatrix(d3dPointsT1,d3dPointsT2,0.01)

    #Max Clique TIME BOISSS

    clique = findMaxClique(W)


    # pick up clique point 3D coords and features for optimization
    cliqued3dPointT1 = d3dPointsT1[clique]
    cliqued3dPointT2 = d3dPointsT2[clique]

    # points = features
    trackedPoints1L = trackPoints1L_3d[clique]
    trackedPoints2L = trackPoints2L_3d[clique]

    dSeed = np.zeros(6)

    optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=2000,
                        args=(trackedPoints1L, trackedPoints2L, cliqued3dPointT1, cliqued3dPointT2, Proj1))

    
    error = optRes.fun
    pointsInClique = len(clique)
    e = error.reshape((pointsInClique*2, 3))
    errorThreshold = 1.0
    xRes1 = np.where(e[0:pointsInClique, 0] >= errorThreshold)
    yRes1 = np.where(e[0:pointsInClique, 1] >= errorThreshold)
    zRes1 = np.where(e[0:pointsInClique, 2] >= errorThreshold)
    xRes2 = np.where(e[pointsInClique:2*pointsInClique, 0] >= errorThreshold)
    yRes2 = np.where(e[pointsInClique:2*pointsInClique, 1] >= errorThreshold)
    zRes2 = np.where(e[pointsInClique:2*pointsInClique, 2] >= errorThreshold)

    pruneIdx = xRes1[0].tolist() + yRes1[0].tolist() + zRes1[0].tolist() + (xRes2[0] - pointsInClique).tolist() + (yRes2[0] - pointsInClique).tolist() +  (zRes2[0] - pointsInClique).tolist()
    if (len(pruneIdx) > 0):
        uPrundeIdx = list(set(pruneIdx))
        trackedPoints1_KLT_L = np.delete(trackedPoints1_KLT_L, uPrundeIdx, axis=0)
        trackedPoints2_KLT_L = np.delete(trackedPoints2_KLT_L, uPrundeIdx, axis=0)
        cliqued3dPointT1 = np.delete(cliqued3dPointT1, uPruneIdx, axis=0)
        cliqued3dPointT2 = np.delete(cliqued3dPointT2, uPruneIdx, axis=0)
        
        optRes = least_squares(minimizeReprojection, optRes.x, method='lm', max_nfev=2000,
                        args=(trackedPoints1_KLT_L, trackedPoints2_KLT_L, cliqued3dPointT1, cliqued3dPointT2, Proj1))
    

    print(optRes.x)
    
#clique size check
# reproj error check
# r, t generation
# plot on map vs ground truth









    

ImT1_L = cv2.imread('./000001.png', 0)    #0 flag returns a grayscale image
ImT1_R = cv2.imread('./1_000001.png', 0)

ImT2_L = cv2.imread('./000003.png', 0)
ImT2_R = cv2.imread('./1_000003.png', 0)

estimateOdometry([ImT1_L,ImT1_R],[ImT2_L,ImT2_R])


exit()