import numpy as np
import cv2

import glob
import os

import time
import argparse

from collections import defaultdict
from scipy.optimize import least_squares
from math import cos, sin

L = 0

R = 1

IMSIZE = (1241, 376)

def getGt(txt):
    """ 
    Parses calib.txt in Kitti dataset
  
    Parameters: 
    txt (string):  path to calib.txt file
  
    Returns: 
    2 np arrays containing left and right Projective matrix
    """
    with open(txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'P0' in line:
            p0 = line.split(':')[1][1:]
            p0 = p0.split(' ')
            assert len(p0)==12
            p0 = list(map(float, p0))
        elif 'P1' in line:
            p1 = line.split(':')[1][1:]
            p1 = p1.split(' ')
            assert len(p1)==12
            p1 = list(map(float, p1))

    return np.asarray(p0).reshape(3,4), np.asarray(p1).reshape(3,4)



def estDisparity(imgL,imgR,engine):
    """ 
    Calculates disparity using left and right images, using an engine with predescribed parameters
  
    Parameters: 
    imgL : Left Image
    imgR : Right Image
    Engine : Disparity engine to be used
  
    Returns: 
    disparity map
    """
    return np.divide(engine.compute(imgL,imgR).astype(np.float32),16.0)


def fastDetect(img1L,img2L,feature_detector = 0):
    """ 
    Feature detection
  
    Parameters: 
    img1L : Left image of first frame
    img2L : Left image of second frame
    feature detector  : 0 = FAST 1 = GFTT
  
    Returns: 
    matched features in left and right images
    """
    H,W = img1L.shape
    TILE_H = 10
    TILE_W = 20
    kp = []

    if(feature_detector == 0):
        featureEngine = cv2.FastFeatureDetector_create()
        
        for y in range(0, H, TILE_H):
            for x in range(0, W, TILE_W):
                imPatch = img1L[y:y+TILE_H, x:x+TILE_W]
                keypoints = featureEngine.detect(imPatch)
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
                if (len(keypoints) > 10):
                    keypoints = sorted(keypoints, key=lambda x: -x.response)
                    for kpt in keypoints[0:10]:
                        kp.append(kpt)
                else:
                    for kpt in keypoints:
                        kp.append(kpt)

    if(feature_detector == 1):
        featureEngine = cv2.GFTTDetector_create(maxCorners=4000, minDistance=8.0, qualityLevel=0.001, useHarrisDetector=False)
        keypoints = featureEngine.detect(img1L)
        
        for kpt in keypoints:
            kp.append(kpt)


    features1 = cv2.KeyPoint_convert(kp)
    features1 = np.expand_dims(features1, axis=1)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    features2, st, err = cv2.calcOpticalFlowPyrLK(img1L, img2L, features1, None, flags=cv2.MOTION_AFFINE, **lk_params)

    # separate points that were tracked successfully
    ptTrackable = np.where(st == 1, 1,0).astype(bool)
    features1_KLT = features1[ptTrackable, ...]
    features2_KLT = features2[ptTrackable, ...]
    features2_KLT = np.around(features2_KLT)

    # among tracked points take points within error measue
    error = 4
    errTrackablePoints = err[ptTrackable, ...]
    errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
    features1_KLT = features1_KLT[errThresholdedPoints, ...]
    features2_KLT = features2_KLT[errThresholdedPoints, ...]


    # check for validity of tracked point Coordinates
    hPts = np.where(features2_KLT[:,1] >= H)
    wPts = np.where(features2_KLT[:,0] >= W)
    outTrackPts = hPts[0].tolist() + wPts[0].tolist()
    outDeletePts = list(set(outTrackPts))

    if len(outDeletePts) > 0:
        features1_KLT_L = np.delete(features1_KLT, outDeletePts, axis=0)
        features2_KLT_L = np.delete(features2_KLT, outDeletePts, axis=0)
    else:
        features1_KLT_L = features1_KLT
        features2_KLT_L = features2_KLT

    return features1_KLT_L,features2_KLT_L

def featureEliminator(img1_disparity,img2_disparity,kp1_matched,kp2_matched,disparityMinThres = 0.0, disparityMaxThres = 100.0):
    """ 
    Calculate 3d positions of the keypoints and eliminates the illegal ones
  
    Parameters: 
    img1_disparity  : disparity map for first stereo pair
    img2_disparity  : disparity map for first stereo pair
    kp1_matched     : matched fetaures for 1st pair
    kp2_matched     : matched features for 2nd pair
    disparityMinThres: Minimum disparity that is considered legal
    disparityMaxThres: Maximum disparity that is considered legal
  
    Returns: 
    Legal features, and their positions on left and right images
    """
    kps1L = kp1_matched
    kps2L = kp2_matched

    kps1R = np.copy(kp1_matched)
    kps2R = np.copy(kp2_matched)

    valPoints = np.zeros(kps1L.shape[0])

    for i in range(kps1L.shape[0]):
        disp1 = img1_disparity[int(kps1L[i,1]), int(kps1L[i,0])]
        disp2 = img2_disparity[int(kps2L[i,1]), int(kps2L[i,0])]
        
        if (disp1 > disparityMinThres and disp1 < disparityMaxThres 
            and disp2 > disparityMinThres and disp2 < disparityMaxThres):
            kps1R[i, 0] = kps1L[i, 0] - disp1
            kps2R[i, 0] = kps2L[i, 0] - disp2
            valPoints[i] = 1
            
    valPoints = valPoints.astype(bool)

    return kps1L[valPoints, ...],kps1R[valPoints, ...],kps2L[valPoints, ...],kps2R[valPoints, ...]

def triangulate3D(features_L_3d,features_R_3d,numPoints,Proj1,Proj2):
    """ 
    Triangulate the points
  
    Parameters: 
    features_L_3d   : Features in Left image 
    features_R_3d   : Features in RIght Image
    numPoints       : Len of features
    Proj1           : K[R|T] matrix of left camera
    Proj2           : K[R|T] matrix of Right camera
  
    Returns: 
    3d points
    """

    Features3d = np.ones((numPoints,3))

    for i in range(numPoints):
        #for i in range(1):
        pLeft = features_L_3d[i,:]
        pRight = features_R_3d[i,:]
        
        X = np.zeros((4,4))
        X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
        X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
        X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
        X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]
        
        _,_,V = np.linalg.svd(X)

        Features3d[i, :] = (V[-1]/V[-1,-1]).T[:-1]

    return Features3d


def generateAdjMatrix(Features3d1,Features3d2,tolerance=0.2):
    """ 
    Generates adjacency matrix for clique detection
  
    Parameters: 
    Features3d1     : Features in 1st Image
    Features3d2     : Features in 2nd image
    tolerance       : allowed tolerance
  
    Returns: 
    Adjacency matrix 
    """

    numPoints = Features3d1.shape[0]
    W = np.zeros((numPoints, numPoints))

    # diff of pairwise euclidean distance between same points in T1 and T2
    for i in range(numPoints):
        for j in range(numPoints):
            T2Dist = np.linalg.norm(Features3d2[i,:] - Features3d2[j,:])
            T1Dist = np.linalg.norm(Features3d1[i,:] - Features3d1[j,:])
            if (abs(T2Dist - T1Dist) < tolerance):
                W[i, j] = 1

    return W

def findMaxClique(W):
    """ 
    Finds maximum clique
  
    Parameters: 
    W = Adjacency matrix
  
    Returns: 
    Points in maxclique using a greedy approach
    """
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
        
        if (len(clique) > 100):
            break   


    return clique



def generateRotMat(psi, theta, sigma):
    """ 
    generates Rotation matrix 
  
    Parameters: 
    psi, theta, sigma, rotations about x,y,z, directions
  
    Returns: 
    Rotation matrix
    """
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

def loss(dof,d2dPoints1, d2dPoints2, d3dPoints1, d3dPoints2, w2cMatrix):
    """ 
    generate cost for LM 
  
    Parameters: 
    dof         : optimizable parameters
    d2dPoints1  : points in image 1 in 2d
    d2dPoints2  : points in image 1 in 2d
    d3dPoints1  : points in image 2 in 3d
    d3dPoints2  : points in image 2 in 3d
    w2cMatrix   : Left projection matrix

    Returns: 
    loss
    """
    perspectiveProj = np.eye(4)
    Rmat = generateRotMat(dof[0], dof[1], dof[2])
    perspectiveProj[:3,:3] = Rmat
    perspectiveProj[0,-1] = dof[3]
    perspectiveProj[1,-1] = dof[4]
    perspectiveProj[2,-1] = dof[5]

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





def estimateOdometry(img1,img2,Proj1,Proj2,img1_disparity,img2_disparity,feature_detector):

    """ 
    main code, which is to be called every loop 
  
    Parameters: 
    img1                : first pair
    img2                : 2nd pair
    Proj1               : Projection for left cam
    Proj2               : Projection for right cam
    img1_disparity      : disparity for 1st frame
    img2_disparity      : disparity for 2nd frame
    feature_detector    : 0 = FAST, 1 = GFTT

    Returns: 
    6d pose, cost
    """

    #Feature detection

    features_1L,features_2L = fastDetect(img1[L],img2[L],feature_detector)

    features_1L_3d,features_1R_3d,features_2L_3d,features_2R_3d = featureEliminator(img1_disparity,img2_disparity,features_1L,features_2L)


    # 3d triangulation
    numPoints = features_1L_3d.shape[0]
    Points3d1 = triangulate3D(features_1L_3d,features_1R_3d,numPoints,Proj1,Proj2)
    Points3d2 = triangulate3D(features_2L_3d,features_2R_3d,numPoints,Proj1,Proj2)

    # Eliminate outliers

    tol = 0.2
    len_clique = 0
    clique = []
    while len_clique < 6 and Points3d1.shape[0] >= 6:
        # in-lier detection algorithm
        W = generateAdjMatrix(Points3d1,Points3d2,tol)
        clique = findMaxClique(W)
        len_clique = len(clique)
        tol *= 2

    
    # pick up clique point 3D coords and features for optimization
    PointsClique3d1 = Points3d1[clique]
    PointsClique3d2 = Points3d2[clique]

    # points = features
    features_1L_3d = features_1L_3d[clique]
    features_2L_3d = features_2L_3d[clique]



    if (features_1L_3d.shape[0] >= 6):
        dSeed = np.zeros(6)

        optRes = least_squares(loss, dSeed, method='lm', max_nfev=2000,
                            args=(features_1L_3d, features_2L_3d, PointsClique3d1, PointsClique3d2, Proj1))

        
        error = optRes.fun
        pointsInClique = len(clique)
        e = error.reshape((pointsInClique*2, 3))
        errorThreshold = 0.5
        xRes1 = np.where(e[0:pointsInClique, 0] >= errorThreshold)
        yRes1 = np.where(e[0:pointsInClique, 1] >= errorThreshold)
        zRes1 = np.where(e[0:pointsInClique, 2] >= errorThreshold)
        xRes2 = np.where(e[pointsInClique:2*pointsInClique, 0] >= errorThreshold)
        yRes2 = np.where(e[pointsInClique:2*pointsInClique, 1] >= errorThreshold)
        zRes2 = np.where(e[pointsInClique:2*pointsInClique, 2] >= errorThreshold)

        pruneIdx = xRes1[0].tolist() + yRes1[0].tolist() + zRes1[0].tolist() + (xRes2[0] - pointsInClique).tolist() + (yRes2[0] - pointsInClique).tolist() +  (zRes2[0] - pointsInClique).tolist()
        if (len(pruneIdx) > 0):
            uPruneIdx = list(set(pruneIdx))
            features_1L_3d = np.delete(features_1L_3d, uPruneIdx, axis=0)
            features_2L_3d = np.delete(features_2L_3d, uPruneIdx, axis=0)
            PointsClique3d1 = np.delete(PointsClique3d1, uPruneIdx, axis=0)
            PointsClique3d2 = np.delete(PointsClique3d2, uPruneIdx, axis=0)
            
            if (features_1L_3d.shape[0] >= 6):
                optRes = least_squares(loss, optRes.x, method='lm', max_nfev=2000,
                            args=(features_1L_3d, features_2L_3d, PointsClique3d1, PointsClique3d2, Proj1))
        

        return optRes.x, optRes.cost

    return [0,0,0,0,0,0],0



if __name__=='__main__':


    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seqPath",required = True,type = str,help='path to kitti sequence directory, for example, ./svo/dataset/sequences/00/')
    parser.add_argument("-f", "--feature",default = 'FAST',type = str,choices = ['GFTT','FAST'],help='which feature detector is to be used?')
    parser.add_argument("-r", "--resultsPath",required = True,type = str,help='path to file where results are to be stored in kitti format, for example ./results/00.txt')
    args = parser.parse_args()
    
    seq = (args.seqPath)
    feat = args.feature

    if(feat == 'GFTT'):
        feat = 1
    else:
        feat = 0


    start = time.time()
    
    
    PATH = args.seqPath

    
    try:
        P0L, P0R = getGt(PATH + 'calib.txt')
    except:
        print("CALIB FILE NOT FOUND")
        exit(1)

    
    
    assert (P0L[:,-1]==np.zeros(3)).all() == 1 # first camera translation is zero
    #K = P0L[:,:-1]

    curr_transX = 0.0
    curr_transZ = 0.0

    f = open(args.resultsPath,"w")

    RmatGlobal = np.eye(3)

    transGlobal = np.zeros(3)
    
    l = len(glob.glob(PATH + "image_0/*"))
    print("Number of images detected : ",l)
    if(l==0):
        print("NO IMAGES FOUND AT ",os.path.join(PATH,"image_0"))
        exit(1)

    disparityEngine = cv2.StereoSGBM_create(minDisparity = 0,numDisparities=32, blockSize=11,P1= 121*8, P2 = 121*32)


    for i in range(1,l):
        path1L = PATH + "image_0/" + str(i-1).zfill(6) + ".png"
        path1R = PATH + "image_1/" + str(i-1).zfill(6) + ".png"
        path2L = PATH + "image_0/" + str(i).zfill(6) + ".png"
        path2R = PATH + "image_1/" + str(i).zfill(6) + ".png"


        if(i==1):
            imageLeft1 = cv2.imread(path1L, 0)   
            imageRight1 = cv2.imread(path1R, 0)
            img1_disparity = estDisparity(imageLeft1,imageRight1,disparityEngine)
        else:
            imageLeft1 =imageLeft2
            imageRight1 =imageRight2
            img1_disparity = img2_disparity

        imageLeft2 = cv2.imread(path2L, 0)
        imageRight2 = cv2.imread(path2R, 0)
        img2_disparity = estDisparity(imageLeft2,imageRight2,disparityEngine)

        dofs, cost = estimateOdometry([imageLeft1,imageRight1],[imageLeft2,imageRight2],P0L,P0R,img1_disparity,img2_disparity,feat)


        transLocal = np.array([dofs[3],dofs[4],dofs[5]])
        RmatLocal  = generateRotMat(dofs[0],dofs[1],dofs[2])

        transGlobal += RmatGlobal@transLocal

        RmatGlobal = RmatLocal@RmatGlobal

        print("I : ",i)
        #print("X : ",transGlobal[0],"  Y : ", transGlobal[1], "  Z : ", transGlobal[2])
        f.write(
                 str(RmatGlobal[0,0]) + " " + str(RmatGlobal[0,1]) + " " + str(RmatGlobal[0,2]) + " " + str(transGlobal[0]) + " " 
               + str(RmatGlobal[1,0]) + " " + str(RmatGlobal[1,1]) + " " + str(RmatGlobal[1,2]) + " " + str(transGlobal[1]) + " " 
               + str(RmatGlobal[2,0]) + " " + str(RmatGlobal[2,1]) + " " + str(RmatGlobal[2,2]) + " " + str(transGlobal[2]) + "\n")

    f.close()
    end = time.time()
    print("Time Per frame = ",(end - start)/l)





















