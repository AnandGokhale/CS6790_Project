import numpy as np
import cv2

from scipy.optimize import least_squares
from math import cos, sin

L = 0

R = 1

IMSIZE = (1241, 376)

def getGt(txt):
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
    return np.divide(engine.compute(imgL,imgR).astype(np.float32),16.0)


def fastDetect(img1L,img2L):
    H,W = img1L.shape
    TILE_H = 10
    TILE_W = 20
    kp = []

    featureEngine = cv2.FastFeatureDetector_create()

    for y in range(0, H, TILE_H):
        for x in range(0, W, TILE_W):
            imPatch = ImT1_L[y:y+TILE_H, x:x+TILE_W]
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

    trackPoints1 = cv2.KeyPoint_convert(kp)
    trackPoints1 = np.expand_dims(trackPoints1, axis=1)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(img1L, img2L, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)

    # separate points that were tracked successfully
    ptTrackable = np.where(st == 1, 1,0).astype(bool)
    trackPoints1_KLT = trackPoints1[ptTrackable, ...]
    trackPoints2_KLT_t = trackPoints2[ptTrackable, ...]
    trackPoints2_KLT = np.around(trackPoints2_KLT_t)

    # among tracked points take points within error measue
    error = 4
    errTrackablePoints = err[ptTrackable, ...]
    errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
    trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
    trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]


    # check for validity of tracked point Coordinates
    hPts = np.where(trackPoints2_KLT[:,1] >= H)
    wPts = np.where(trackPoints2_KLT[:,0] >= W)
    outTrackPts = hPts[0].tolist() + wPts[0].tolist()
    outDeletePts = list(set(outTrackPts))

    if len(outDeletePts) > 0:
        trackPoints1_KLT_L = np.delete(trackPoints1_KLT, outDeletePts, axis=0)
        trackPoints2_KLT_L = np.delete(trackPoints2_KLT, outDeletePts, axis=0)
    else:
        trackPoints1_KLT_L = trackPoints1_KLT
        trackPoints2_KLT_L = trackPoints2_KLT

    return trackPoints1_KLT_L,trackPoints2_KLT_L

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


def generateAdjMatrix(d3dPointsT1,d3dPointsT2,distDifference=0.2):
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
        
        if (len(clique) > 100):
            break   


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

    # print (perspectiveProj)

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





def estimateOdometry(img1,img2,Proj1,Proj2):
    #img1 = [img1L,img1R]
    #img2 = [img2L,img2R]

    #Prepocessing the image
    img1[L]  = cv2.resize(img1[L], IMSIZE) 
    img1[R]  = cv2.resize(img1[R], IMSIZE) 
    img2[L]  = cv2.resize(img2[L], IMSIZE) 
    img2[R]  = cv2.resize(img2[R], IMSIZE) 


    block = 11
    P1 = block*block*8
    P2 = block*block*32

    disparityEngine = cv2.StereoSGBM_create(minDisparity = 0,numDisparities=32, blockSize=block,P1= P1, P2 = P2)


    img1_disparity = estDisparity(img1[L],img1[R],disparityEngine)
    img2_disparity = estDisparity(img2[L],img2[R],disparityEngine)

    #Feature detection

    trackPoints1_KLT_L,trackPoints2_KLT_L = fastDetect(img1[L],img2[L])




    trackPoints1L_3d,trackPoints1R_3d,trackPoints2L_3d,trackPoints2R_3d = featureEliminator(img1_disparity,img2_disparity,trackPoints1_KLT_L,trackPoints2_KLT_L)


    # 3d triangulation
    numPoints = trackPoints1L_3d.shape[0]
    d3dPointsT1 = triangulate3D(trackPoints1L_3d,trackPoints1R_3d,numPoints,Proj1,Proj2)
    d3dPointsT2 = triangulate3D(trackPoints2L_3d,trackPoints2R_3d,numPoints,Proj1,Proj2)

    # Eliminate inliers
    # 
    distDifference = 0.2
    lClique = 0
    clique = []
    while lClique < 6 and d3dPointsT1.shape[0] >= 6:
        # in-lier detection algorithm
        W = generateAdjMatrix(d3dPointsT1,d3dPointsT2,distDifference)
        clique = findMaxClique(W)
        lClique = len(clique)
        distDifference *= 2

    
    # pick up clique point 3D coords and features for optimization
    cliqued3dPointT1 = d3dPointsT1[clique]
    cliqued3dPointT2 = d3dPointsT2[clique]

    # points = features
    trackedPoints1L = trackPoints1L_3d[clique]
    trackedPoints2L = trackPoints2L_3d[clique]



    if (trackedPoints1L.shape[0] >= 6):
        dSeed = np.zeros(6)

        optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=2000,
                            args=(trackedPoints1L, trackedPoints2L, cliqued3dPointT1, cliqued3dPointT2, Proj1))

        
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
            trackedPoints1L = np.delete(trackedPoints1L, uPruneIdx, axis=0)
            trackedPoints2L = np.delete(trackedPoints2L, uPruneIdx, axis=0)
            cliqued3dPointT1 = np.delete(cliqued3dPointT1, uPruneIdx, axis=0)
            cliqued3dPointT2 = np.delete(cliqued3dPointT2, uPruneIdx, axis=0)
            
            if (trackedPoints1L.shape[0] >= 6):
                optRes = least_squares(minimizeReprojection, optRes.x, method='lm', max_nfev=2000,
                            args=(trackedPoints1L, trackedPoints2L, cliqued3dPointT1, cliqued3dPointT2, Proj1))
        

        return optRes.x, optRes.cost

    return [0,0,0,0,0,0],0



if __name__=='__main__':
    P0L, P0R = getGt('../01/01/calib.txt')
    P1L, P1R = getGt('../01/01/calib.txt') # same calibration file
    # print(P0)
    # print(P1)
    
    assert (P0L[:,-1]==np.zeros(3)).all() == 1 # first camera translation is zero
    K = P0L[:,:-1]

    '''
    ImT1_L = cv2.imread('./000001.png', 0)    #0 flag returns a grayscale image
    ImT1_R = cv2.imread('./1_000001.png', 0)

    ImT2_L = cv2.imread('./000003.png', 0)
    ImT2_R = cv2.imread('./1_000003.png', 0)

    dofs, cost = estimateOdometry([ImT1_L,ImT1_R],[ImT2_L,ImT2_R])
    print('Residual error :', cost)
    R = genEulerZXZMatrix(dofs[0], dofs[1], dofs[2])
    t = np.array(dofs[3:])

    inds = [0,2]
    poses = getPose('./dataset/poses/00.txt', inds)

    R1 = np.linalg.inv(K)@poses[0][:,:-1]
    t1 = np.linalg.inv(K)@poses[0][:,-1]

    R2 = np.linalg.inv(K)@poses[1][:,:-1]
    t2 = np.linalg.inv(K)@poses[1][:,-1]

    R_rel = R1.T@R2
    t_rel = t2-t1

    t_err, theta = getErrors(R, R_rel, t, t_rel)
    print('Translation error :', t_err, '\t Rotation error:', theta)

    '''

    curr_transX = 0.0
    curr_transZ = 0.0

    f = open("tryingrecur.txt","w")

    RmatGlobal = np.eye(3)

    transGlobal = np.zeros(3)

    for i in range(1,1101):
        path1L = "../01/01/image_0/" + str(i-1).zfill(6) + ".png"
        path1R = "../01/01/image_1/" + str(i-1).zfill(6) + ".png"
        path2L = "../01/01/image_0/" + str(i).zfill(6) + ".png"
        path2R = "../01/01/image_1/" + str(i).zfill(6) + ".png"

        ImT1_L = cv2.imread(path1L, 0)    #0 flag returns a grayscale image
        ImT1_R = cv2.imread(path1R, 0)

        ImT2_L = cv2.imread(path2L, 0)
        ImT2_R = cv2.imread(path2R, 0)
        dofs, cost = estimateOdometry([ImT1_L,ImT1_R],[ImT2_L,ImT2_R],P0L,P0R)


        transLocal = np.array([dofs[3],dofs[4],dofs[5]])
        RmatLocal  = genEulerZXZMatrix(dofs[0],dofs[1],dofs[2])

        transGlobal += RmatGlobal@transLocal

        RmatGlobal = RmatLocal@RmatGlobal

        print("I : ",i)
        print("X : ",transGlobal[0],"  Y : ", transGlobal[1], "  Z : ", transGlobal[2])
        f.write("0.0 0.0 0.0 " + str(transGlobal[0]) + " 0.0 0.0 0.0 " + str(transGlobal[1]) + " 0.0 0.0 0.0 " + str(transGlobal[2]) + "\n")

    f.close()