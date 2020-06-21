import numpy as np 
import cv2 
from numpy.linalg import svd
def get3Dpoints(pts1, pts2, P1, P2):
    Xs = np.array((len(pts1, 3)))
    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]
        A = np.empty((4,4))
        A[0, :] = pt1[0]*P1[2,:] - P1[0,:]
        A[1, :] = pt1[1]*P1[2,:] - P1[1,:]
        A[2, :] = pt2[0]*P2[2,:] - P2[0,:]
        A[3, :] = pt2[1]*P2[2,:] - P2[1,:] 
        u, d, v = svd(A)
        X = v[-1]
        X = X/X[-1]
        Xs[i] = X[:-1]
    return Xs

def filterKps(kps, disparity):
        '''Filter out keypoints which have a known disparity value'''
        inds =  [i for i in range(len(kps)) if disparity[int(kps[i].pt[1]), int(kps[i].pt[0])]!=-1]
        return inds

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


def getMatches(kps1, des1, kps2, des2):
        ''' Given keypoints and descriptors of two images, returns a set of matched keypoints'''
        # list of matched keypoints
        kps1_m = []
        kps2_m = []

        # flann matching / A3
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)

        for (m,_) in matches:
                kps2_m.append(kps2[m.trainIdx])
                kps1_m.append(kps1[m.queryIdx])
        return kps1_m, kps2_m

def getErrors(R_est, R_gt, t_est, t_gt):
        '''Returns translation error (in same units as translation vector)  and rotation error (in degrees)'''
        t_err = np.linalg.norm(t_est-t_gt, ord=2)
        cos_theta =  0.5*(np.trace(R_gt.T@R_est)-1) # axis angle representation of residual rotation 
        return t_err, np.arccos(cos_theta)*180/np.pi

def getCorrespondence(pts, disparity):
        '''Given a set of valid points, find the corresponding points in the right camera frame
        using the disp map
        
        valid points - points with known disparity
        '''
        disp_vals = disparity[tuple(pts[:,::-1].T)]
        return pts - np.c_[disp_vals, np.zeros_like(disp_vals)]

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


def getPose(txt, inds):
    '''Get the poses of cameras indexed by inds from the txt file'''
    # assert type(inds) == np.ndarray, f"an array of indices must be passed, got {type(inds)}"
    with open(txt, 'r') as f:
        lines = f.readlines()
    cam_lines = [lines[ind] for ind in inds]
    poses = []
    for line in cam_lines:
        pose = list(map(float, line.split(' ')))
        poses.append(np.asarray(pose).reshape(3, 4))
    return poses 