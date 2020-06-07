import numpy as np 

def Camera_params():
    #P_left
    P0 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
                [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]) 

    #P_right
    P1  = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02], 
                [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

    assert (P0[:3,:3]==P1[:3,:3]).all() # just checking if matrices are for rectified images

    R = np.eye(3) # rotation matrix between left and right camera views
    T = P1[:,3] # translation between the views
    K = P0[:3, :3] # calibration matrix / Camera matrix
    distCoeffs = np.zeros(5)
    return P0,P1,R,T,K,distCoeffs


