#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import glob
import os
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
np.set_printoptions(suppress=True)
global_path = "/media/eugeniu/T7/calibration/calib/" #//change this to your system

#Basler RGB intrinsics
# K = np.array(   [[1396.42642353,    0. ,         986.87440169],
#                 [   0.  ,       1398.23028572  ,607.60275135],
#                 [   0.       ,     0. ,           1.        ]])

# D = np.array( [[-0.12242753 , 0.05002637 , 0.00010842 , 0.00189244 , 0.03743694]])

#read intrinsic of each thermal
intr_thermal_left = global_path+"intrinsic_left_thermal_ST.npz"
intr_thermal_center = global_path+"intrinsic_center_thermal_ST.npz"
intr_thermal_right = global_path+"intrinsic_right_thermal_ST.npz"

extrinsic_left = global_path + "extrinsic_left_thermal_to_baseler.npz"
extrinsic_center = global_path + "extrinsic_center_thermal_to_baseler.npz"
extrinsic_right = global_path + "extrinsic_right_thermal_to_baseler.npz"


#an example for the left thermal camera - do the same 

#the dict contains
# np.savez(
#             save_path,
#             K=K,                  intrinsic matrix
#             dist=dist,            distortion
#             rvecs=np.array(rvecs, dtype=object),   
#             tvecs=np.array(tvecs, dtype=object),
#             object_points=np.array(object_points, dtype=object),      chessboard 3d points
#             image_points=np.array(image_points, dtype=object),        chessboard 2d points
#             errors=errors                                             calibration errors 
#         )

left_thermal_intrinsic = np.load(intr_thermal_left, allow_pickle=True)
#the extrinsic calibration has the following dictionary ---------------------------------------------------------

# np.savez(
#             save_result,
#             objpoints = np.array(objpoints, dtype=object),
#             imgpoints_l = np.array(imgpoints_l, dtype=object),
#             imgpoints_r = np.array(imgpoints_r, dtype=object),
#             K_thermal = K_thermal,
#             D_thermal = D_thermal,
#             K_baseler=K,
#             D_baseler=D,
#             R = R,
#             T = T,
#             E = E,
#             F = F
#     )

# loaded = np.load(extrinsic_left, allow_pickle=True)

# K_thermal = loaded["K_thermal"]  #intrinsic matrix of thermal cam
# D_thermal = loaded["D_thermal"]  #distortion of thermal cam
# K_baseler = loaded["K_baseler"]  
# D_baseler = loaded["D_baseler"]
# R = loaded["R"]                  #extrinsic rotation from thermal cam to basler camera
# T = loaded["T"]                  #extrinsic translation from thermal cam to basler camera
 
#print("D_thermal ", D_thermal)

# D_thermal[0,-1] = 0
# print("D_thermal ", D_thermal)


square = 0.1  # m (the size of each chessboard square is 10cm)
objp = np.zeros((10 * 7, 3), np.float32) #chessboard is 7x10
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square
term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)

 #the following will show a test for extrinsic of thermal left with baseler
def test_test_thermal_to_basler():
    loaded = np.load(extrinsic_left, allow_pickle=True)
    K_thermal = loaded["K_thermal"]
    D_thermal = loaded["D_thermal"]   
    R = loaded["R"]
    T = loaded["T"]

    thermal_img, basler_img = global_path+"some_data_left/thermal_image_0001.npy", global_path+"some_data_left/baseler_image_0001.npy"
    thermal_img, basler_img = np.load(thermal_img), np.load(basler_img)
      
    valid_pixels = thermal_img[thermal_img > 0]
    min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
    max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
    img_clipped = np.clip(thermal_img, min_val, max_val)
    img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
    #invert_img_b = np.array(255.0 - img8, dtype='uint8')
    invert_img_b = img8
                
    ret_t, corners_t = cv2.findChessboardCorners(invert_img_b, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_t:
        cv2.drawChessboardCorners(invert_img_b, (10, 7), corners_t, ret_t)


    ret_b, cornersR = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    #if ret_b:
    #    cv2.drawChessboardCorners(basler_img, (10, 7), cornersR, ret_b)

    if ret_t and ret_b:
        success, rvecL, tvecL = cv2.solvePnP(objp, corners_t, K_thermal, D_thermal)

        # Convert rvecL to rotation matrix
        R_chess_L, _ = cv2.Rodrigues(rvecL)

        # Homogeneous transform of chessboard in left frame
        T_chess_L = np.eye(4)
        T_chess_L[:3,:3] = R_chess_L
        T_chess_L[:3, 3] = tvecL.ravel()

        # Stereo extrinsics: transform from left -> right
        T_RL = np.eye(4)
        T_RL[:3,:3] = R  # rotation from left to right
        T_RL[:3, 3]  = T.ravel()  # translation from left to right

        # Transform chessboard pose to right camera frame
        T_chess_R = T_RL @ T_chess_L
        R_chess_R = T_chess_R[:3,:3]
        t_chess_R = T_chess_R[:3, 3]

        # Project 3D object points into right image
        proj_points, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_chess_R)[0], 
                                   t_chess_R, K, D)
            
        if ret_b:
            # cornersR = cv2.cornerSubPix(basler_img, cornersR, (11,11), (-1,-1),
            #                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                
            error = cv2.norm(cornersR, proj_points, cv2.NORM_L2) / len(proj_points)
            print("Reprojection error between predicted and detected corners:", error)
            cv2.drawChessboardCorners(basler_img, (10, 7), proj_points, ret_b)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b)
        
        key = cv2.waitKey(0)


    cv2.destroyAllWindows()

#test_test_thermal_to_basler()

def invert_transform(R, t):
    """
    Invert a rigid body transform (R, t).

    Parameters
    ----------
    R : (3,3) ndarray
        Rotation matrix
    t : (3,) ndarray
        Translation vector

    Returns
    -------
    R_inv : (3,3) ndarray
        Inverse rotation matrix
    t_inv : (3,) ndarray
        Inverse translation vector
    """
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv

#project from basler to thermal
def test_basler_to_thermal():
    # loaded = np.load(extrinsic_left, allow_pickle=True)  #0.16

    '''
    for center camera - extrinsics
    with intrinsics as they are - 0.14 px reprojection error
    with fx = fy and k3 = 0     - 0.151 px
    same but with weighted      - 0.150 px
    '''

    '''
    for left camera - extrinsics
    with intrinsics as they are -  0.16 reprojection error
    with fx = fy and k3 = 0     -  0.173 px
    same but with weighted      -  0.15 px
    '''
    extrinsic_center = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/extrinsic_left_thermal_to_baseler.npz"

    format = ""
    extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_left_thermal_to_baseler_{}.npz".format(format)
    format = "weighted"
    extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_left_thermal_to_baseler_{}.npz".format(format)
    format = "weighted"
    extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_left_thermal_to_baseler_{}2.npz".format(format)


    loaded = np.load(extrinsic_center, allow_pickle=True)  #0.16


    K = loaded["K_baseler"]
    D = loaded["D_baseler"]
    K_thermal = loaded["K_thermal"]
    D_thermal = loaded["D_thermal"]   
    R = loaded["R"]
    T = loaded["T"]
    print("R:\n",R)
    print("T:\n",T)
    print("K_thermal:\n",K_thermal)
    print("before D_thermal:", D_thermal)

    # thermal_img, basler_img = global_path+"some_data_center/thermal_image_0001.npy", global_path+"some_data_center/baseler_image_0001.npy"

    thermal_img, basler_img = global_path+"some_data_left/thermal_image_0001.npy", global_path+"some_data_left/baseler_image_0001.npy"
    thermal_img, basler_img = np.load(thermal_img), np.load(basler_img)
      
    valid_pixels = thermal_img[thermal_img > 0]
    min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
    max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
    img_clipped = np.clip(thermal_img, min_val, max_val)
    img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
    #invert_img_b = np.array(255.0 - img8, dtype='uint8')
    invert_img_b = img8
                
    ret_t, corners_t = cv2.findChessboardCorners(invert_img_b, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    invert_img_b = cv2.cvtColor(invert_img_b.copy(),cv2.COLOR_GRAY2RGB)
    # if ret_t:
    #     cv2.drawChessboardCorners(invert_img_b, (10, 7), corners_t, ret_t)


    ret_b, cornersR = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_b:
       cv2.drawChessboardCorners(basler_img, (10, 7), cornersR, ret_b)

    if ret_t and ret_b:
        success, rvecR, tvecR = cv2.solvePnP(objp, cornersR, K, D)

        # Convert rvecR to rotation matrix
        R_chess_R, _ = cv2.Rodrigues(rvecR)

        # Homogeneous transform of chessboard in right frame
        T_chess_R = np.eye(4)
        T_chess_R[:3,:3] = R_chess_R
        T_chess_R[:3, 3] = tvecR.ravel()

        # Stereo extrinsics: transform from right -> left
        T_LR = np.eye(4)
        R_inv, t_inv = invert_transform(R = R, t = T)
        T_LR[:3,:3] = R_inv  # rotation from right to left
        T_LR[:3, 3]  = t_inv.ravel()  # translation from right to left

        # Transform chessboard pose to left camera frame
        T_chess_L = T_LR @ T_chess_R
        R_chess_L = T_chess_L[:3,:3]
        t_chess_L = T_chess_L[:3, 3]

        # Project 3D object points into right image
        proj_points, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_chess_L)[0], 
                                   t_chess_L, K_thermal, D_thermal)
            
        if ret_b:
            # cornersR = cv2.cornerSubPix(basler_img, cornersR, (11,11), (-1,-1),
            #                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                
            error = cv2.norm(corners_t, proj_points, cv2.NORM_L2) / len(proj_points)
            print("Reprojection error between predicted and detected corners:", error)
            cv2.drawChessboardCorners(invert_img_b, (10, 7), proj_points, ret_b)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b)
        
        key = cv2.waitKey(0)


    cv2.destroyAllWindows()

test_basler_to_thermal()

#project from basler to thermal - that is out of thermal FOV
def test_basler_to_thermal_out_fov():
    loaded = np.load(extrinsic_left, allow_pickle=True)
    #loaded = np.load(extrinsic_right, allow_pickle=True)
    K_thermal = loaded["K_thermal"]
    D_thermal = loaded["D_thermal"]   
    R = loaded["R"]
    T = loaded["T"]
    
    print("K_thermal:\n",K_thermal)

    print("before D_thermal:", D_thermal)

    #D_thermal[0][-1] = 0
    #D_thermal[0][-1] = 100
    #print("D_thermal:", D_thermal)

    

    thermal_img, basler_img = global_path+"some_data_left/out_fov/tehrmal_left.npy", global_path+"some_data_left/out_fov/basler.npy"
    #thermal_img, basler_img = global_path+"some_data_left/thermal_image_0001.npy", global_path+"some_data_left/out_fov/basler.npy"

    thermal_img, basler_img = np.load(thermal_img), np.load(basler_img)
      
    valid_pixels = thermal_img[thermal_img > 0]
    min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
    max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
    img_clipped = np.clip(thermal_img, min_val, max_val)
    img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
    #invert_img_b = np.array(255.0 - img8, dtype='uint8')
    invert_img_b = img8
                
    ret_t, corners_t = cv2.findChessboardCorners(invert_img_b, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # if ret_t:
    #     cv2.drawChessboardCorners(invert_img_b, (10, 7), corners_t, ret_t)


    ret_b, cornersR = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_b:
       cv2.drawChessboardCorners(basler_img, (10, 7), cornersR, ret_b)

    if (ret_t and ret_b) or True:
        success, rvecR, tvecR = cv2.solvePnP(objp, cornersR, K, D)

        # Convert rvecR to rotation matrix
        R_chess_R, _ = cv2.Rodrigues(rvecR)

        # Homogeneous transform of chessboard in right frame
        T_chess_R = np.eye(4)
        T_chess_R[:3,:3] = R_chess_R
        T_chess_R[:3, 3] = tvecR.ravel()

        # Stereo extrinsics: transform from right -> left
        T_LR = np.eye(4)
        R_inv, t_inv = invert_transform(R = R, t = T)
        T_LR[:3,:3] = R_inv  # rotation from right to left
        T_LR[:3, 3] = t_inv.ravel()  # translation from right to left

        # Transform chessboard pose to left camera frame
        T_chess_L = T_LR @ T_chess_R
        R_chess_L = T_chess_L[:3,:3]
        t_chess_L = T_chess_L[:3, 3]

        # Project 3D object points into right image
        proj_points, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_chess_L)[0], 
                                   t_chess_L, K_thermal, D_thermal)
        
        cv2.drawChessboardCorners(invert_img_b, (10, 7), proj_points, ret_t)
        
        #print("proj_points:\n",proj_points)
        if ret_t:
            # cornersR = cv2.cornerSubPix(basler_img, cornersR, (11,11), (-1,-1),
            #                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                
            error = cv2.norm(corners_t, proj_points, cv2.NORM_L2) / len(proj_points)
            print("Reprojection error between predicted and detected corners:", error)
            #cv2.drawChessboardCorners(invert_img_b, (10, 7), proj_points, ret_t)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b)
        
        key = cv2.waitKey(0)


    cv2.destroyAllWindows()

#test_basler_to_thermal_out_fov()