#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import glob
import os
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt

#  Set print options to avoid scientific notation
np.set_printoptions(suppress=True)

global_path = "/media/eugeniu/T7/calibration/saved_data_raw/" #//change this to your system

# #Basler RGB
# K = np.array(   [[1396.42642353,    0. ,         986.87440169],
#                 [   0.  ,       1398.23028572  ,607.60275135],
#                 [   0.       ,     0. ,           1.        ]])

# D = np.array( [[-0.12242753 , 0.05002637 , 0.00010842 , 0.00189244 , 0.03743694]])

# # intrinsic matrix
# K = np.array([[1363.18778612,    0.,          978.05715426],
#                       [   0.,         1362.77700097,  607.65710195],
#                       [   0.,            0.,            1.        ]])
# # distortion
# D = np.array([-0.15425933, 0.13876932, -0.00066874, 0.00093961, -0.06687253])

K = np.array([[1363.18778612,    0.,          978.05715426],
                      [   0.,         1362.77700097,  607.65710195],
                      [   0.,            0.,            1.        ]])

D = np.array( [[-0.12242753 , 0.05002637 , 0.00010842 , 0.00189244 , 0.03743694]])


# #0.1 px reprojection error
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/center/"
# intr_thermal_ = global_path+"intrinsic_center_thermal_ST.npz"
# save_result = save_dir+"extrinsic_center_thermal_to_baseler.npz"

# #right - 0.17 px reprojection error
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/right/"
# intr_thermal_ = global_path+"intrinsic_right_thermal_ST.npz"
# save_result = save_dir+"extrinsic_right_thermal_to_baseler.npz"

# # left - - 0.12 px reprojection error 
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/"
intr_thermal_ = global_path+"intrinsic_left_thermal_ST.npz"
# save_result = save_dir+"extrinsic_left_thermal_to_baseler.npz"





format = "" # 
format = "weighted"

#left thermal 
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/"
intr_thermal_ = "/media/eugeniu/T7/calibration/a_new_results/0/intrinsic_left_thermal_{}.npz".format(format)

intr_thermal_ = "/media/eugeniu/T7/calibration/a_new_results/3/intrinsic_left_thermal_{}.npz".format(format)
# intr_thermal_ = "/media/eugeniu/T7/calibration/a_new_results/intrinsic_left_thermal_{}.npz".format(format)

save_result = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_left_thermal_to_baseler_{}.npz".format(format)
save_result2 = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_left_thermal_to_baseler_{}2.npz".format(format)


# #center thermal 
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/center/"
# intr_thermal_ = "/media/eugeniu/T7/calibration/a_new_results/0/intrinsic_center_thermal_{}.npz".format(format)
# save_result = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_center_thermal_to_baseler_{}.npz".format(format)


# #right thermal 
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/right/"
# intr_thermal_ = "/media/eugeniu/T7/calibration/a_new_results/0/intrinsic_right_thermal_{}.npz".format(format)
# intr_thermal_ = "/media/eugeniu/T7/calibration/a_new_results/3/intrinsic_right_thermal_{}.npz".format(format)

# save_result = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_right_thermal_to_baseler_{}.npz".format(format)
# save_result2 = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_right_thermal_to_baseler_{}2.npz".format(format)


thermal_intrinsic = np.load(intr_thermal_, allow_pickle=True)
K_thermal = thermal_intrinsic["K"]
D_thermal = thermal_intrinsic["dist"]

print("K_thermal:\n",K_thermal)
print("D_thermal:\n",D_thermal.transpose())
# K_thermal = np.array( [[1612.26241364 ,   0.   ,       409.91189903],
#  [   0.   ,      1607.40899763,  227.9029271 ],
#  [   0.    ,        0.         ,   1.        ]])
# D_thermal = np.array([[-0.13352532 , 0.02932648, -0.00101115,  0.01103758 , 0.        ]])


b_path, t_path =  save_dir + "basler/", save_dir + "thermal/"

def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    return [np.load(os.path.join(folder, f)) for f in files]

images_16bit = load_images(t_path)
basler = load_images(b_path)

objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane. - thermal camera
imgpoints_r = []  # 2d points in image plane. - rgb camera

square = 0.1  # m (the size of each chessboard square is 10cm)
objp = np.zeros((10 * 7, 3), np.float32) #chessboard is 7x10
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square

term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)
def axisEqual3D_init(ax,data, cube = 5):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(data, axis=0)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    x_min, x_max = centers[0] - cube, centers[0]+cube
    y_min, y_max = centers[1] - cube, centers[1]+cube
    z_min, z_max = centers[2] - cube, centers[2]+cube

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

def align_svd(pc1, pc2):
    mu_pc1 = np.mean(pc1, axis=1)
    mu_pc2 = np.mean(pc2, axis=1)

    pc1_norm = pc1 - mu_pc1.reshape(-1, 1)
    pc2_norm = pc2 - mu_pc2.reshape(-1, 1)

    W = np.matmul(pc2_norm, pc1_norm.T)                         # calculate cross-covariance
    u, s, v_T = np.linalg.svd(W, full_matrices=True)            # decompose using SVD

    R = np.matmul(v_T.T, u.T)                                   # calculate rotation
    T = mu_pc1 - np.matmul(R, mu_pc2)                           # calculate translation

    return T.reshape(-1,1), R  # translation and rotation

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    
def stereo_calibrate():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    all_thermal_3d, all_rgb_3d = [],[]
    
    i=0
    for i in range(0, 33):
        basler_img = cv2.imread("/media/eugeniu/T7/calibration/a_prev_results/Thermal_overlay_static_example/chess/extrinsic/left/{}_rgb.jpg".format(i))
        
        invert_img_b = cv2.imread("/media/eugeniu/T7/calibration/a_prev_results/Thermal_overlay_static_example/chess/extrinsic/left/{}_left.jpg".format(i))
        invert_img_b = cv2.bitwise_not(invert_img_b)

        invert_img_b_Draw = invert_img_b.copy()
        ret_t, corners_t = cv2.findChessboardCorners(invert_img_b, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_t:
            invert_img_b = cv2.cvtColor(invert_img_b.copy(),cv2.COLOR_RGB2GRAY)
            corners_t = cv2.cornerSubPix(invert_img_b, corners_t, (11,11), (-1,-1),
                                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            
            
            cv2.drawChessboardCorners(invert_img_b_Draw, (10, 7), corners_t, ret_t)
            invert_img_b = cv2.cvtColor(invert_img_b.copy(),cv2.COLOR_GRAY2RGB)
        else:
            print("cannot detect the chessboard...")
        
        basler_img = cv2.cvtColor(basler_img.copy(),cv2.COLOR_RGB2GRAY)

        ret_b, corners_b = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_b:
            corners_b = cv2.cornerSubPix(basler_img, corners_b, (11,11), (-1,-1),
                                              (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            basler_img = cv2.cvtColor(basler_img.copy(),cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(basler_img, (10, 7), corners_b, ret_b)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b_Draw)
        cv2.waitKey(20)

        if ret_t and ret_b:
            objpoints.append(objp)
            imgpoints_l.append(corners_t)
            imgpoints_r.append(corners_b)

            # Find the rotation and translation vectors.
            # corners_t = cv2.undistortPoints(corners_t, K_thermal, D_thermal)
            # corners_b = cv2.undistortPoints(corners_b, K, D)

            thermal_ret, thermal_rvecs, thermal_tvecs = cv2.solvePnP(objp, corners_t, K_thermal, D_thermal)
            rgb_ret, rgb_rvecs, rgb_tvecs = cv2.solvePnP(objp, corners_b, K, D)
            if thermal_ret and rgb_ret:
                thermal_predicted = cv2.projectPoints(objp, thermal_rvecs, thermal_tvecs, K_thermal, D_thermal)[0].squeeze()
                rgb_predicted = cv2.projectPoints(objp, rgb_rvecs, rgb_tvecs, K, D)[0].squeeze()

                thermal_img3 = cv2.drawChessboardCorners(np.copy(invert_img_b),(10, 7), corners_t, thermal_ret)
                rgb_img3 = cv2.drawChessboardCorners(np.copy(basler_img),(10, 7), corners_b, rgb_ret)

                for j in range(len(thermal_predicted)):
                    cv2.circle(thermal_img3, tuple(thermal_predicted[j]), 3, (0,255,0), 2)
                    cv2.circle(rgb_img3, tuple(rgb_predicted[j]), 3, (0,255,0), 2)

                P_2 = np.hstack((cv2.Rodrigues(rgb_rvecs)[0], rgb_tvecs))           #from world to RGB cam
                P_1 = np.hstack((cv2.Rodrigues(thermal_rvecs)[0], thermal_tvecs))   #from world to Thermal cam

                ones = np.ones(len(objp))[:, np.newaxis]
                transformed_ = np.hstack((objp, ones))

                cloud_in_RGB_frame =     np.dot(P_2, transformed_.T).T
                cloud_in_Thermal_frame = np.dot(P_1, transformed_.T).T
                
                ax.scatter(*cloud_in_RGB_frame.T, marker='o', label='RGB-{}'.format(i)) 
                ax.scatter(*cloud_in_Thermal_frame.T, marker='v', label='Thermal-{}'.format(i)) #

                data = np.concatenate([cloud_in_RGB_frame,cloud_in_Thermal_frame], axis=0)
                axisEqual3D_init(ax=ax,data=data, cube = 2)
                # 
                
                fig.canvas.draw()
                fig.canvas.flush_events()

                if len(all_thermal_3d) == 0:
                    all_thermal_3d = cloud_in_Thermal_frame
                    all_rgb_3d = cloud_in_RGB_frame
                else:
                    all_thermal_3d = np.concatenate([all_thermal_3d,cloud_in_Thermal_frame], axis=0)
                    all_rgb_3d = np.concatenate([all_rgb_3d,cloud_in_RGB_frame], axis=0)

                #print('all_thermal_3d:{}, all_rgb_3d:{}'.format(np.shape(all_thermal_3d), np.shape(all_rgb_3d)))
                cv2.imshow("thermal chessboard", thermal_img3)
                cv2.imshow("rgb chessboard", cv2.resize(rgb_img3, None, fx=.5,fy=.5))
                cv2.waitKey(1)

    i = 0 
    for thermal_img, basler_img in zip(images_16bit, basler):
        i += 1
        if True:
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
            corners_t = cv2.cornerSubPix(invert_img_b, corners_t, (11,11), (-1,-1),
                                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            
            invert_img_b_Draw = cv2.cvtColor(invert_img_b.copy(),cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(invert_img_b_Draw, (10, 7), corners_t, ret_t)
            invert_img_b = cv2.cvtColor(invert_img_b.copy(),cv2.COLOR_GRAY2RGB)

        
        basler_img = cv2.cvtColor(basler_img.copy(),cv2.COLOR_RGB2GRAY)

        ret_b, corners_b = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_b:
            corners_b = cv2.cornerSubPix(basler_img, corners_b, (11,11), (-1,-1),
                                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            basler_img = cv2.cvtColor(basler_img.copy(),cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(basler_img, (10, 7), corners_b, ret_b)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b_Draw)

        if ret_t and ret_b:
            objpoints.append(objp)
            imgpoints_l.append(corners_t)
            imgpoints_r.append(corners_b)

            # Find the rotation and translation vectors.
            # corners_t = cv2.undistortPoints(corners_t, K_thermal, D_thermal)
            # corners_b = cv2.undistortPoints(corners_b, K, D)

            thermal_ret, thermal_rvecs, thermal_tvecs = cv2.solvePnP(objp, corners_t, K_thermal, D_thermal)
            rgb_ret, rgb_rvecs, rgb_tvecs = cv2.solvePnP(objp, corners_b, K, D)
            if thermal_ret and rgb_ret:
                thermal_predicted = cv2.projectPoints(objp, thermal_rvecs, thermal_tvecs, K_thermal, D_thermal)[0].squeeze()
                rgb_predicted = cv2.projectPoints(objp, rgb_rvecs, rgb_tvecs, K, D)[0].squeeze()

                thermal_img3 = cv2.drawChessboardCorners(np.copy(invert_img_b),(10, 7), corners_t, thermal_ret)
                rgb_img3 = cv2.drawChessboardCorners(np.copy(basler_img),(10, 7), corners_b, rgb_ret)

                for j in range(len(thermal_predicted)):
                    cv2.circle(thermal_img3, tuple(thermal_predicted[j]), 3, (0,255,0), 2)
                    cv2.circle(rgb_img3, tuple(rgb_predicted[j]), 3, (0,255,0), 2)

                P_2 = np.hstack((cv2.Rodrigues(rgb_rvecs)[0], rgb_tvecs))           #from world to RGB cam
                P_1 = np.hstack((cv2.Rodrigues(thermal_rvecs)[0], thermal_tvecs))   #from world to Thermal cam

                ones = np.ones(len(objp))[:, np.newaxis]
                transformed_ = np.hstack((objp, ones))

                cloud_in_RGB_frame =     np.dot(P_2, transformed_.T).T
                cloud_in_Thermal_frame = np.dot(P_1, transformed_.T).T
                
                ax.scatter(*cloud_in_RGB_frame.T, marker='o', label='RGB-{}'.format(i)) 
                ax.scatter(*cloud_in_Thermal_frame.T, marker='v', label='Thermal-{}'.format(i)) #

                data = np.concatenate([cloud_in_RGB_frame,cloud_in_Thermal_frame], axis=0)
                axisEqual3D_init(ax=ax,data=data, cube = 2)
                # 
                
                fig.canvas.draw()
                fig.canvas.flush_events()

                if len(all_thermal_3d) == 0:
                    all_thermal_3d = cloud_in_Thermal_frame
                    all_rgb_3d = cloud_in_RGB_frame
                else:
                    all_thermal_3d = np.concatenate([all_thermal_3d,cloud_in_Thermal_frame], axis=0)
                    all_rgb_3d = np.concatenate([all_rgb_3d,cloud_in_RGB_frame], axis=0)

                #print('all_thermal_3d:{}, all_rgb_3d:{}'.format(np.shape(all_thermal_3d), np.shape(all_rgb_3d)))
                cv2.imshow("thermal chessboard", thermal_img3)
                cv2.imshow("rgb chessboard", cv2.resize(rgb_img3, None, fx=.5,fy=.5))
                cv2.waitKey(1)
                

        key = cv2.waitKey(20)
        if key == 27:  # ESC
            break
        
    cv2.waitKey(0)
    plt.show()
    cv2.destroyAllWindows()

    flags = cv2.CALIB_FIX_INTRINSIC
    print("\n Start extrinsic calibration with {} images".format(len(imgpoints_r)))
    rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_l, imgpoints_r, K_thermal, D_thermal, K, D, imageSize=None, criteria=term_criteria, flags=flags)

    print('Stereo calibraion Thermal done')
    print('rms_stereo:{}'.format(rms_stereo))
    print('Rotation R')
    print(R)
    print('R radians:\n',cv2.Rodrigues(R)[0].T,' \ndegrees: ',[ np.rad2deg(i) for i in cv2.Rodrigues(R)[0].T][0])

    print('Translation T')
    print(T)

    # return 
    np.savez(
            save_result,
            objpoints = np.array(objpoints, dtype=object),
            imgpoints_l = np.array(imgpoints_l, dtype=object),
            imgpoints_r = np.array(imgpoints_r, dtype=object),
            K_thermal = K_thermal,
            D_thermal = D_thermal,
            K_baseler=K,
            D_baseler=D,
            R = R,
            T = T,
            E = E,
            F = F
    )
    print("Saved ",save_result)
    
    print('TOTAL 3D POINTS  - > all_thermal_3d:{}, all_rgb_3d:{}'.format(np.shape(all_thermal_3d), np.shape(all_rgb_3d)))
    T,R = align_svd(pc1 = all_rgb_3d.T, pc2 = all_thermal_3d.T)
    
    print('R radians:\n',cv2.Rodrigues(R)[0].T,' \ndegrees: ',[ np.rad2deg(i) for i in cv2.Rodrigues(R)[0].T][0])
    print('T ',T.T)
    t = T
   
    
    np.savez(
            save_result2,
            objpoints = np.array(objpoints, dtype=object),
            imgpoints_l = np.array(imgpoints_l, dtype=object),
            imgpoints_r = np.array(imgpoints_r, dtype=object),
            K_thermal = K_thermal,
            D_thermal = D_thermal,
            K_baseler=K,
            D_baseler=D,
            R = R,
            T = T,
            E = E,
            F = F,
            all_rgb_3d = np.array(all_rgb_3d, dtype=object),
            all_thermal_3d = np.array(all_thermal_3d, dtype=object),
    )
    print("Saved ",save_result2)

    verify_params = True
    if verify_params:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #baseler = cv2.imread('chess/big/baseler_{}.png'.format(cam))
        #lidar = np.array(np.load('chess/big/cloud_{}.npy'.format(cam))[:,:3], dtype=np.float32)
        baseler = basler_img
        lidar = all_rgb_3d

        print("lidar:", np.shape(lidar))

        cloud_in_cam_frame = lidar# projetPoints(_3D=lidar,R=rvec_cam,t=tvec_cam, K=cameraMatrix_cam,D=dist_coeff_cam, img=np.copy(baseler), title="verify LiDAR overlay on RGB")

        thermal = invert_img_b

        #---------------------------------------------------------------------------------------------------------------------------------------------
        lidar_in_rgb_frame_to_thermal = (cloud_in_cam_frame - T.squeeze()) @ R  #from RGB to Thermal--------------------------------------------------

        px_lidar2therm = cv2.projectPoints(lidar_in_rgb_frame_to_thermal, np.identity(3), np.array([0., 0., 0.]), K_thermal, D_thermal)[0].squeeze().astype(int)
        inrange_ = np.where((px_lidar2therm[:, 0] > 0) &
                                (px_lidar2therm[:, 1] > 0) &
                                (px_lidar2therm[:, 0] < thermal.shape[1] - 1) &
                                (px_lidar2therm[:, 1] < thermal.shape[0] - 1))
        
        px_lidar2therm = px_lidar2therm[inrange_[0]].round().astype('int')
        _3D = lidar_in_rgb_frame_to_thermal[inrange_[0]]

        print("cloud_in_cam_frame:{}, lidar_in_rgb_frame_to_thermal:{}".format(np.shape(cloud_in_cam_frame), np.shape(lidar_in_rgb_frame_to_thermal)))
        ax.scatter(*cloud_in_cam_frame.T, alpha=.5, s=.4, label='cloud_in_cam_frame')
        ax.scatter(*lidar_in_rgb_frame_to_thermal.T, alpha=1, s=.5, label='lidar_in_rgb_frame_to_thermal')
        plt.legend()

        # fig.canvas.draw()
        # fig.canvas.flush_events()
        plt.show()
        
        img = np.copy(thermal)
        for j in range(len(px_lidar2therm)):
            cv2.circle(img, tuple(px_lidar2therm[j]), 2, (0,255,0), 1)
            # cv2.circle(img, tuple(px_lidar2therm[j]), 1, cols[j], -1)

        cv2.imshow("verify LiDAR2Therm", img)
        cv2.waitKey(0)
        

        if len(_3D) > 0:
            distance = np.linalg.norm(_3D, axis=1)
            MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
            colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(1), 1.0)) for c in colours])
            cols = 255 * colours

            skip = 2 # 3
            ax.scatter(*cloud_in_cam_frame[::skip,:].T, alpha=.5, s=.4, label='cloud_in_cam_frame')
            ax.scatter(*_3D[::skip,:].T, alpha=1, s=.5, label='lidar_in_rgb_frame_to_thermal')
            plt.legend()

            fig.canvas.draw()
            fig.canvas.flush_events()

            img = np.copy(thermal)
            for j in range(len(px_lidar2therm)):
                #cv2.circle(img, tuple(px_lidar2therm[j]), 2, (0,255,0), 1)
                cv2.circle(img, tuple(px_lidar2therm[j]), 1, cols[j], -1)

            cv2.imshow("verify LiDAR2Therm", img)
            cv2.waitKey(1)
        else:
            print("NO POINTS TO PROJECT ON THE IMAGE================")

        #--------------------------------------------------------------------------------------------------------------------------------------------
        #from Thermal back to RGB
        lidar_in_rgb_frame_to_thermal_back = (_3D @ R.T)+ t.squeeze()  #from Thermal to RGB --------------------------------------------------

        px_lidar2therm_back = cv2.projectPoints(lidar_in_rgb_frame_to_thermal_back, np.identity(3), np.array([0., 0., 0.]), K, D)[0].squeeze().astype(int)
        inrange_ = np.where((px_lidar2therm_back[:, 0] > 0) & (px_lidar2therm_back[:, 1] > 0) &(px_lidar2therm_back[:, 0] < baseler.shape[1] - 1) & (px_lidar2therm_back[:, 1] < baseler.shape[0] - 1))
        px_lidar2therm_back = px_lidar2therm_back[inrange_[0]].round().astype('int')
        _3D = lidar_in_rgb_frame_to_thermal_back[inrange_[0]]

        if len(_3D) > 0:
            distance = np.linalg.norm(_3D, axis=1)
            MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
            colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(1), 1.0)) for c in colours])
            cols = 255 * colours

            img = np.copy(baseler)
            for j in range(len(px_lidar2therm_back)):
                #cv2.circle(img, tuple(px_lidar2therm_back[j]), 2, (0,255,0), 1)
                cv2.circle(img, tuple(px_lidar2therm_back[j]), 2, cols[j], -1)

            cv2.imshow("verify LiDAR2Therm back projection", cv2.resize(img, None, fx=.5,fy=.5))
            cv2.waitKey(1)
        else:
            print("NO POINTS TO PROJECT ON THE IMAGE================")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


stereo_calibrate()

def test_extrinsics():
    loaded = np.load(save_result, allow_pickle=True)
    K_thermal = loaded["K_thermal"]
    D_thermal = loaded["D_thermal"]   
    R = loaded["R"]
    T = loaded["T"]

    #LEFT FRAME IS THE THERMAL CAMERA ALWAYS HERE 
    for thermal_img, basler_img in zip(images_16bit, basler):

        if True:
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
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

# test_extrinsics()