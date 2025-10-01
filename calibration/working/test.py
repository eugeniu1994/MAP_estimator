import numpy as np
import cv2
import open3d as o3d
import math

#  Set print options to avoid scientific notation
np.set_printoptions(suppress=True)

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

def readCalibration_tests(camera='basler'):

    if camera == 'basler': #good
        K = np.array([[1363.18778612,    0.,          978.05715426],
                      [   0.,         1362.77700097,  607.65710195],
                      [   0.,            0.,            1.        ]])
        D = np.array( [[-0.12242753 , 0.05002637 , 0.00010842 , 0.00189244 , 0.03743694]])

        # extrinsic params - LiDAR2CAM
        # rvec = cv2.Rodrigues(np.array([1.67218474,0.01947635,0.00455869]))[0]
        # tvec = np.array([[0.00436079],[-0.0896923],[-0.441341]])

        # ###################################################
        rvec = np.array([[0.999883,    0.00902631,  0.0123187],
                                  [0.0131557 , -0.0994464 , -0.994956 ],
                                  [-0.00775573, 0.995002  , -0.0995536]])
        #rvec = np.matmul(lidar_cam_rot, eulerAnglesToRotationMatrix(np.array([-0.005,0.0,-0.0013]))) # NOTE: A fix by Jyri
        tvec =  np.array([[0.00286508], [-0.0854374], [-0.427633]])
    #good
    elif camera == 'thermal_center' or camera=='center': #camera == 'thermal_left':
        K = np.array([[1495.84601928,    0.,          355.3401074 ],
                      [   0.,         1495.58217734,  257.10045874],
                      [   0.,            0.,            1.        ]])

        D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])

        rvec = cv2.Rodrigues(np.array([-0.00351747 , 0.00264308 , 0.0092578]))[0]
        tvec = np.array([[-0.00241581 , 0.06494797 , 0.05620015]])


        format = ""
        extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_center_thermal_to_baseler_{}.npz".format(format)
        format = "weighted"
        extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_center_thermal_to_baseler_{}.npz".format(format)

        loaded = np.load(extrinsic_center, allow_pickle=True) 

        K = loaded["K_thermal"]
        D = loaded["D_thermal"]   
        rvec = loaded["R"]
        tvec = loaded["T"]

    elif camera == 'thermal_right' or camera == 'right': #camera == 'thermal_center':
        # intrinsic matrix
        K = np.array([[1499.29684013  ,  0.    ,      338.76978523],
                      [   0.     ,    1500.27024765,  239.59764372],
                      [   0.     ,       0.    ,        1.        ]])
        # distortion
        #D = np.array([ -0.13541172 , -4.69373918  , 0.00286321   ,0.00275366 ,142.56355612])
        D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])

        rvec = cv2.Rodrigues(np.array([-0.0036622 ,  0.4046476  , 0.00358055]))[0]
        tvec = np.array([[0.05503872, 0.06602019, 0.03417408]])

        # format = ""
        format = "weighted"
        extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_right_thermal_to_baseler_{}.npz".format(format)

        # # extrinsic_center = "/media/eugeniu/T7/calibration/a_new_results/extrinsic_right_thermal_to_baseler_{}2.npz".format(format)

        loaded = np.load(extrinsic_center, allow_pickle=True) 

        K = loaded["K_thermal"]
        D = loaded["D_thermal"]   
        rvec = loaded["R"]
        tvec = loaded["T"]

    #good
    elif camera == 'thermal_left' or camera == 'left': #camera == 'thermal_right':
        # intrinsic matrix
        K = np.array([[1496.0431082,    0.,         349.23737282],
                      [   0.,        1497.37541302, 238.32624878],
                      [   0.,           0.,           1.        ]])
        # distortion
        D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])

        rvec = cv2.Rodrigues(np.array([0.0006695,  -0.41386521,  0.01294857]))[0]
        tvec = np.array([[-0.0569287,   0.05887872,  0.03450623]])

    else:
        print("NO CALIBRATION FOR:", camera)

    return rvec, tvec, K, D

def readCalibration(camera='basler'):
    def eulerAnglesToRotationMatrix(theta):
        theta = theta*np.pi/180.0
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    
    if camera == 'basler':
        K = np.array([[1363.18778612,    0.,          978.05715426],
                      [   0.,         1362.77700097,  607.65710195],
                      [   0.,            0.,            1.        ]])
        D = np.array( [[-0.12242753 , 0.05002637 , 0.00010842 , 0.00189244 , 0.03743694]])

        rvec = np.array([[0.999883,    0.00902631,  0.0123187],
                                  [0.0131557 , -0.0994464 , -0.994956 ],
                                  [-0.00775573, 0.995002  , -0.0995536]])
        
        #this has not beed observed in the calibration data, Juri found them heuristically
        # rvec = np.matmul(rvec, eulerAnglesToRotationMatrix(np.array([-0.005,0.0,-0.0013]))) # NOTE: A fix by Jyri
        tvec =  np.array([[0.00286508], [-0.0854374], [-0.427633]])

    elif camera == 'thermal_center' or camera=='center': #camera == 'thermal_left':
        K = np.array([[1524.24454171 ,   0.     ,     336.01298952],
                     [   0.     ,    1524.24454171 , 243.0487864 ],
                     [   0.     ,       0.        ,    1.        ]])

        D = np.array([-0.08193608, -1.42747637, -0.00023972,  0.00110947,  0.        ])

        rvec = np.array([[ 0.99990387, -0.01028711 ,-0.00929656],
                        [ 0.01023551,  0.99993204, -0.00558048],
                        [ 0.00935333 , 0.00548479,  0.99994121]])
        tvec = np.array([[-0.00422653], [0.06651897 ], [-0.01650378]])

    elif camera == 'thermal_right' or camera == 'right': #camera == 'thermal_center':
        # intrinsic matrix
        K = np.array([[1546.84773827 ,   0.     ,     332.52602693],
                    [   0.    ,     1546.84773827,  205.23907589],
                    [   0.    ,        0.     ,       1.        ]])
        # distortion
        D = np.array( [[-0.07407248], [-1.43229151], [-0.00330972], [ 0.0028883 ],[ 3.        ]] )
        
        rvec = np.array([[ 0.92133068 , 0.00379052 , 0.38876138],
                        [ 0.00413093  ,0.99980058 ,-0.01953825],
                        [-0.38875791,  0.01960714 , 0.92113129]])

        tvec = np.array([[ 0.01943256], [0.06894031], [-0.05259934]])

    elif camera == 'thermal_left' or camera == 'left': #camera == 'thermal_right':
        # intrinsic matrix
        K = np.array([[1496.0431082,    0.,         349.23737282],
                      [   0.,        1497.37541302, 238.32624878],
                      [   0.,           0.,           1.        ]])
        # distortion
        D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])
        
        rvec = cv2.Rodrigues(np.array([0.00192539 ,-0.41169108,  0.00974791]))[0]
        tvec = np.array([[-0.06363304] , [0.06323869] , [0.03463458]])
    else:
        print("NO CALIBRATION FOR:", camera)

    return rvec, tvec, K, D

def get_z(T_cam_world, T_world_pc, K):
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
    xyz_hom = np.hstack((T_world_pc, np.ones((T_world_pc.shape[0], 1))))
    xy_hom = np.dot(proj_mat, xyz_hom.T).T

    z = xy_hom[:, -1]
    #z = xy_hom[:, 0]
    #z = xy_hom[:, 1]
    z = np.asarray(z).squeeze()
    return z

def overlay():
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

    square = 0.1  # m (the size of each chessboard square is 10cm)
    objp = np.zeros((10 * 7, 3), np.float32) #chessboard is 7x10
    objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)


    def projetPoints(_3D,R,t, K,D, img, title):
        P = np.hstack((R, t))

        T = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        Z = get_z(T, _3D, K)
        _3D = _3D[Z > 3]    #this is done to use only points further from 4m distance in front of the car
        # _3D = _3D[_3D[:,1] > 4]

        distance = np.linalg.norm(_3D, axis=1)

        _3D = _3D[distance < 25] #this is just a test

        # _3D = _3D[distance < 15] #just for color - remove for bigger cloud
        # _3D = _3D[distance < 11] #just for color - remove for bigger cloud

        ones = np.ones(len(_3D))[:, np.newaxis]
        transformed_ = np.hstack((_3D, ones))
        cloud_in_camera_frame = np.dot(P, transformed_.T).T
        print("cloud_in_camera_frame ", np.shape(cloud_in_camera_frame))
        # project onto a plane using normalized image coordinates
        # x = cloud_in_camera_frame[:, 0] / cloud_in_camera_frame[:,2]
        # y = cloud_in_camera_frame[:, 1] / cloud_in_camera_frame[:,2]
        # # find radial distance (square is good enough)
        # r2 = x*x + y*y
        # # reject points behind camera and points with too large radial distance
        # # the radial distortion model with the current parameters has local maxima around 1.8
        # # Thus, the model is valid between r2 values between 0 and 1.8^2 = 3.24
        # valid = np.logical_and(cloud_in_camera_frame[:,2] > 4, r2 < 999999)
        
        # cloud_in_camera_frame = cloud_in_camera_frame[valid,:]   


        #px = cv2.projectPoints(_3D, R, t, K, D)[0].squeeze()
        px = cv2.projectPoints(cloud_in_camera_frame, np.identity(3), np.array([0., 0., 0.]), K, D)[0].squeeze().astype(int)

        inrange_ = np.where((px[:, 0] > 0) & (px[:, 1] > 0) &(px[:, 0] < img.shape[1] - 1) & (px[:, 1] < img.shape[0] - 1))
        points2D = px[inrange_[0]].round().astype('int')
        _3D = _3D[inrange_[0]]
        distance = np.linalg.norm(_3D, axis=1)
        MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
        #MIN_DISTANCE, MAX_DISTANCE = 5, 50
        colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(1), 1.0)) for c in colours])
        cols = 255 * colours
        #points2D = points2D[distance < 13]
        #points2D = points2D[np.argsort(distance)]
        for j in range(len(points2D)):
            cv2.circle(img, tuple(points2D[j]), 2, cols[j], -1)
            #cv2.circle(img, tuple(points2D[j]), 2, (0,255,0), -1)

        cv2.imshow(title, cv2.resize(img, None, fx=.5, fy=.5))
        cv2.waitKey(1)

        return cloud_in_camera_frame[inrange_[0]]

    cams = ['left','center','right'] #
    short_range_cloud = True

    # cams = ['right'] # good
    # cams = ['center'] #good
    # cams = ['left'] # #bad 

    rvec_cam, tvec_cam, cameraMatrix_cam, dist_coeff_cam = readCalibration(camera='basler')

    for i,cam in enumerate(cams):
        R, t, cameraMatrix_thermal, dist_coeff_thermal = readCalibration(camera=cam)

        if short_range_cloud:
            baseler = cv2.imread('data/close/baseler_{}.png'.format(cam))
            lidar = np.array(np.load('data/close/cloud_{}.npy'.format(cam))[:,:3], dtype=np.float32)
            thermal = cv2.imread('data/close/{}.png'.format(cam))
        else:
            baseler = cv2.imread('data/far/image_test.png')
            pcd = o3d.io.read_point_cloud("data/far/cloud_test.pcd")
            lidar = np.asarray(pcd.points)
            thermal = cv2.imread('data/far/thermal_{}.png'.format(cam))


        cloud_in_cam_frame = projetPoints(_3D=lidar,R=rvec_cam,t=tvec_cam, K=cameraMatrix_cam,D=dist_coeff_cam, img=np.copy(baseler), title="LiDAR overlay on RGB {}".format(i))
        #Lidar cloud in Baseler camera frame - project to thermal-----------------------------
        lidar_in_rgb_frame_to_thermal = (cloud_in_cam_frame - t.squeeze()) @ R  #from RGB to Thermal--------------------------------------------------

        px_lidar2therm = cv2.projectPoints(lidar_in_rgb_frame_to_thermal, np.identity(3), np.array([0., 0., 0.]), cameraMatrix_thermal, dist_coeff_thermal)[0].squeeze().astype(int)
        inrange_ = np.where((px_lidar2therm[:, 0] > 0) &
                            (px_lidar2therm[:, 1] > 0) &
                            (px_lidar2therm[:, 0] < thermal.shape[1] - 1) &
                            (px_lidar2therm[:, 1] < thermal.shape[0] - 1))
        px_lidar2therm = px_lidar2therm[inrange_[0]].round().astype('int')
        _3D = lidar_in_rgb_frame_to_thermal[inrange_[0]]

        if len(_3D) > 0:
            distance = np.linalg.norm(_3D, axis=1)
            MIN_DISTANCE, MAX_DISTANCE = np.min(distance), np.max(distance)
            colours = (distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
            colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(1), 1.0)) for c in colours])
            cols = 255 * colours
            img = np.copy(thermal)
            for j in range(len(px_lidar2therm)):
                #cv2.circle(img, tuple(px_lidar2therm[j]), 2, (0,255,0), 1)
                cv2.circle(img, tuple(px_lidar2therm[j]), 1, cols[j], -1)

            cv2.imshow("LiDAR2Therm-{}".format(cam), img)
            cv2.waitKey(1)
        else:
            print("NO POINTS TO PROJECT ON THE IMAGE================")

        #--------------------------------------------------------------------------------------------------------------------------------------------
        #from Thermal back to RGB
        lidar_in_rgb_frame_to_thermal_back = (_3D @ R.T)+ t.squeeze()  #from Thermal to RGB --------------------------------------------------

        px_lidar2therm_back = cv2.projectPoints(lidar_in_rgb_frame_to_thermal_back, np.identity(3), np.array([0., 0., 0.]), cameraMatrix_cam, dist_coeff_cam)[0].squeeze().astype(int)
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

            cv2.imshow("LiDAR2Therm back projection - {}".format(cam), cv2.resize(img, None, fx=.5,fy=.5))
            cv2.waitKey(1)
        else:
            print("NO POINTS TO PROJECT ON THE IMAGE================")

        #next press enter to check for other images

        #reporjections between the cameras using the chessboard 
        basler_img = 'data/extrinsic/baseler_{}.npy'.format(cam)
        thermal_img = 'data/extrinsic/{}.npy'.format(cam)

        thermal, baseler = np.load(thermal_img), np.load(basler_img)
        baseler = cv2.cvtColor(baseler.copy(),cv2.COLOR_RGB2GRAY)
        valid_pixels = thermal[thermal > 0]
        min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
        max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
        img_clipped = np.clip(thermal, min_val, max_val)
        img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
        #thermal = np.array(255.0 - img8, dtype='uint8')
        thermal = img8


        ret_t, corners_t = cv2.findChessboardCorners(thermal, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret_t:
            corners_t = cv2.cornerSubPix(thermal, corners_t, (11,11), (-1,-1),term_criteria)
            thermal = cv2.cvtColor(thermal.copy(),cv2.COLOR_GRAY2RGB)
            # cv2.drawChessboardCorners(thermal, (10, 7), corners_t, ret_t)

        
        ret_b, cornersR = cv2.findChessboardCorners(baseler, (10, 7),flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_b:
            cornersR = cv2.cornerSubPix(baseler, cornersR, (11,11), (-1,-1),term_criteria)
            baseler = cv2.cvtColor(baseler.copy(),cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(baseler, (10, 7), cornersR, ret_b)

        if ret_t and ret_b:
            #Basler 2 thermal projection
            success, rvecR, tvecR = cv2.solvePnP(objp, cornersR, cameraMatrix_cam, dist_coeff_cam)

            # Convert rvecR to rotation matrix
            R_chess_R, _ = cv2.Rodrigues(rvecR)

            # Homogeneous transform of chessboard in right frame
            T_chess_R = np.eye(4)
            T_chess_R[:3,:3] = R_chess_R
            T_chess_R[:3, 3] = tvecR.ravel()

            # Stereo extrinsics: transform from right -> left
            T_LR = np.eye(4)
            R_inv, t_inv = invert_transform(R = R, t = t)
            T_LR[:3,:3] = R_inv  # rotation from right to left
            T_LR[:3, 3]  = t_inv.ravel()  # translation from right to left

            # Transform chessboard pose to left camera frame
            T_chess_L = T_LR @ T_chess_R
            R_chess_L = T_chess_L[:3,:3]
            t_chess_L = T_chess_L[:3, 3]

            # Project 3D object points into right image
            proj_points, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_chess_L)[0], 
                                   t_chess_L, cameraMatrix_thermal, dist_coeff_thermal)
                            
            error = cv2.norm(corners_t, proj_points, cv2.NORM_L2) / len(proj_points)
            print("Reprojection error of Basler2Thermal (predicted and detected corners):", error)
            # cv2.drawChessboardCorners(thermal, (10, 7), proj_points, ret_b)
            for j in range(len(proj_points)):
                # cv2.circle(thermal, tuple(proj_points[j]), 2, cols[j], -1)
                x, y = proj_points[j].ravel()  # flattens [[x, y]] - [x, y]
                pt = (int(x), int(y))
                cv2.circle(thermal, pt, 3, (0, 255, 0), 2)

            #--------------------------------------------------------------------
            success, rvecL, tvecL = cv2.solvePnP(objp, corners_t, cameraMatrix_thermal, dist_coeff_thermal)

            # Convert rvecL to rotation matrix
            R_chess_L, _ = cv2.Rodrigues(rvecL)

            # Homogeneous transform of chessboard in left frame
            T_chess_L = np.eye(4)
            T_chess_L[:3,:3] = R_chess_L
            T_chess_L[:3, 3] = tvecL.ravel()

            # Stereo extrinsics: transform from left -> right
            T_RL = np.eye(4)
            T_RL[:3,:3] = R  # rotation from left to right
            T_RL[:3, 3]  = t.ravel()  # translation from left to right

            # Transform chessboard pose to right camera frame
            T_chess_R = T_RL @ T_chess_L
            R_chess_R = T_chess_R[:3,:3]
            t_chess_R = T_chess_R[:3, 3]

            # Project 3D object points into right image
            proj_points, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_chess_R)[0], t_chess_R, cameraMatrix_cam, dist_coeff_cam)
                
                    
            error = cv2.norm(cornersR, proj_points, cv2.NORM_L2) / len(proj_points)
            print("Reprojection error of Thermal2Basler (predicted and detected corners):", error)
            #cv2.drawChessboardCorners(baseler, (10, 7), proj_points, ret_b)
            for j in range(len(proj_points)):
                # cv2.circle(thermal, tuple(proj_points[j]), 2, cols[j], -1)
                x, y = proj_points[j].ravel()  # flattens [[x, y]] - [x, y]
                pt = (int(x), int(y))
                cv2.circle(baseler, pt, 3, (255, 0, 0), 2)


            cv2.imshow("Basler {}".format(cam), cv2.resize(baseler, None, fx=.5,fy=.5))
            cv2.imshow("Thermal-{}".format(cam), thermal)



        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    #project point cloud on the rgb image, and from there to each thermal camera and back
    #projections_from_basler2thermal_and_back

    overlay()
