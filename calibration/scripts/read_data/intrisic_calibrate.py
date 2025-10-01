import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


#left thermal camera
save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_left" 
save_dir2 = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/thermal"
save_path = "/media/eugeniu/T7/calibration/a_new_results/intrinsic_left_thermal_{}.npz"

# #center
# save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_center" 
# save_dir2 = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/center/thermal"
# save_path = "/media/eugeniu/T7/calibration/a_new_results/intrinsic_center_thermal_{}.npz"

# #right
# save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_right" 
# save_dir2 = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/right/thermal"
# save_path = "/media/eugeniu/T7/calibration/a_new_results/intrinsic_right_thermal_{}.npz"





# save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_center" 
# save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_center_thermal_{}.npz"
# save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_center_thermal_filt_{}.npz"




#save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_left_thermal_filt_{}.npz"



# save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_right" 
# save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_right_thermal_{}.npz"
# save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_right_thermal_filt_{}.npz"

# save_path = "/media/eugeniu/T7/calibration/saved_data_raw/2intrinsic_left_thermal_{}.npz"
# save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/2intrinsic_left_thermal_filt_{}.npz"

# save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_center" 
# save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_center_thermal_{}.npz"
# save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_center_thermal_filt_{}.npz"

is_thermal = True 

# is_thermal = False 
# save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/baseler" 
# save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_baseler_{}.npz"
# save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_baseler_filt_{}.npz"


pattern_size = (10, 7)  
square_size = 0.1 #m  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)


clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def convert_16bit_to_8bit_auto(img16, clehe = False, lower_pct=2, upper_pct=98):
    nonzero = img16[img16 > 0]
    min_val = np.percentile(nonzero, lower_pct)
    max_val = np.percentile(nonzero, upper_pct)
    img_clipped = np.clip(img16, min_val, max_val)
    img8 = ((img_clipped - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)

    if clehe:
        return clahe_.apply(img8)

    return img8

def convert_16bit_to_8bit_manual(img16,  clehe = False, min_val=22200, max_val=23200):
    img_clipped = np.clip(img16, min_val, max_val)
    img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

    if clehe:
        return clahe_.apply(img8)
    
    return img8

def method_auto(img):
    return convert_16bit_to_8bit_auto(img, clehe=False)

def method_auto_clehe(img):
    return convert_16bit_to_8bit_auto(img, clehe=True)

def method_manual(img):
    return convert_16bit_to_8bit_manual(img, clehe=False)

def method_manual_clehe(img):
    return convert_16bit_to_8bit_manual(img, clehe=True)


def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    return [np.load(os.path.join(folder, f)) for f in files]

def generate_object_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size
    return objp

images_16bit = load_images(save_dir)
images_16bit_flip = np.full(len(images_16bit), True, dtype=bool)


images_16bit2 = load_images(save_dir2)
images_16bit2_flip = np.full(len(images_16bit2), False, dtype=bool)

print("images_161:{}, images_16bit2:{}".format(np.shape(images_16bit), np.shape(images_16bit2)))
print("images_16bit_flip:{}, images_16bit2_flip:{}".format(np.shape(images_16bit_flip), np.shape(images_16bit2_flip)))


# images_16bit = images_16bit + images_16bit2
# images_16bit_flip = np.concatenate([images_16bit_flip, images_16bit2_flip])
# print("Combined images_16bit_:", np.shape(images_16bit),", images_16bit_flip:",np.shape(images_16bit_flip))

objp = generate_object_points(pattern_size, square_size)

Distorsion_models = {    
     'ST': ['Standard', 0, 'Standard'],
                            #  'RAT': ['Rational', cv2.CALIB_RATIONAL_MODEL, 'CALIB_RATIONAL_MODEL'],
                            #  'THP': ['Thin Prism', cv2.CALIB_THIN_PRISM_MODEL, 'CALIB_THIN_PRISM_MODEL'],
                            #  'TIL': ['Tilded', cv2.CALIB_TILTED_MODEL, 'CALIB_TILTED_MODEL'],  # }
                            #  'RAT+THP': ['Rational+Thin Prism', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL,
                            #              'CALIB_RATIONAL_MODEL + CALIB_THIN_PRISM_MODEL'],
                            #  'THP+TIL': ['Thin Prism+Tilded', cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                            #              'CALIB_THIN_PRISM_MODEL + CALIB_TILTED_MODEL'],
                            #  'RAT+TIL': ['Rational+Tilded', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_TILTED_MODEL,
                            #              'CALIB_RATIONAL_MODEL + CALIB_TILTED_MODEL'],
                            #  'CMP': ['Complete',
                            #          cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                            #          'Complete']
                                     }

def calibrate_scenario(label, convert16bit_2_8bit, d_model = "", flags=None):
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane

    print('Start chessboard detection...')
    id = 0
    for idx, img16 in enumerate(images_16bit):
        # id += 1
        # if id % 10 !=0:
        #     continue

        # img8 = convert16bit_2_8bit(img16)
        # invert_img = np.array(256 - img8, dtype='uint8')

        if is_thermal:
            valid_pixels = img16[img16 > 0]
            min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
            max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
            img_clipped = np.clip(img16, min_val, max_val)
            img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 

            if images_16bit_flip[idx]:
                invert_img = np.array(255.0 - img8, dtype='uint8')
            else:
                invert_img = img8


            gray = invert_img
            #invert_img = cv2.cvtColor(invert_img, cv2.COLOR_GRAY2BGR)
            
        else:
            gray = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)
            #invert_img = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)

        #backtorgb = invert_img# cv2.cvtColor(invert_img.copy(),cv2.COLOR_GRAY2RGB)
        backtorgb = cv2.cvtColor(invert_img.copy(),cv2.COLOR_GRAY2RGB)

        #ret, corners = cv2.findChessboardCorners(invert_img, pattern_size, None)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)


        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(backtorgb, pattern_size, corners, ret)
            object_points.append(objp)
            image_points.append(corners)
            #print(f"Chessboard detected in image {idx}")
        else:
            print(f"Chessboard NOT detected in image {idx}")

        cv2.imshow("Detection with {}".format(label), backtorgb)
        cv2.waitKey(10) 
    
    

    cv2.destroyAllWindows()
    
    if len(object_points) < 3:
        print("Not enough valid images for calibration. Need at least 3.")
        exit()

    print('Found {} good images'.format(len(object_points)))
    print('Start calibration... with {} images'.format(len(image_points)))

    if True:
        corners = np.squeeze(image_points).reshape(-1, 2)
        print("corners:", np.shape(corners))

        w,h = images_16bit[0].shape[::-1]  # (width, height)
        center = np.array([w/2, h/2])

        distances = np.linalg.norm(corners - center, axis=1)  # shape (N,)
        max_weight = 3 # 10  # maximum duplication
        min_dist = distances.min()
        max_dist = distances.max()

        weights = 1 + (distances - min_dist) / (max_dist - min_dist) * (max_weight - 1)
        weights = weights.astype(int)  # integer number of repetitions
        weighted_corners = []
        weighted_obj_pts = []
        backtorgb = cv2.cvtColor(invert_img.copy(),cv2.COLOR_GRAY2RGB)
        for (x, y), w in zip(corners, weights):
            # color: closer to center = blue, farther = red
            color_val = int(255 * (w / max_weight))
            cv2.circle(backtorgb, (x, y), radius=2, color=(0, 0, color_val), thickness=-1)
        
        cv2.imshow("Radius", backtorgb)
        cv2.waitKey(1)
        print("weights:", np.shape(weights))

        backtorgb = cv2.cvtColor(invert_img.copy(),cv2.COLOR_GRAY2RGB)
        for pt2d, pt3d, w in zip(image_points, object_points, weights):
            for w_i in range(w):
                weighted_corners.append(pt2d)
                weighted_obj_pts.append(pt3d)
                color_val = int(255 * (w_i / max_weight))


        #weighted_corners = np.array(weighted_corners, dtype=np.float32).reshape(-1,1,2)

        print("object_points:{}, image_points:{}".format(np.shape(object_points), np.shape(image_points)))
        print("weighted_obj_pts:{}, weighted_corners:{}".format(np.shape(weighted_obj_pts), np.shape(weighted_corners)))

        img_corners = corners.astype(int)  # (N, 2)
        backtorgb = cv2.cvtColor(invert_img.copy(),cv2.COLOR_GRAY2RGB)
        for (x, y) in img_corners:
            cv2.circle(backtorgb, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imshow("Original data", backtorgb)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

    print("Calibration started...")
    image_shape = images_16bit[0].shape[::-1]  # (width, height)
    image_shape = invert_img.shape[::-1]  # (width, height)


    flags = (#cv2.CALIB_USE_INTRINSIC_GUESS |
                cv2.CALIB_FIX_K3 #|  
            #  cv2.CALIB_FIX_ASPECT_RATIO #Keep fx/fy fixed (aspect ratio)
             )

    dist_init = np.zeros((5,1), dtype=np.float64) #standard distortion model
    # dist_init[4] = 3.0  # k3
    # Initialize camera matrix 
    #image_shape 512, 640 ->  cx = 320, cy = 256
    cx = image_shape[1] / 2
    cy = image_shape[0] / 2
    fx = fy = 1582.0  # or some rough guess

    K_init = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float64)

    
    ''' left camra
    RMSE: 0.7301924386059366
    Camera Matrix:
    [[1581.220732      0.          391.01372936]
    [   0.         1581.220732    222.99464925]
    [   0.            0.            1.        ]]
    Distortion Coefficients:
    [-0.10370359 -0.56686083 -0.00182508  0.00917698  3.        ]
    Init Mean reprojection error  0.07806709646877757
    std_error: 0.039017905913592714

    
    RMSE: 0.7179790186809287
    Camera Matrix:
    [[1580.61920492    0.          389.27636981]
    [   0.         1580.61920492  223.09421759]
    [   0.            0.            1.        ]]
    Distortion Coefficients:
    [-0.09914888 -0.61015511 -0.00190044  0.00908167  3.        ]
    Init Mean reprojection error  0.0764508397743339
    std_error: 0.03898034681943418
    '''

    '''center camera
    === Initial Calibration Results with auto method ===
    RMSE: 0.6101969055162344
    Camera Matrix:
    [[1525.36011998    0.          336.91288344]
    [   0.         1525.36011998  248.4613069 ]
    [   0.            0.            1.        ]]
    Distortion Coefficients:
    [-0.07792726 -1.5952295   0.00058022  0.0011142   3.        ]
    Init Mean reprojection error  0.06800428870386802
    std_error: 0.026354594031021344
    Calibration data saved to /media/eugeniu/T7/calibration/a_new_results/intrinsic_center_thermal_.npz
    Weighted Calibration started...

    === Weighted Calibration Results with auto method ===
    RMSE: 0.6214491250961768
    Camera Matrix:
    [[1524.18374937    0.          336.03079758]
    [   0.         1524.18374937  243.10647038]
    [   0.            0.            1.        ]]
    Distortion Coefficients:
    [-0.07554643 -1.6907075  -0.00023587  0.00111297  3.        ]
    Init Mean reprojection error  0.06872116124677256
    std_error: 0.02818737065294587
    '''


    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_shape, None, None, flags=flags  #K_init, dist_init
    )

    print("\n=== Initial Calibration Results with {} method ===".format(label))
    print("RMSE:",ret)
    print("Camera Matrix:\n", K)
    print("Distortion Coefficients:\n", dist.ravel())
    
    #reprojection error
    total_error = 0
    errors = []

    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
        total_error += error

    mean_error = total_error / len(object_points)
    print("Init Mean reprojection error ", mean_error) 

    errors = np.array(errors)
    std_error = np.std(errors)
    print("std_error:",std_error)
    format = ""
    np.savez(
            save_path.format(format),
            ret = ret,
            K=K,
            dist=dist,
            rvecs=np.array(rvecs, dtype=object),   # rvecs/tvecs are lists of arrays, so store as object
            tvecs=np.array(tvecs, dtype=object),
            object_points=np.array(object_points, dtype=object),
            image_points=np.array(image_points, dtype=object),
            errors=errors
        )
    print(f"Calibration data saved to {save_path.format(format)}")
    
    # exit()
    print("Weighted Calibration started...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        weighted_obj_pts, weighted_corners, image_shape, None, None, flags=flags #K_init, dist_init,
    )

    print("\n=== Weighted Calibration Results with {} method ===".format(label))
    print("RMSE:",ret)
    print("Camera Matrix:\n", K)
    print("Distortion Coefficients:\n", dist.ravel())
    
    #reprojection error
    total_error = 0
    errors = []

    for i in range(len(weighted_obj_pts)):
        imgpoints2, _ = cv2.projectPoints(weighted_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(weighted_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
        total_error += error

    mean_error = total_error / len(weighted_obj_pts)
    print("Init Mean reprojection error ", mean_error) 

    errors = np.array(errors)
    std_error = np.std(errors)
    print("std_error:",std_error)
    format = "weighted"
    np.savez(
            save_path.format(format),
            ret = ret,
            K=K,
            dist=dist,
            rvecs=np.array(rvecs, dtype=object),   # rvecs/tvecs are lists of arrays, so store as object
            tvecs=np.array(tvecs, dtype=object),
            object_points=np.array(weighted_obj_pts, dtype=object),
            image_points=np.array(weighted_corners, dtype=object),
            errors=errors
        )
    print(f"Calibration data saved to {save_path.format(format)}")
    





    return ret, K, dist, rvecs, tvecs, mean_error, errors, [], []


    # threshold = mean_error + 2 * std_error  #statistical threshold
    # #threshold = mean_error +  std_error  #statistical threshold
    # #threshold = mean_error
    
    # #Get indices of good (low-error) images
    # good_indices = np.where(errors < threshold)[0]

    # #Filter object and image points
    # object_points_filtered = [object_points[i] for i in good_indices]
    # image_points_filtered = [image_points[i] for i in good_indices]

    # # Recalibrate
    # ret, K_new, dist_new, rvecs, tvecs = cv2.calibrateCamera(
    #     object_points_filtered, image_points_filtered, image_shape, K, dist, flags=flags
    # )

    # for i in range(len(object_points_filtered)):
    #     imgpoints2, _ = cv2.projectPoints(object_points_filtered[i], rvecs[i], tvecs[i], K, dist)
    #     error = cv2.norm(image_points_filtered[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    #     errors_filtered.append(error)
    #     total_error_filtered += error

    # mean_error_filtered = total_error_filtered / len(object_points_filtered)
    # print("Filtered Mean reprojection error ", mean_error_filtered) 

    # print("\n=== Filtered Calibration Results with {} method ===".format(label))
    # print("Camera Matrix:\n", K)
    # print("Distortion Coefficients:\n", dist.ravel())

    # if True:
    #     np.savez(
    #         save_path_filt.format(d_model),
    #         K=K_new,
    #         dist=dist_new,
    #         rvecs=np.array(rvecs, dtype=object),   # rvecs/tvecs are lists of arrays, so store as object
    #         tvecs=np.array(tvecs, dtype=object),
    #         object_points_filtered=np.array(object_points, dtype=object),
    #         image_points_filtered=np.array(image_points, dtype=object),
    #         errors=errors_filtered
    #     )
    #     print(f"Calibration data saved to {save_path_filt.format(d_model)}")

    #     # ===== Read it back =====
    #     loaded = np.load(save_path_filt.format(d_model), allow_pickle=True)
    #     K_loaded = loaded["K"]
    #     dist_loaded = loaded["dist"]

    #     print("\n=== Loaded Calibration Data ===")
    #     print("Loaded Camera Matrix (K):\n", K_loaded)
    #     print("Loaded Distortion Coefficients:\n", dist_loaded.ravel())

    #     # Check if saving was correct
    #     print("\nK match:", np.allclose(K_new, K_loaded))
    #     print("dist match:", np.allclose(dist_new, dist_loaded))

    # return ret, K, dist, rvecs, tvecs, mean_error, errors, mean_error_filtered, errors_filtered

# methods = [method_auto, method_auto_clehe, method_manual, method_manual_clehe]
# labels = ["auto", "auto-clehe", "manual", "manual-clehe"]

methods = [ method_manual_clehe]
labels = [ "auto"]


# fig, axes = plt.subplots(4, 2, figsize=(14, 8))
# axes = axes.flatten()

# fig2, axes2 = plt.subplots(4, 2, figsize=(14, 8))
# axes2 = axes2.flatten()

i = 0
for key in Distorsion_models:
    print()
    print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
                  Distorsion_models[key][2])
    #flags = Distorsion_models[key][1]

    flags = (cv2.CALIB_FIX_K3 |  # keep k3 = 0
            Distorsion_models[key][1])

    method = methods[0]
    label = labels[0]
    rmse, K, dist, rvecs, tvecs, mean_error, errors, mean_error_filtered, errors_filtered = calibrate_scenario(label, method, key, flags=flags)

    # ax = axes[i]
    # ax.plot(errors, marker='o', linestyle='-', label='Per-Image Error - rmse:{}'.format(rmse))
    # ax.axhline(mean_error, color='r', linestyle='--', label=f'Mean = {mean_error:.4f}')
    # ax.set_title(f'Reprojection - {key} distortion')
    # ax.set_xlabel('Image Index')
    # ax.set_ylabel('Reprojection Error (pixels)')
    # ax.legend()
    # ax.grid(True)
    
    # rmse, K, dist, rvecs, tvecs, mean_error, errors, mean_error_filtered, errors_filtered = calibrate_scenario(label, method, key, flags=flags)
    # ax2 = axes2[i]
    # ax2.plot(errors_filtered, marker='o', linestyle='-', label='Per-Image Error - rmse:{}'.format(rmse))
    # ax2.axhline(mean_error_filtered, color='r', linestyle='--', label=f'Mean = {mean_error_filtered:.4f}')
    # ax2.set_title(f'Filtered Reprojection Error using {key}')
    # ax2.set_xlabel('Image Index')
    # ax2.set_ylabel('Reprojection Error (pixels)')
    # ax2.legend()
    # ax2.grid(True)

    # i+= 1
    # plt.draw()


# plt.tight_layout()
# plt.show()