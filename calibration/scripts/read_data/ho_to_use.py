import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)



pattern_size = (10, 7)  
square_size = 0.1 #m  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

global_path = "/media/eugeniu/T7/calibration/saved_data_raw/" #//change this to your system

#the path for the calibration parameters with raw data
intr_baseler_path = global_path+"intrinsic_baseler.npz"
intr_thermal_left = global_path+"intrinsic_left_thermal.npz"
intr_thermal_center = global_path+"intrinsic_center_thermal.npz"
intr_thermal_right = global_path+"intrinsic_right_thermal.npz"


#the path for the calibration parameters after filtering the points using 3 sigma rule from the mean error
intr_baseler_path_filt = global_path+"intrinsic_baseler_filt.npz"
intr_thermal_left_filt = global_path+"intrinsic_left_thermal_filt.npz"
intr_thermal_center_filt = global_path+"intrinsic_center_thermal_filt.npz"
intr_thermal_right_filt = global_path+"intrinsic_right_thermal_filt.npz"

def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    return [np.load(os.path.join(folder, f)) for f in files]

def read_image(img16, is_thermal = True):
    if is_thermal:
        valid_pixels = img16[img16 > 0]
        min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
        max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
        img_clipped = np.clip(img16, min_val, max_val)
        img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
        invert_img = np.array(255.0 - img8, dtype='uint8')
    else:
        invert_img = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)

    return invert_img


img_baseler_path = global_path+"baseler"
img_thermal_left = global_path+"intr_left"
img_thermal_center = global_path+"intr_center"
img_thermal_right = global_path+"intr_right"

images_16bit_left_path = load_images(img_thermal_left)
images_16bit_right_path = load_images(img_thermal_right)
images_16bit_center_path = load_images(img_thermal_center)
images_baseler_path = load_images(img_baseler_path)

for i in range(0,2 ):
    thermal_left = read_image(images_16bit_left_path[i], is_thermal = True)
    thermal_right = read_image(images_16bit_right_path[i], is_thermal = True)
    thermal_center = read_image(images_16bit_center_path[i], is_thermal = True)
    baseler = read_image(images_baseler_path[i], is_thermal = False)

    print("thermal_left shape ", np.shape(thermal_left))
    print("baseler shape ", np.shape(baseler))


#read dict intrinsic values images
baseler_intrinsic = np.load(intr_baseler_path, allow_pickle=True)
thermal_left_intrinsic = np.load(intr_thermal_left, allow_pickle=True)
thermal_center_intrinsic = np.load(intr_thermal_center, allow_pickle=True)
thermal_right_intrinsic = np.load(intr_thermal_right, allow_pickle=True)

baseler_intrinsic_filt = np.load(intr_baseler_path_filt, allow_pickle=True)
thermal_left_intrinsic_filt = np.load(intr_thermal_left_filt, allow_pickle=True)
thermal_center_intrinsic_filt = np.load(intr_thermal_center_filt, allow_pickle=True)
thermal_right_intrinsic_filt = np.load(intr_thermal_right_filt, allow_pickle=True)

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

#intrinsic baseler
print("\n Original intrinsics")
K = baseler_intrinsic["K"]
dist = baseler_intrinsic["dist"] 
errors = baseler_intrinsic["errors"] 
mean_reproj_error = np.mean(errors)

print("K:\n", K)
print("dist:", dist)
print("mean_reproj_error:", mean_reproj_error)


#intrinsic baseler filtered
print("\n Filtered intrinsics")
K = baseler_intrinsic_filt["K"]
dist = baseler_intrinsic_filt["dist"] 
errors = baseler_intrinsic_filt["errors"] 
mean_reproj_error = np.mean(errors)
print("K:\n", K)
print("dist:", dist)
print("mean_reproj_error:", mean_reproj_error)




pattern_size = (10, 7)  # (columns, rows) of inner corners
for i in range(0,10 ):
    thermal_left = read_image(images_16bit_left_path[i], is_thermal = True)
    thermal_right = read_image(images_16bit_right_path[i], is_thermal = True)
    thermal_center = read_image(images_16bit_center_path[i], is_thermal = True)
    baseler = read_image(images_baseler_path[i], is_thermal = False)

    print("thermal_left shape ", np.shape(thermal_left))


    found, corners = cv2.findChessboardCorners(thermal_left, pattern_size)
    if found:
        cv2.drawChessboardCorners(thermal_left, pattern_size, corners, found)

    found, corners = cv2.findChessboardCorners(thermal_right, pattern_size)
    if found:
        cv2.drawChessboardCorners(thermal_right, pattern_size, corners, found)

    found, corners = cv2.findChessboardCorners(thermal_center, pattern_size)
    if found:
        cv2.drawChessboardCorners(thermal_center, pattern_size, corners, found)

    found, corners = cv2.findChessboardCorners(baseler, pattern_size)
    if found:
        cv2.drawChessboardCorners(baseler, pattern_size, corners, found)


    # Undistort the left image
    K = thermal_left_intrinsic["K"]
    dist = thermal_left_intrinsic["dist"] 
    dist[-1] = 0
    
    undistorted_thermal_left = cv2.undistort(thermal_left, K, dist)
    found, corners = cv2.findChessboardCorners(undistorted_thermal_left, pattern_size)
    if found:
        cv2.drawChessboardCorners(undistorted_thermal_left, pattern_size, corners, found)

    thermal_left = cv2.vconcat([thermal_left, undistorted_thermal_left])

    # Undistort the right image
    K = thermal_right_intrinsic["K"]
    dist = thermal_right_intrinsic["dist"] 
    undistorted_thermal_right = cv2.undistort(thermal_right, K, dist)
    found, corners = cv2.findChessboardCorners(undistorted_thermal_right, pattern_size)
    if found:
        cv2.drawChessboardCorners(undistorted_thermal_right, pattern_size, corners, found)

    thermal_right = cv2.vconcat([thermal_right, undistorted_thermal_right])


    # Undistort the center image
    K = thermal_center_intrinsic["K"]
    dist = thermal_center_intrinsic["dist"] 
    undistorted_thermal_center = cv2.undistort(thermal_center, K, dist)
    found, corners = cv2.findChessboardCorners(undistorted_thermal_center, pattern_size)
    if found:
        cv2.drawChessboardCorners(undistorted_thermal_center, pattern_size, corners, found)

    thermal_center = cv2.vconcat([thermal_center, undistorted_thermal_center])


    # Undistort the baseler image
    K = baseler_intrinsic["K"]
    dist = baseler_intrinsic["dist"] 
    undistorted_baseler = cv2.undistort(baseler, K, dist)
    found, corners = cv2.findChessboardCorners(undistorted_baseler, pattern_size)
    if found:
        cv2.drawChessboardCorners(undistorted_baseler, pattern_size, corners, found)

    baseler = cv2.vconcat([baseler, undistorted_baseler])



    cv2.imshow("thermal_left", thermal_left)
    cv2.imshow("thermal_right", thermal_right)
    cv2.imshow("thermal_center", thermal_center)
    cv2.imshow("baseler", cv2.resize(baseler, None, fx=.4, fy=.4))

    cv2.waitKey(0)  

cv2.destroyAllWindows()