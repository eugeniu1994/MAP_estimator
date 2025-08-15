#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import glob
import matplotlib.pyplot as plt


bags_folder = "/media/eugeniu/T7/calibration/rosbags"
 
#thermal camera center
image_topic = "/thermal_camera_Flir_Center_0161886/image_raw"       
file_prefix = "intrinsic_calibration_center_thermal_camera_Flir_Center_"
save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_thermal_center" 

#thermal camera left
image_topic = "/thermal_camera_Flir_Left_0161605/image_raw"       
file_prefix = "intrinsic_calibration_left_thermal_camera_Flir_Left"
save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_thermal_left" 

#thermal camera right
image_topic = "/thermal_camera_Flir_Right_0161599/image_raw"       
file_prefix = "intrinsic_calibration_right_thermal_camera_Flir_Right"
save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/intr_thermal_right" 


chessboard_size = (10, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def detect_chessboard(img):
    invert_img = np.array(256 - img, dtype='uint8')
    ret, corners = cv2.findChessboardCorners(invert_img, chessboard_size, None)
    if ret:
        #print('corners [0]', corners[0])
        backtorgb = cv2.cvtColor(invert_img.copy(),cv2.COLOR_GRAY2RGB)

        #print("[INFO] Chessboard detected with {} corners".format(len(corners)))
        #corners_refined = cv2.cornerSubPix(invert_img, corners, (11, 11), (-1, -1), criteria)
        #checkerboard_img = cv2.drawChessboardCorners(backtorgb, chessboard_size, corners_refined, ret) 
        checkerboard_img = cv2.drawChessboardCorners(backtorgb, chessboard_size, corners, ret)

        return True, checkerboard_img
    else:
        print("No chessboard detected...")
        return False, img
    
bridge = CvBridge()
image_counter = 0

bag_files = sorted(glob.glob(os.path.join(bags_folder, f"{file_prefix}*.bag")))
if not bag_files:
    print(f"No bag files found in '{bags_folder}' starting with '{file_prefix}'")
    exit(1)

print(f"Found {len(bag_files)} bag files.")
show_histogram = False
wait_time = 0
for bag_path in bag_files:
    print(f"Reading bag: {bag_path}")
    try:
        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[image_topic]):
                
                try:
                    img16 = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                except Exception as e:
                    print(f"Failed to convert image: {e}")
                    continue
                
                # print("\nmsg.encoding:",msg.encoding)
                # print("img16 encoding:",type(img16), "dtype:", img16.dtype, "shape:", img16.shape)            
                # print("min:", np.min(img16), "max:", np.max(img16))

                #Normalize using histogram stretch---------------------------------------------------
                # Compute min and max ignoring 0 (bad pixels)
                valid_pixels = img16[img16 > 0]
                min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
                max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this

                # Clip to [min_val, max_val] and normalize to 8-bit
                img_clipped = np.clip(img16, min_val, max_val)
                #img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
                
                img8 = np.array(255.0 - img8, dtype='uint8') 

                # min_val, max_val = 22200,23200 #for center thermal
                # min_val, max_val = 22600,23400 #for left thermal
                # min_val, max_val = 22800,23550 #for right thermal

                # img_clipped = np.clip(img16, min_val, max_val)

                # img8_test = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                
                if True:
                    # CLAHE (adaptive histogram equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img8_clahe = clahe.apply(img8)
                    

                    #img8_eq = cv2.equalizeHist(img8)

                if show_histogram:
                    hist, bin_edges = np.histogram(img16, bins=65536, range=(0, 65535))
                    plt.plot(bin_edges[:-1], hist)
                    plt.title("16-bit Thermal Image Histogram")
                    plt.xlabel("Pixel Intensity (raw)")
                    plt.ylabel("Frequency")
                    plt.grid(True)
                    plt.draw()

                    plt.figure()
                    hist, bin_edges = np.histogram(img8, bins=256, range=(0, 255))
                    plt.plot(bin_edges[:-1], hist, alpha = .7, label = 'img8')
                    plt.title("8-bit Thermal Image Histogram")
                    plt.xlabel("Pixel Intensity (raw)")
                    plt.ylabel("Frequency")
                    plt.grid(True)
                    plt.legend()
                    plt.draw()

                    # plt.figure()
                    # hist, bin_edges = np.histogram(img8_test, bins=256, range=(0, 255))
                    # plt.plot(bin_edges[:-1], hist, alpha = .7, label = 'img8_test')
                    # plt.title("8-bit test Thermal Image Histogram")
                    # plt.xlabel("Pixel Intensity (raw)")
                    # plt.ylabel("Frequency")
                    # plt.grid(True)
                    # plt.legend()
                    # plt.draw()

                    plt.figure()
                    hist, bin_edges = np.histogram(img8_clahe, bins=256, range=(0, 255))
                    plt.plot(bin_edges[:-1], hist, alpha = .7,label = 'img8_clahe')
                    plt.title("8-bit img8_clahe Thermal Image Histogram")
                    plt.xlabel("Pixel Intensity (raw)")
                    plt.ylabel("Frequency")
                    plt.grid(True)
                    plt.legend()
                    plt.draw()

                    # plt.figure()
                    # hist, bin_edges = np.histogram(img8_eq, bins=256, range=(0, 255))
                    # plt.plot(bin_edges[:-1], hist,alpha = .7, label='img8_eq')
                    # plt.title("8-bit img8_eq Thermal Image Histogram")
                    # plt.xlabel("Pixel Intensity (raw)")
                    # plt.ylabel("Frequency")
                    # plt.grid(True)
                    # plt.legend()
                    # plt.draw()
                    show_histogram = False
                    plt.show()

                #if True:
                ret, img8 = detect_chessboard(img8)
                    # if ret:
                #ret, img8_test = detect_chessboard(img8_test)
                #ret, img8_clahe = detect_chessboard(img8_clahe)
                
                    #img8_eq = detect_chessboard(img8_eq)
                    

                cv2.imshow("img8 chessboard", img8)
                #cv2.imshow("img8_test chessboard", img8_test)
                cv2.imshow("img8_clahe chessboard", img8_clahe)
                #cv2.imshow("img8_eq chessboard", img8_eq)
                    
                cv2.waitKey(1)

                print("Press SPACE for next, 's' to save, or ESC to quit.")
                key = cv2.waitKey(1) & 0xFF

                if True:
                    if ret:
                        key = cv2.waitKey(wait_time)
                    else: 
                        cv2.waitKey(1)
                    if key == 27:  # ESC
                        print("Exiting.")
                        cv2.destroyAllWindows()
                        exit(0)
                    elif key == ord('s') or key == ord('S'):
                        if ret:
                            filename = os.path.join(save_dir, f"image_{image_counter:04d}.npy")
                            np.save(filename, img16)
                            print(f"Saved: {filename}")
                            image_counter += 1
                    elif key == ord('p') or key == ord('P'):
                        show_histogram = True   
                    elif key == ord('q') or key == ord('Q'):
                        wait_time = 50
                    elif key == ord('a') or key == ord('A'):
                        wait_time = 0            
                    elif key == 32:  # SPACE
                        continue
                    else:
                        print("Invalid key. Press SPACE, 's', or ESC.")
    except Exception as e:
        print(f"Error reading bag '{bag_path}': {e}")

cv2.destroyAllWindows()
print("Done.")
