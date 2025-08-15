#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import glob
import matplotlib.pyplot as plt



bridge = CvBridge()

bags_folder = "/media/eugeniu/T7/calibration/rosbags/"

# Define bag file patterns
basler_pattern = "extrinsic_cameras_camera_basler_front_*.bag"
thermal_pattern = "_extrinsic_cameras_thermal_camera_Flir_Center_*.bag"

basler_pattern = "extrinsic_cameras_camera_basler_front_24219235_2025-08-11-16-02-29_0.bag"

basler_files = glob.glob(bags_folder+basler_pattern)
thermal_files = glob.glob(bags_folder+thermal_pattern)

# Define camera topics
thermal_topic = "/thermal_camera_Flir_Center_0161886/image_raw"  
basler_topic = "/camera_basler_front_24219235/image_raw"
basler_extras = "/camera_basler_front_24219235/extras"


todo - read the baseler and gnss 
for each baseler - save it into a buffer 

when trigger got -> use the start and end gnss time to interpolate the time of the triggering
for thermal - where to get the time ? 



bags = []
for f in basler_files + thermal_files:
    bags.append(rosbag.Bag(f, "r"))


for f in basler_files + thermal_files:
    print("F",f)

bag_iters = [bag.read_messages() for bag in bags]

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

try:
    while True:
        for i, bag_iter in enumerate(bag_iters):
            try:
                topic, msg, t = next(bag_iter)
            except StopIteration:
                continue  # skip if bag is done

            if topic in basler_topic:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                window_name = topic.split("/")[-2] + "_basler"
                cv2.imshow("window_name", cv2.resize(cv_image, None, fx=.4, fy=.4))

            #elif topic in basler_extras:
                #print("extras")
                # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                # window_name = topic.split("/")[-2] + "_basler"
            elif topic == thermal_topic:
                #cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_image = read_image(cv_image, is_thermal = True)
                window_name = "thermal_camera"
                cv2.imshow(window_name, cv_image)
            else:
                continue


            

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
finally:
    for bag in bags:
        bag.close()
    cv2.destroyAllWindows()








 




chessboard_size = (10, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
                key = cv2.waitKey(0)

                
    except Exception as e:
        print(f"Error reading bag '{bag_path}': {e}")

cv2.destroyAllWindows()
print("Done.")
