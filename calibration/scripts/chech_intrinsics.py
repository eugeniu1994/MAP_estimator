import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

Distorsion_models = {'ST': ['Standard', 0, 'Standard'],
                             'RAT': ['Rational', cv2.CALIB_RATIONAL_MODEL, 'CALIB_RATIONAL_MODEL'],
                             'THP': ['Thin Prism', cv2.CALIB_THIN_PRISM_MODEL, 'CALIB_THIN_PRISM_MODEL'],
                             'TIL': ['Tilded', cv2.CALIB_TILTED_MODEL, 'CALIB_TILTED_MODEL'],  # }
                             'RAT+THP': ['Rational+Thin Prism', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL,
                                         'CALIB_RATIONAL_MODEL + CALIB_THIN_PRISM_MODEL'],
                             'THP+TIL': ['Thin Prism+Tilded', cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                                         'CALIB_THIN_PRISM_MODEL + CALIB_TILTED_MODEL'],
                             'RAT+TIL': ['Rational+Tilded', cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_TILTED_MODEL,
                                         'CALIB_RATIONAL_MODEL + CALIB_TILTED_MODEL'],
                             'CMP': ['Complete',
                                     cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL,
                                     'Complete']
                                     }

#standard ST
save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_right_thermal_{}.npz"
save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_right_thermal_filt_{}.npz"
title = "Distance from the expected right camera"

#complete CMP
save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_left_thermal_{}.npz"
save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_left_thermal_filt_{}.npz"
title = "Distance from the expected left camera"


save_path = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_center_thermal_{}.npz"
save_path_filt = "/media/eugeniu/T7/calibration/saved_data_raw/intrinsic_center_thermal_filt_{}.npz"
title = "Distance from the expected center camera"


cy = 512./2
cx = 640./2

expected_center = np.array([cx,cy])
print("cx:",cx,", cy:",cy)
errors = []
labels = []
all_K, all_D = [],[]
plt.figure()
plt.title(title)
i=0
for key in Distorsion_models:
    #print()
    # print(key, '->', Distorsion_models[key][0], ' , ', Distorsion_models[key][1], ' , ',
    #               Distorsion_models[key][2])
    
    f = save_path.format(key)
    intrinsic = np.load(f, allow_pickle=True)
    K = intrinsic["K"]
    D = intrinsic["dist"]
    all_K.append(K)
    all_D.append(D)
    computed_center = np.array([K[0,2], K[1,2]])
    d = np.linalg.norm(expected_center - computed_center)

    f2 = save_path.format(key)  
    intrinsic2 = np.load(f2, allow_pickle=True)
    K2 = intrinsic2["K"]
    D2 = intrinsic2["dist"]
    all_K.append(K2)
    all_D.append(D2)
    computed_center = np.array([K2[0,2], K2[1,2]])
    d2 = np.linalg.norm(expected_center - computed_center)  

    errors.append(d)
    errors.append(d2)

    labels.append(key)
    labels.append(key+"_filt")

    
    plt.scatter(i,d, label=key)
    i+=1

    plt.scatter(i,d2, label=key+"_filt")
    i+=1

errors = np.array(errors)
labels = np.array(labels)

min_idx = np.argmin(errors)
print("Smallest error ",errors[min_idx]," obtained with ", labels[min_idx])
print("Best K:\n", all_K[min_idx])
print("Best D:\n", all_D[min_idx])

plt.grid(True)
plt.legend()
plt.ylabel("Distance from the expected center")
plt.show()

