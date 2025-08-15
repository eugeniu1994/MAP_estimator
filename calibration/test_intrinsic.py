import cv2
import numpy as np
import matplotlib.pyplot as plt

#Center thermal camera

K = np.array([[1503.45468045,    0.    ,      324.32583388],
              [0.   ,      1503.07102039 , 262.46373444],
              [0,  0,  1]])  # Intrinsic matrix

dist = np.array([-0.05857507, -5.11546886 , 0.00207716 , 0.00055662 ,59.30916158])  # Distortion coefficients


img_path  = "/media/eugeniu/T7/calibration/saved_data_raw/intr_thermal_center/image_0010.npy"

def convert_16bit_to_8bit_auto(img16, clehe = False, lower_pct=2, upper_pct=98):
    nonzero = img16[img16 > 0]
    min_val = np.percentile(nonzero, lower_pct)
    max_val = np.percentile(nonzero, upper_pct)
    img_clipped = np.clip(img16, min_val, max_val)
    img8 = ((img_clipped - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)

    return img8

img16 = np.load(img_path)
print('img16:', np.shape(img16))

img8 = convert_16bit_to_8bit_auto(img16)

cv2.imshow("img8", img8)
cv2.waitKey(1)

img = np.array(256 - img8, dtype='uint8')

cv2.imshow("img", img)
cv2.waitKey(0)

gray = img 


pattern_size = (10, 7)  # (columns, rows) of inner corners
found, corners = cv2.findChessboardCorners(gray, pattern_size)

img_with_corners = img.copy()
if found:
    cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, found)

# === Undistort the image ===
h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

img_undistorted = cv2.undistort(img, K, dist, None, new_K)


gray_undistorted = img_undistorted # cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
found_undistorted, corners_undistorted = cv2.findChessboardCorners(gray_undistorted, pattern_size)

img_undistorted_with_corners = img_undistorted.copy()
if found_undistorted:
    cv2.drawChessboardCorners(img_undistorted_with_corners, pattern_size, corners_undistorted, found_undistorted)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image with Corners')
plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Undistorted Image with Corners')
plt.imshow(cv2.cvtColor(img_undistorted_with_corners, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
