# MAP LIO estimator

## Work in Progress
This code was developed as part of the work for the paper:

**Robust Tightly Coupled MLS-ALS Fusion with 2D/3D LiDAR Using Data-Driven Covariances for Accurate 3D Mapping**, submitted to the *ISPRS Journal of Photogrammetry and Remote Sensing*.

![MAP_LIO]([https://github.com/eugeniu1994/MAP_LIO/blob/master/demo.png]))

![MAP_LIO]([https://github.com/eugeniu1994/FPR/blob/c896008b1138366255c03d361d6a3ca314d78f84/paper-teaser.png](https://github.com/eugeniu1994/MAP_LIO/blob/master/demo.png?raw=true))


---
## ✨ Features

- **Range image projection** for efficient organisation of LiDAR points  
- **Rank estimation**   
- **Rank-based voxelization**:  
  - First point per voxel (highest rank)  
  - Weighted average of points (rank-weighted centroid)  
- **Scan cleaning**: drop low-rank/noisy points (configurable percentage)  
- Designed for **robust LiDAR odometry & mapping in rain, fog, or dust**  

---

## Requirements

To build and run this project, the following libraries are required:

- [Eigen]([http://eigen.tuxfamily.org](https://libeigen.gitlab.io/))
- [PCL (Point Cloud Library)](http://pointclouds.org)
- [Sophus]([https://github.com/strasdat/Sophus](https://github.com/strasdat/Sophus))
For prior map usage with ALS data (`.las` files), the [LASTools](https://lastools.github.io/) library is required.
---

## 🚀 Build Instructions

```sh
cd ~/catkin_ws/src/ #change this according to your system
git clone https://github.com/eugeniu1994/FPR.git
cd ..
```

### Lidar-Inertial Navigation only (no prior map / ALS)
```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="map_lio" -DUSE_ALS=OFF
```
## If you want to use prior ALS map data: 
```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="map_lio" -DUSE_ALS=ON
```

▶️ Usage

Example of running the system with ROS:
```bash
roslaunch map_lio hesai.launch bag_file:=bag_file_path
```


🛠️ TODO

Add support for different LiDARs (currently only Hesai is supported)
Note: In the paper, the Riegl VUX LiDAR was used, which requires proprietary software API to read the data. Therefore, this implementation has been omitted from public access


##  License
Academic Research Use Only
This software is provided for academic research purposes only. It is not licensed for commercial use.
By downloading or using this software, you agree to use it solely for non-commercial, research-oriented purposes.
For commercial licensing inquiries, please contact the authors.


📧 Maintainer
Eugeniu Vezeteu
📩 vezeteu.eugeniu@yahoo.com
 











