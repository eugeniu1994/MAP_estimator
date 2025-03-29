import os
import numpy as np
import open3d as o3d
from scipy.spatial import distance
import pandas as pd
from pypcd4 import PointCloud
import laspy

t = np.array([ 5135054.31356109771877527237, -4274603.96961564384400844574 ,   -1499.89635409739435090160])
R = np.array([[0.59443595439439411710 ,-0.80205163270152801669, -0.05795752413655200219],
              [ 0.79900818338198764668 , 0.59723605695351378309, -0.06996438496288906461],
              [ 0.09072937237654582487 ,-0.00471919012463235116 ,0.99586440353731175978]])

t = np.dot(-R.transpose(), t )
R = R.transpose()

# Define input and output directories
input_folder = "/media/eugeniu/T7 Shield/Masalantie_MLS-ALS/clouds/"
output_folder = "/media/eugeniu/T7 Shield/Masalantie_MLS-ALS/clouds_croped/" 

#for point clouds in ALS frame
#pcd_files = sorted([f for f in os.listdir(input_folder) if f.startswith("als_") and f.endswith(".pcd")])
pcd_files = ([f for f in os.listdir(input_folder) if f.startswith("als_") and f.endswith(".pcd")])

#for point clouds in MLS frame
pcd_files = ([f for f in os.listdir(input_folder) if f.startswith("mls_") and f.endswith(".pcd")])

print('There are {} files'.format(len(pcd_files)))

distance_threshold = 100 #m

current_segment = None
segment_mean = None
segment_idx = 0

def load_pcd_with_fields(pcd_path):
    """Load a .pcd file while keeping all fields using pypcd4."""
    pcd = PointCloud.from_path(pcd_path)
    
    print('pcd.fields:',pcd.fields)
    data = pcd.numpy()

    return pcd, data

def save_segment(current_segment, segment_idx):
    filtered_data = current_segment[(current_segment[:, 4] >= 3) & (current_segment[:, 4] <= 7)]

    xyz = filtered_data[:,:3] # N x 3

    xyz_als = np.dot(xyz, R.T) + t
    filtered_data[:,:3] = xyz_als
    print('current_segment:',np.shape(current_segment),', filtered_data:', np.shape(filtered_data))

    #path = output_folder+'whole_cloud_segment_{}.txt'.format(segment_idx)
    #np.savetxt(path, current_segment, fmt='%.10f', delimiter='\t')

    path = output_folder+'filtered_cloud_segment_{}.txt'.format(segment_idx)
    np.savetxt(path, filtered_data, fmt='%.10f', delimiter='\t')

    
for file in pcd_files:
    pcd_path = os.path.join(input_folder, file)
    #print(f"\nReading {file}")

    pcd, points = load_pcd_with_fields(pcd_path)
    #print('data:{}'.format(np.shape(points)))

    mean_point = np.mean(points[:, :3], axis=0)  # Compute mean XYZ coordinates

    if current_segment is None:
        current_segment = points
        segment_mean = mean_point
        pcd_template = pcd 
    else:
        # Compute distance between current mean and previous segment mean
        dist = np.linalg.norm(segment_mean - mean_point)

        if dist < distance_threshold:
            current_segment = np.vstack((current_segment, points))
            print('current_segment ', np.shape(current_segment))
        else:
            print('save the current segment')
            save_segment(current_segment, segment_idx)

            segment_idx += 1
            current_segment = points
            segment_mean = mean_point
            pcd_template = pcd 

    