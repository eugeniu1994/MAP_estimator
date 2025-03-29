import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_lib

def estimate_frequency_first_two(data):
    if len(data) < 2:
        return None 

    time_diff = data[1,0] - data[0,0]  # Difference between first two timestamps
    frequency = 1 / time_diff if time_diff > 0 else 0
    return frequency

def extract_gnss_data_vux(file_path, time_limit = 0):
    data = []
    data_section = False  # Flag to start reading after headers

    with open(file_path, 'r') as file:
        for line in file:
            # Detect the header row
            if re.match(r'\s*GPSTime\s+Easting\s+Northing', line):
                data_section = True  # Start reading data from the next line
                continue  # Skip the header row itself
            
            if data_section:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                
                # Ensure there are at least 5 values (GPSTime, Easting, Northing, H-Ell, Heading)
                if len(numbers) >= 5:
                    gps_time, easting, northing, h_ell, heading = map(float, numbers[:5])
                    # if gps_time < time_limit:
                    #     continue

                    data.append([gps_time, easting, northing, h_ell, heading])

    return np.array(data)

def extract_gnss_data_mls(file_path):
    data = []
    data_section = False  

    with open(file_path, 'r') as file:
        for line in file:
            # Detect the header row
            if re.match(r'\s*UTCTime\s+GPSTime\s+Easting\s+Northing', line):
                data_section = True 
                continue  # Skip the header row itself
            
            if data_section:
                columns = line.strip().split()
                
                # Ensure there are enough values in the row
                if len(columns) >= 5:
                    try:
                        gps_time = float(columns[1])  # GPSTime (2nd column)
                        easting = float(columns[2])   # Easting (3rd column)
                        northing = float(columns[3])  # Northing (4th column)
                        h_ell = float(columns[4])     # H-Ell (5th column)
                        heading = float(columns[5])   # Heading (6th column)
                        data.append([gps_time, easting, northing, h_ell, heading])
                    except ValueError:
                        continue  # Skip lines that cause errors

    return np.array(data)




# Read the file
file_path = "/media/eugeniu/T7/roamer/mls-gt/MLS.txt"
hesai = np.loadtxt(file_path, usecols=(1, 2, 3))  # Extract only columns 1, 2, 3 (translation)
print('hesai:', np.shape(hesai))

#als 2 mls 
t = np.array([4.181350e+06, 5.355192e+06, 2.210141e+05])
R = np.array([[-8.231261e-01, -5.678501e-01 ,-3.097665e-03],
              [ 5.675232e-01, -8.224404e-01, -3.884915e-02],
              [ 1.951285e-02, -3.373575e-02,  9.992403e-01]])

R_inv = R.T  # Transpose of rotation matrix
t_inv = -R_inv @ t  # Negative of rotated translation


print('t_inv ', t_inv)

#hesai = (R_inv @ hesai.transpose()).transpose() #+ t_inv
hesai = (R_inv@(hesai-t).transpose()).transpose()
#hesai = hesai-t

np.set_printoptions(suppress=True)

mls_file_path = "/media/eugeniu/T7/roamer/mls-gt/car-MLS-25-07.txt"  
mls_data = extract_gnss_data_mls(mls_file_path) #gps_time, easting, northing, h_ell, heading


vux_file_path = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodout.txt"  
vux_data = extract_gnss_data_vux(vux_file_path, mls_data[0,0]) #gps_time, easting, northing, h_ell, heading

mls_data = mls_data[::100,:]
mls_data = mls_data[:len(vux_data)]

# mls_data = mls_data[540:600,:]
# vux_data = vux_data[540:600,:]

#vux_data = vux_data[:1200,:]
# Skip the first 18 minutes (1080 rows) 18×60=1080
vux_data = vux_data[:1080]

# mls_data -= mls_data[0,:]
# vux_data -= vux_data[0,:]


print('shapes vux_data:', np.shape(vux_data), ' mls_data:', np.shape(mls_data))

frequency_vux = estimate_frequency_first_two(vux_data)
frequency_mls = estimate_frequency_first_two(mls_data)
print('frequency_vux:', frequency_vux, ' frequency_mls:', frequency_mls)

print('vux_data:\n',vux_data[:5,0])
print('mls_data:\n',mls_data[:5,0])

print('last point ')
print('vux_data:\n',vux_data[-1,0])
print('mls_data:\n',mls_data[-1,0])






first_mls_time = mls_data[0,0]
first_vux_time = vux_data[0,0]

# time_diff = np.linalg.norm(vux_data[:,0] - mls_data[:,0])
# print('time_diff ',time_diff)

# diff = np.linalg.norm(vux_data[:,1:4] - mls_data[:,1:4], axis = 1)

# print('diff:', np.shape(diff))

# plt.figure()
# plt.plot(vux_data[:,0],diff, label = 'difference norm')
# plt.legend()
# plt.draw()


# plt.figure()
# plt.plot(vux_data[:,0],vux_data[:,1]- mls_data[:,1], label = 'difference X')
# plt.plot(vux_data[:,0],vux_data[:,2]- mls_data[:,2], label = 'difference Y')
# plt.plot(vux_data[:,0],vux_data[:,3]- mls_data[:,3], label = 'difference Z')
# plt.legend()
# #plt.show()


# plt.plot(vux_data[:,0], vux_data[:,-1],label = 'vux')
# plt.plot(mls_data[:,0], mls_data[:,-1],label = 'mls')
# plt.legend()
# plt.show()

def estimate_extrinsic_transformation(P1, P2):
    """
    Estimate the extrinsic transformation (R, t) between two sensors based on their trajectories.
    
    Parameters:
        P1 (np.ndarray): Trajectory from the first sensor (Nx3).
        P2 (np.ndarray): Trajectory from the second sensor (Nx3).
    
    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
    """
    # Ensure the inputs are numpy arrays
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    
    # Center the data
    centroid1 = np.mean(P1, axis=0)
    centroid2 = np.mean(P2, axis=0)
    print('centroid1:',centroid1)
    print('centroid2:',centroid2)

    Q1 = P1 - centroid1
    Q2 = P2 - centroid2
    
    # Compute the covariance matrix
    H = Q1.T @ Q2
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T
    
    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    

    print('centroid1-centroid2 = ',centroid1-centroid2)
    # Compute the translation vector
    t = centroid1 - R @ centroid2
    
    return R, t

# Perform ICP registration
def icp_registration(src_pcd, tgt_pcd, threshold = 0.1): #0.1m # Max correspondence distance
    trans_init = np.eye(4)  # Initial guess (identity)
    # Run Point-to-Point ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))

    return reg_p2p.transformation

plot_data = True
if plot_data:
    import open3d as o3d

    vux_points = vux_data[:,1:4]
    vux_points -= vux_points[0] #move to origin
    point_cloud_vux = o3d.geometry.PointCloud()
    point_cloud_vux.points = o3d.utility.Vector3dVector(vux_points)
    point_cloud_vux.colors = o3d.utility.Vector3dVector(np.full((len(vux_points), 3), [0, 0, 1]))

    mls_points = mls_data[:,1:4]
    mls_points -= mls_points[0] #move to origin
    mls_points[:,-1] += 350
    point_cloud_mls = o3d.geometry.PointCloud()
    point_cloud_mls.points = o3d.utility.Vector3dVector(mls_points)
    point_cloud_mls.colors = o3d.utility.Vector3dVector(np.full((len(mls_points), 3), [0, 1, 0]))

    #print('vux_points[0]:',vux_data[0])
    #print('mls_points[0]:',mls_points[0])

    #diff = vux_points[0] - mls_points[0]
    #print('diff:',diff)


    # # Trajectory from sensor 1 (Nx3)
    # P1 = vux_points # np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # # Trajectory from sensor 2 (Nx3)
    # P2 = mls_points # np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

    # print('P1:', np.shape(P1)," P2:", np.shape(P2))

    # # Estimate the extrinsic transformation
    # R, t = estimate_extrinsic_transformation(P1, P2)

    # print("Rotation matrix R:\n", R)
    # print("Translation vector t:\n", t)
    # euler_angles = R_lib.from_matrix(R).as_euler('xyz', degrees=True)  # 'xyz' is the rotation order
    # print("Euler angles (degrees):", euler_angles)

    # # Verify the transformation
    # P2_transformed = (R @ P2.T).T + t
    # #print("Transformed P2:\n", P2_transformed)
    # #P2_transformed[:,-1] += 450

    #diff_mls2vux = np.linalg.norm(mls_points - vux_points)
    #print('diff_mls2vux ',diff_mls2vux)
    #diff_transformed2vux = np.linalg.norm(P2_transformed - vux_points)
    #print('diff_transformed2vux ',diff_transformed2vux)

    #point_cloud_transformed = o3d.geometry.PointCloud()
    #point_cloud_transformed.points = o3d.utility.Vector3dVector(P2_transformed)


    #hesai[:,-1] -= 10
    #hesai -= vux_points[0]
    point_cloud_hesai = o3d.geometry.PointCloud()
    point_cloud_hesai.points = o3d.utility.Vector3dVector(hesai)
    # Set color to RED (RGB: [1, 0, 0])
    point_cloud_hesai.colors = o3d.utility.Vector3dVector(np.full((len(hesai), 3), [1, 0, 0]))
    
    #transformation = icp_registration(point_cloud_hesai, point_cloud_vux,10)
    #print("Estimated Transformation point_cloud_hesai to point_cloud_vux :\n", transformation)
    #transformation = icp_registration(point_cloud_mls, point_cloud_vux,1)
    #print("\bEstimated Transformation point_cloud_mls to point_cloud_vux :\n", transformation)


    o3d.visualization.draw_geometries([point_cloud_vux, point_cloud_mls,], window_name="Point Cloud Visualization")