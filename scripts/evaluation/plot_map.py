import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString

import numpy as np
import os

font = 16

plt.rcParams.update({'font.size': font})
plt.rc('axes', titlesize=font)     # Set the font size of the title
plt.rc('axes', labelsize=font)     # Set the font size of the x and y labels
plt.rc('xtick', labelsize=12)    # Set the font size of the x tick labels
plt.rc('ytick', labelsize=12)    # Set the font size of the y tick labels
plt.rc('legend', fontsize=14)    # Set the font size of the legend
plt.rc('font', size=font)          # Set the general font size'''


def traveled_distance(points):
    # Compute differences between consecutive points
    diffs = np.diff(points, axis=0)
    # Compute Euclidean distances for each segment
    distances = np.linalg.norm(diffs, axis=1)
    # Sum all distances
    total_distance = np.sum(distances)
    return total_distance

translation_ENU_to_origin = np.array([4525805.18109165318310260773, 5070965.88124799355864524841,  114072.22082747340027708560])
rotation_ENU_to_origin = np.array([[-0.78386212744029792887, -0.62091317757072628236,  0.00519529438949398702],
                                    [0.62058202788821892337, -0.78367126767620609584, -0.02715310076057271885],
                                    [0.02093112101431114647, -0.01806018100107483967,  0.99961778597386541367]])


trajectory_file = '/home/eugeniu/x_test/MLS.txt'

def get_traj():
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    traj_all = np.loadtxt(trajectory_file)
    if traj_all.ndim != 2 or traj_all.shape[1] < 4:
        raise ValueError("Trajectory file must contain at least 4 columns (timestamp, x, y, z)")

    traj_local = traj_all[:, 2:5]
    if np.isnan(traj_local).any():
        raise ValueError("Trajectory contains NaN values")

    print(f"[INFO] Loaded trajectory with shape: {traj_local.shape}")

    # Inverse of rotation and translation
    R_origin_to_ENU = rotation_ENU_to_origin.T
    t_origin_to_ENU = -R_origin_to_ENU @ translation_ENU_to_origin

    # Apply transformation to convert local to ENU
    traj_ENU = (R_origin_to_ENU @ traj_local.T).T + t_origin_to_ENU

    if np.isnan(traj_ENU).any():
        raise ValueError("Converted ENU trajectory contains NaN values")

    print(f"[INFO] First ENU point: {traj_ENU[0]}")
    return traj_ENU

traj_points_ENU = get_traj()

print('traveled_distance:',traveled_distance(traj_points_ENU))

def get_cloud():
    input_folder = '/home/eugeniu/Desktop/LIG-poses/clouds/test3'
    combined_points = []
    i=0
    k=25
    for filename in os.listdir(input_folder):
        print('filename ',filename)
        if filename.endswith('.txt'):
            i+=1
            input_path = os.path.join(input_folder, filename)
            
            data = np.loadtxt(input_path)
            
            _3D_values_in_mls = data[:,:3]
            _3D_values_in_ENU = (rotation_origin_to_ENU @ _3D_values_in_mls.T).T + translation_origin_to_ENU
            _3D_values_in_ENU[:,2] = data[:,3] # intensity
            
            if len(combined_points) == 0:
                combined_points = _3D_values_in_ENU[::k]
            else:
                combined_points = np.vstack((combined_points, _3D_values_in_ENU[::k]))
            
            if i>20:
                break
        
    return combined_points[:,:2],combined_points[:,2]

def get_trees():
    trajectory_name = 'Robust MLS + Dense ALS + Map Fusion'
    
    path = '/home/eugeniu/Desktop/trees/'
        
    method_path = path+trajectory_name
                        
    als_trees = np.genfromtxt(method_path+"/ALS_Correspondences_xyz.txt", delimiter=',')
    mls_trees = np.genfromtxt(method_path+"/MLS_corresponding_trees_xyz.txt", delimiter=',')
            
    print('\nmethod:{}  als_trees:{}, mls_trees:{}'.format(trajectory_name, np.shape(als_trees), np.shape(mls_trees)))


    return als_trees[:,:2], mls_trees[:,:2]
    
            
etrs_tm35fin = 'EPSG:3067'
wgs84 = 'EPSG:4326'  # WGS84 for latitude and longitude

start = traj_points_ENU[0]
print('start:',start)

# Create a GeoDataFrame with the EVO location point in the ETRS-TM35FIN projection
evo_point = gpd.GeoDataFrame(
    {'geometry': [Point(398252.412, 6786205.990)]},
    #{'geometry': [Point(start[0],start[1])]},
    crs=etrs_tm35fin
)

evo_area = evo_point.buffer(2000)  # 2000 meters = 2 km


# Define 2D points (East and North) in ETRS-TM35FIN (EPSG:3067)
trajectory_points_etrstm35fin = np.array([
    [398200.0, 6786000.0],  # Starting point near the EVO area
    [398300.0, 6786050.0],  # Additional points to smooth the curve
    [398400.0, 6786075.0],
    [398500.0, 6786100.0],  # Mid-point
    [398600.0, 6786150.0],
    [398700.0, 6786180.0],
    [398800.0, 6786205.990], # End point (close to the EVO point)
])

trajectory_points_etrstm35fin = traj_points_ENU[::15,:2]

# Create GeoDataFrame from the points (ETRS-TM35FIN)
trajectory_gdf = gpd.GeoDataFrame(
    {'geometry': [Point(x, y) for x, y in trajectory_points_etrstm35fin]},
    crs=etrs_tm35fin
)

# als_trees, mls_trees = get_trees()
# print('als_trees:',np.shape(als_trees),', mls_trees:',np.shape(mls_trees))
# als_trees_gdf = gpd.GeoDataFrame(
#     {'geometry': [Point(x, y) for x, y in als_trees]},
#     crs=etrs_tm35fin
# )
# mls_trees_gdf = gpd.GeoDataFrame(
#     {'geometry': [Point(x, y) for x, y in mls_trees]},
#     crs=etrs_tm35fin
# )

#print('start iterating over the data')
#data, intensity = get_cloud()
#print('data:', np.shape(data))
#cloud_gdf = gpd.GeoDataFrame(
#    {'geometry': [Point(x, y) for x, y in data]},
#    crs=etrs_tm35fin
#)
#print('here1')

# Reproject to Web Mercator (EPSG:3857) to match the tile provider
evo_area = evo_area.to_crs(epsg=3857)
# Reproject the trajectory back to Web Mercator (EPSG:3857) for plotting with the basemap
trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=3857)

#als_trees_gdf_mercator = als_trees_gdf.to_crs(epsg=3857)
#mls_trees_gdf_mercator = mls_trees_gdf.to_crs(epsg=3857)


#cloud_gdf_mercator = cloud_gdf.to_crs(epsg=3857)
#print('here2')



fig, ax = plt.subplots(figsize=(20, 20))

# Plot the area boundary
evo_area.boundary.plot(ax=ax, color='red')
# Plot the trajectory points in Web Mercator (projected for plotting)
trajectory_gdf_mercator.plot(ax=ax, color='red', marker='o', markersize=15, label='Trajectory')

# mls_trees_gdf_mercator.plot(ax=ax, color='tab:orange', marker='o', markersize=130, label='MLS trees')
# als_trees_gdf_mercator.plot(ax=ax,  marker='*', markersize=100, label='Reference ALS trees')


#cloud_gdf_mercator.plot(ax=ax, c=intensity, cmap='rainbow', markersize=.5, label='MLS')
#print('done plot')

# Add the satellite basemap using ESRI's satellite imagery
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

#ctx.add_basemap(ax, source="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
#ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=30)  # Adjust zoom level for higher detail


# Set axis limits based on the area
ax.set_xlim(evo_area.total_bounds[[0, 2]])  # X limits (longitude)
ax.set_ylim(evo_area.total_bounds[[1, 3]])  # Y limits (latitude)

ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
plt.legend()
# Set the formatter for x and y axis to avoid scientific notation
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
# Display the plot
plt.show()
