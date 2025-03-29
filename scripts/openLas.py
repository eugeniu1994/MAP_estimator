import numpy as np
import open3d as o3d
import laspy
from datetime import datetime, timezone

f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-B/240725_092351.las"
#f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-B/240725_092351_v12.las"
#f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-B/out_v12.las"

#f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-A/240725_093813.las"

def convert_las_to_12(input_las_path, output_las_path, target_point_format=3):
    """
    Converts a LAS/LAZ file to LAS 1.2 while preserving offsets and scales.
    
    Parameters:
        input_las_path (str): Path to the input LAS/LAZ file.
        output_las_path (str): Path to save the converted LAS 1.2 file.
        target_point_format (int): The point format for LAS 1.2 (default is 3).
    """
    # Read the original LAS file
    las = laspy.read(input_las_path)

    # Create a new header for LAS 1.2 with the correct point format
    header = laspy.LasHeader(version="1.2", point_format=target_point_format)

    # Preserve offsets and scales
    header.offsets = las.header.offsets
    header.scales = las.header.scales
    header.min = las.header.min
    header.max = las.header.max

    # Create a new LAS file with compatible format
    new_las = laspy.LasData(header)

    # Copy only the compatible fields (XYZ, intensity, classification, etc.)
    for dim in new_las.point_format.dimension_names:
        if dim in las.point_format.dimension_names:
            new_las[dim] = las[dim]

    # Save the new LAS file
    new_las.write(output_las_path)

    print(f"Saved as {output_las_path} with LAS 1.2 and Point Format {target_point_format}")
    print(f"Preserved offsets: {header.offsets}")
    print(f"Start point: {new_las.x[0]}, {new_las.y[0]}, {new_las.z[0]}")


#output_file = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-B/out_v12.las"
#output_file = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-A/240725_093813_v12.las"

#convert_las_to_12(f1,output_file)


# Open the .las file
las_file_path = f1
las = laspy.read(las_file_path)



# Print installed laspy version
import pkg_resources
laspy_version = pkg_resources.get_distribution("laspy").version
print(f"Installed laspy version: {laspy_version}")


las_version = f"{las.header.version.major}.{las.header.version.minor}"
print(f"LAS version of the point cloud: {las_version}")

np.set_printoptions(suppress=True)
_offset = las.header.offsets  # Index 2 corresponds to Z
print('header._offset ', _offset)

# Extract point coordinates (X, Y, Z)
points = np.vstack((las.x, las.y, las.z)).transpose()

points = points[::100]

start_point = points[0]
print('start_point ', start_point)
#points -= start_point

# Compute Euclidean distance from the start point
#distances = np.linalg.norm(points, axis=1)

# Filter points within 50 meters
#points = points[distances <= 50]
#print('filtered points ', np.shape(points))

# Display the number of points
num_points = len(las.points)
print(f"Number of points: {num_points}")

#original_als_cloud:93169613

print("Point fields (attributes):")
for dimension in las.point_format.dimensions:
    print(dimension.name)

def gps_to_unix(gps_time):
    gps_epoch_diff = 315964800  # Difference between GPS epoch (1980) and Unix epoch (1970)
    leap_seconds = 18           # Number of leap seconds as of 2023

    # Convert GPS time to Unix time
    unix_time = gps_time + gps_epoch_diff - leap_seconds
    return unix_time

# Print the first few rows of data

SECONDS_PER_WEEK = 604800

num_rows_to_print = 2  # Number of rows to print
for i in range(num_rows_to_print):
    print(f"Point {i + 1}:")
    print(f"  X: {las.x[i]}")
    print(f"  Y: {las.y[i]}")
    print(f"  Z: {las.z[i]}")
    #print(f"  Intensity: {las.intensity[i]}")
    print(f"  GPS Time: {las.gps_time[i]}")

    unix_time = gps_to_unix(las.gps_time[i])
    print(f" unix_time: {unix_time}")

    human_readable = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"Human-Readable Date: {human_readable}")

    gps_time = las.gps_time[i]
    number_of_weeks = gps_time // SECONDS_PER_WEEK
    tow = gps_time % SECONDS_PER_WEEK
    print('number_of_weeks:',number_of_weeks, ', tow:',tow)


    print("-" * 30)

collection_date = datetime(2024, 7, 25, 0, 0, 0)
expected_unix_time = collection_date.timestamp()
print(f"Expected Unix Time for July 25, 2024: {expected_unix_time}")


if False:
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Optional: Color the points (if color information is available in the .las file)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # Normalize RGB values to the range [0, 1]
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Visualization")


