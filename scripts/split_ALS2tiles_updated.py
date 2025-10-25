import os
import numpy as np
import glob
import laspy
from collections import defaultdict

# tile_size = 50  # Size of each tile in meters

# output_directory = '/media/eugeniu/T7/Evo_drone24_from_Jesse_cropped2' 
# input_laz_directory = '/media/eugeniu/T7/Evo_drone24_from_Jesse' 

# Area of interest as a square
# east_min, east_max = 397925.961 , 399682.961
# north_min, north_max = 6785448.985, 6786228.985

# east_min, east_max = 398108., 400183
# north_min, north_max = 6785510, 6786381

tile_size = 50  # Size of each tile in meters

output_directory = '/media/eugeniu/T7/Maanmittauslaitos_Tiedostopalvelu_REST-20250716T094416321466852/data_las_cropped' 
input_laz_directory = '/media/eugeniu/T7/Maanmittauslaitos_Tiedostopalvelu_REST-20250716T094416321466852/data_laz' 

east_min, east_max = 397123., 402000.
north_min, north_max = 6784218, 6789000

output_directory = "/media/eugeniu/T7/Lowcost_trajs/MML_5pt/CALIBRATE_2023/data_las_cropped"
input_laz_directory = "/media/eugeniu/T7/Lowcost_trajs/MML_5pt/CALIBRATE_2023"
east_min, east_max = -99999999999, 99999999999.
north_min, north_max = -99999999999, 99999999999.

def get_tile_key(x, y, tile_size, x_min, y_min):
    tile_x = int((x - x_min) // tile_size)
    tile_y = int((y - y_min) // tile_size)
    return tile_x, tile_y

def process_laz_files_incremental(input_folder, output_folder, tile_size, east_min, east_max, north_min, north_max):

    laz_files = glob.glob(os.path.join(input_folder, '**/*.laz'), recursive=True)
    print(f'Found {len(laz_files)} .laz files.')

    for laz_file in laz_files:
        print(f'\nProcessing: {laz_file}')
        point_cloud = laspy.read(laz_file)

        x, y, z = point_cloud.x, point_cloud.y, point_cloud.z
        mask = (x >= east_min) & (x <= east_max) & (y >= north_min) & (y <= north_max)

        if not np.any(mask):
            continue

        print(f'Points in AOI: {np.count_nonzero(mask)}')

        filtered_points = point_cloud.points[mask]
        filtered_x = x[mask]
        filtered_y = y[mask]

        tile_dict = defaultdict(list)
        for i in range(len(filtered_points)):
            tile_x, tile_y = get_tile_key(filtered_x[i], filtered_y[i], tile_size, east_min, north_min)
            tile_dict[(tile_x, tile_y)].append(i)

        for (tile_x, tile_y), indices in tile_dict.items():
            
            x_start = east_min + tile_x * tile_size
            y_start = north_min + tile_y * tile_size

            tile_filename = f'tile_x_{int(x_start)}_y_{int(y_start)}.las'
            tile_path = os.path.join(output_folder, tile_filename)

            # Create tile point cloud
            header = laspy.LasHeader(version="1.2", point_format=point_cloud.header.point_format)
            header.offsets = point_cloud.header.offsets
            header.scales = point_cloud.header.scales

            tile_pcloud = laspy.LasData(header)
            tile_pcloud.points = filtered_points[indices]

            if os.path.exists(tile_path):
                existing = laspy.read(tile_path)
    
                tile_x = tile_pcloud.x
                tile_y = tile_pcloud.y
                tile_z = tile_pcloud.z
                
                existing_x = existing.x
                existing_y = existing.y
                existing_z = existing.z

                # shapes of the arrays (x, y, z) to confirm they exist
                # print(f'tile_pcloud.x shape: {tile_x.shape}')
                # print(f'existing.x shape: {existing_x.shape}')
                
                # Concatenate the points arrays (x, y, z)
                combined_x = np.concatenate((existing_x, tile_x))
                combined_y = np.concatenate((existing_y, tile_y))
                combined_z = np.concatenate((existing_z, tile_z))
                # print(f'combined_z shape: {combined_z.shape}')
                # Create a new combined points array
                # combined_points = np.core.records.fromarrays([combined_x, combined_y, combined_z],
                #                                             names='x, y, z', formats='f4, f4, f4')
                
                # print('combined_points ',np.shape(combined_points))
                # tile_pcloud.points = combined_points
                tile_pcloud.x = combined_x
                tile_pcloud.y = combined_y
                tile_pcloud.z = combined_z

                tile_pcloud.write(tile_path)
                print(f'Appended to: {tile_path}')

            else:
                tile_pcloud.write(tile_path)
                print(f'Created tile: {tile_path}')

    print('\nAll tiles saved!')

process_laz_files_incremental(input_laz_directory, output_directory, tile_size, east_min, east_max, north_min, north_max)