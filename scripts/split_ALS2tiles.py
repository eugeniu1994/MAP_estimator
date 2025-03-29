
import os
import numpy as np
import glob
import laspy


def split_tile(point_cloud, tile_size=50.0, target_point_format=3):
    """Split the point cloud into smaller tiles."""
    # Extract the points and their x, y coordinates
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).T
        
    print('points ', np.shape(points))
    
    x_min, x_max = point_cloud.x.min(), point_cloud.x.max()
    y_min, y_max = point_cloud.y.min(), point_cloud.y.max()
    print('x_min:{}, x_max:{}, y_min:{}, y_max:{}'.format(x_min,x_max, y_min,y_max))
    
    # Determine the number of smaller tiles
    x_steps = int(np.ceil((x_max - x_min) / tile_size))
    y_steps = int(np.ceil((y_max - y_min) / tile_size))
    print('x_steps:{}, y_steps:{}'.format(x_steps,y_steps))
    tiles = []

    for i in range(x_steps):
        for j in range(y_steps):
            x_start = x_min + i * tile_size
            x_end = x_start + tile_size
            y_start = y_min + j * tile_size
            y_end = y_start + tile_size
            
            # Filter points within the current tile
            mask = (points[:, 0] >= x_start) & (points[:, 0] < x_end) & \
                   (points[:, 1] >= y_start) & (points[:, 1] < y_end)
            
            if mask.any():
                tile_points = points[mask]
                print('tile_points:', np.shape(tile_points))

                header = laspy.LasHeader(version="1.2", point_format=target_point_format)
                header.offsets = point_cloud.header.offsets
                header.scales = point_cloud.header.scales

                # Create a new LAS file with compatible format
                tile_pcloud = laspy.LasData(header)
                tile_pcloud.x = tile_points[:, 0]
                tile_pcloud.y = tile_points[:, 1]
                tile_pcloud.z = tile_points[:, 2]
                
                tiles.append((tile_pcloud, x_start, y_start))
    
    return tiles

def process_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.laz'):
            input_path = os.path.join(input_folder, filename)
            output_base = os.path.splitext(filename)[0]
            print('\ninput_path ',input_path)
            print('output_base ',output_base,'\n')
            
            pcloud = laspy.read(input_path)
            print('pcloud:', np.shape(pcloud.x))

            # Split the point cloud into smaller tiles
            tiles = split_tile(pcloud)
            
            # Save each tile as a .las file
            for idx, (tile_pcloud, x_start, y_start) in enumerate(tiles):
                tile_filename = f'tile_x_{int(x_start)}_y_{int(y_start)}.las'

                tile_path = os.path.join(output_folder, tile_filename)
                
                tile_pcloud.write(tile_path)
                print(f'Saved {tile_path}')

tile_size = 100  # Size of each tile in meters

output_directory = "/media/eugeniu/T7 Shield/Masalantie_data_ALS_split_las"
input_laz_directory = '/media/eugeniu/T7 Shield/Masalantie_data' 
laz_files = glob.glob(os.path.join(input_laz_directory, '**/*.laz'), recursive=True)

print('there are {} laz_files'.format(len(laz_files)))

#uncomment this 
#process_folder(input_laz_directory, output_directory)
print('Processing complete!')