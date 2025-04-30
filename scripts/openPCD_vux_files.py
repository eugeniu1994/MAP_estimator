
#pip install open3d


try:
    import open3d as o3d
except:
    print('No open3d')

import numpy as np
from pathlib import Path
import argparse
import re

def natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.stem)]

def save_as_txt(pcd, output_path):
    points = np.asarray(pcd)
    np.savetxt(output_path, points, fmt="%.10f", delimiter=",")

def apply_transformation(pcd, T):
    T = np.array(T, dtype=np.float64)
    
    points = np.asarray(pcd, dtype=np.float64)
    points[:,:3] = points[:,:3].dot(T[:3, :3].T) + T[:3, 3]

    return points

def load_and_stack_full_fields(txt_paths):
    all_data = []

    for path in txt_paths:
        data = np.loadtxt(str(path))  # assumes 5 columns: x y z range time
        if data.ndim == 1:
            data = data[np.newaxis, :]  # ensure 2D if only one point in file
        all_data.append(data)

    if not all_data:
        print("No data loaded.")
        return np.empty((0, 5))  # empty 2D array with 5 columns

    stacked_data = np.vstack(all_data)

    print(f"\nStacked {stacked_data.shape[0]} points with shape {stacked_data.shape} (x, y, z, range, time)")
    return stacked_data

def read_se3_and_inverse_from_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    matrices = {}
    current_key = None
    current_matrix = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith("T_"):
            if current_key and current_matrix:
                matrices[current_key] = np.array(current_matrix, dtype=float)
                current_matrix = []
            current_key = line
        else:
            current_matrix.append([float(x) for x in line.split()])
    
    if current_key and current_matrix:
        matrices[current_key] = np.array(current_matrix, dtype=float)

    return matrices["T_als2mls"], matrices["T_mls2als"]

def main(input_dir, output_dir, local_global_T, group_size = 100, visualize=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_pcds = sorted(input_path.glob("*.txt"), key=natural_sort_key)

    total_files = len(all_pcds)
    print(f"Found {total_files} .pcd files.")

    T_als2mls, T_mls2als = read_se3_and_inverse_from_txt(local_global_T)

    print("ALS to MLS:\n", T_als2mls)
    print("MLS to ALS:\n", T_mls2als)
    
    transform_mls_to_als_frame = True

    for i in range(0, total_files, group_size):
        group_files = all_pcds[i:i + group_size]
        if not group_files:
            continue

        print(f"Merging files {i+1} to {i+len(group_files)}...")
        merged_pcd = load_and_stack_full_fields(group_files)  # now they are in the mls frame

        if transform_mls_to_als_frame:
            if T_mls2als is not None:
                merged_pcd = apply_transformation(merged_pcd, T_mls2als)
                print("Transformation from MLS to ALS applied to the merged point cloud.")

        output_file = output_path / f"merged_{i//group_size:03d}.txt"
        save_as_txt(merged_pcd, str(output_file))
        
        print(f"Saved: {output_file}")

        if visualize:
            print("Visualizing...")
            try:
                merged_pcd_o3d = o3d.geometry.PointCloud()
                merged_pcd_o3d.points = o3d.utility.Vector3dVector(merged_pcd)
                o3d.visualization.draw_geometries([merged_pcd_o3d])
            except KeyboardInterrupt:
                print("\nVisualization interrupted. Closing window...")


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Merge .pcd files into chunks of 100.")
    parser.add_argument("--input_dir", default="/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu1/", help="Folder containing .pcd files")
    parser.add_argument("--output_dir", default="/home/eugeniu/vux-georeferenced/merged/", help="Folder to save merged .pcd files")
    parser.add_argument("--local_global_T", default="/home/eugeniu/vux-georeferenced/als2mls_dense.txt", help="File with mls to als transform")
    parser.add_argument("--visualize", action="store_false", help="Visualize merged point clouds")
    parser.set_defaults(visualize=False)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.local_global_T, 300, args.visualize)

