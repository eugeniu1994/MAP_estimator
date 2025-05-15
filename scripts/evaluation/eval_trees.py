import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches


known_als2mls_inv: 398269.1135182129 6786162.7477222066     131.0252137406
als_to_mls_inv   : 398269.2192482995 6786162.3634432964     132.7766588416




MLS_all_trees_xyz_path = "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/trees/Robust MLS + Dense ALS + Map Fusion + GNSS/MLS_all_trees_xyz.txt"
ALS_all_trees_xyz_path = "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/trees/Robust MLS + Dense ALS + Map Fusion + GNSS/ALS_all_trees_xyz.txt"

ALS_all_trees_xyz_path = "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/trees/ALS_experimental/ALS_all_trees_xyz.txt"

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


T_als2mls, T_mls2als = read_se3_and_inverse_from_txt("/home/eugeniu/vux-georeferenced/als2mls_dense.txt")

print("ALS to MLS:\n", T_als2mls)
print("MLS to ALS:\n", T_mls2als)

def apply_transformation(pcd, T):
    T = np.array(T, dtype=np.float64)
    
    points = np.asarray(pcd, dtype=np.float64)
    #points[:,:3] = points[:,:3].dot(T[:3, :3].T) + T[:3, 3]

    return points

def reference_trees():
    # Load point data
    car_points = np.loadtxt(MLS_all_trees_xyz_path, delimiter=',')    # shape: (N, 3)
    drone_points = np.loadtxt(ALS_all_trees_xyz_path, delimiter=',')  # shape: (M, 3)

    print('car_points:', np.shape(car_points),' drone_points:',np.shape(drone_points))

    valid_rows_car = car_points[:,0] < 398600
    valid_rows_drone = drone_points[:,0] < 398600

    # x and y coordinates
    # car_xy = car_points[valid_rows_car, :2]
    # drone_xy = drone_points[valid_rows_drone, :2]

    car_xy = car_points[valid_rows_car]
    drone_xy = drone_points[valid_rows_drone]
    car_xy[:,2] -= car_xy[0,2]
    drone_xy[:,2] -= drone_xy[0,2]

    print('Filtered car_points:', np.shape(car_xy),' drone_points:',np.shape(drone_xy))

    # plt.figure(figsize=(10, 9))
    # plt.scatter(car_xy[:, 0], car_xy[:, 1], c='green', label='Car-based Detection', s=50, marker='o', alpha=0.5)
    # plt.scatter(drone_xy[:, 0], drone_xy[:, 1], c='blue', label='Drone-based Detection', s=25, marker='*', alpha=1)

    # plt.xlabel('X (meters)')
    # plt.ylabel('Y (meters)')
    # plt.title('Tree Locations Detected by Car and Drone Platforms')
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.draw()

    return car_xy, drone_xy

def AssociateTrees(query_trees, reference_trees, dist_threshold = 1, title = '', plot = False):
    # Build KD-tree from reference points
    tree = cKDTree(reference_trees)

    distances, indices = tree.query(query_trees, k=1)

    # Mask of matches within 1 meter
    within_1m = distances < dist_threshold
    matched_indices = indices[within_1m]
    print('matched_indices:', np.shape(matched_indices),' with threshold of ',dist_threshold)

    #unique_matched_ref_points = reference_trees[np.unique(matched_indices)]

    
    colors = np.where(distances < dist_threshold, 'red', 'black')
    alphas = np.full(query_trees.shape[0], 0.1)  # Default: faded out
    alphas[distances < dist_threshold] = 1.0    # Highlight matched ones

    if plot:
        plt.figure(figsize=(8, 8))
        for i, point in enumerate(query_trees):
            plt.scatter(point[0], point[1], color=colors[i], alpha=alphas[i], label='VUX points', s=10)
        #plt.scatter(query_trees[:, 0], query_trees[:, 1], c=colors, s=10, label='VUX points')
        plt.scatter(reference_trees[:, 0], reference_trees[:, 1], c='blue', s=60, alpha=0.3, label='Reference Points')
        #plt.scatter(unique_matched_ref_points[:, 0], unique_matched_ref_points[:, 1], c='blue', s=60, alpha=0.3, label='Reference trees')

        #plt.legend()

        # Add legend manually
        ref_handle = plt.Line2D([0], [0], marker='o', color='w', label='Reference Trees',
                                markerfacecolor='blue', alpha=0.3, markersize=10)
        query_handle = plt.Line2D([0], [0], marker='o', color='w', label=title +' Query Trees',
                                markerfacecolor='red', markersize=8)

        plt.legend(handles=[ref_handle, query_handle], loc='best')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Nearest Neighbor Check (< 1m → red, ≥ 1m → black)')
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.draw()  

    #plt.show()
    # Build a map from reference index to its closest query index
    ref_to_best_query = {}
    for query_idx, (ref_idx, dist) in enumerate(zip(indices, distances)):
        if dist >= dist_threshold:
            continue
        if ref_idx not in ref_to_best_query or dist < ref_to_best_query[ref_idx][1]:
            ref_to_best_query[ref_idx] = (query_idx, dist)

    # Final matched query and reference trees (1-to-1, closest only)
    final_query_indices = [v[0] for v in ref_to_best_query.values()]
    final_ref_indices = list(ref_to_best_query.keys())

    query_matched = query_trees[final_query_indices]
    ref_matched = reference_trees[final_ref_indices]

    return query_matched, ref_matched
    #------------------------------------------------
    

car_xy, drone_xy = reference_trees()

def read_query_tree_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip  (first 2 lines)
    for line in lines[2:]:
        if not line.strip():
            continue  
        parts = line.strip().split()
        try:
            x = float(parts[2])
            y = float(parts[3])
            dbh_cm = float(parts[4])
            dbh_m = dbh_cm / 100.0  # convert to meters
            #data.append([x, y, dbh_m])

            Lowest_height_tree = float(parts[5])
            data.append([x, y, Lowest_height_tree])

        except (IndexError, ValueError):
            continue  # skip lines that can't be parsed
    
    rv = np.array(data)
    print('rv ', np.shape(rv))
    rv[:,2] -= rv[0,2]

    return rv

vux_trees = "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_1/results/Results/Stem_curves/Found_stem_tree_attributes.txt"

tree_data = read_query_tree_file(vux_trees)  
print('tree_data ', np.shape(tree_data))

#np.savetxt("/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_1/point_cloud.txt", tree_data, fmt="%.6f")

#convert to MLS frame 
apply_transformation(tree_data, T_als2mls)

#convert to MLS frame 
apply_transformation(car_xy, T_als2mls)
apply_transformation(drone_xy, T_als2mls)

d_trees = tree_data #[:,:2]

def plot_car_drone_vux(car_xy, drone_xy, d_trees):
    plt.figure(figsize=(10, 9))
    plt.scatter(car_xy[:, 0], car_xy[:, 1], c='green', label='Car-based Detection', s=60, marker='o', alpha=0.5)
    plt.scatter(drone_xy[:, 0], drone_xy[:, 1], c='blue', label='Drone-based Detection', s=35, marker='*', alpha=1)

    plt.scatter(d_trees[:, 0], d_trees[:, 1], c='red', label='VUX-based Detection', s=10, alpha=1)

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Tree Locations Detected by Car and Drone Platforms')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.draw()

plot_car_drone_vux(car_xy, drone_xy, d_trees)

query_trees,ref_trees = AssociateTrees(d_trees, car_xy, dist_threshold = 1)
#AssociateTrees(d_trees, drone_xy, dist_threshold = 1)

plt.draw()
plt.show()

def align_3d_points(source, target):
    # Step 1: Compute the centroids of both arrays
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    # Step 2: Center the points around the centroids
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Step 3: Compute the covariance matrix
    H = np.dot(source_centered.T, target_centered)

    # Step 4: Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Step 6: Check for reflection and correct it
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Step 7: Compute the translation vector
    t = centroid_target - np.dot(R, centroid_source)

    # Step 8: Apply the rotation and translation to align the source points
    aligned_source = np.dot(source_centered, R) + centroid_target

    return aligned_source, R, t

#--------------------------------------------------------------------------------------

methods = {
    'GNSS-IMU 0': '/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_0/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'GNSS-IMU 1': "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_2/results/Results/Stem_curves/Found_stem_tree_attributes.txt",
    'GNSS-IMU 2': "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_4/results/Results/Stem_curves/Found_stem_tree_attributes.txt",
    'GNSS-IMU 3': "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_6/results/Results/Stem_curves/Found_stem_tree_attributes.txt",

    'Hesai 0': '/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_1/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'Hesai 1': "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_3/results/Results/Stem_curves/Found_stem_tree_attributes.txt",
    'Hesai 2': "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_5/results/Results/Stem_curves/Found_stem_tree_attributes.txt",
    'Hesai 3': "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/config_7/results/Results/Stem_curves/Found_stem_tree_attributes.txt",
}

models = {
    # "Model A": (query_A, reference),  # shape (N,2), (N,2)
}

dist_threshold =  1
for label, folder in methods.items():
    model_trees = read_query_tree_file(folder)[:,:3] #x y height

    #convert to MLS frame 
    apply_transformation(model_trees, T_als2mls)

    print('\n{} loaded {} trees'.format(label, np.shape(model_trees)))

    query_trees,ref_trees = AssociateTrees(model_trees[:,:2], car_xy[:,:2], dist_threshold = dist_threshold, title=label, plot=True)

    #query_trees, R, t = align_3d_points(query_trees, ref_trees)

    print('query_trees:{}, ref_trees:{}'.format(np.shape(query_trees), np.shape(ref_trees)))

    #query_trees,ref_trees = AssociateTrees(query_trees,ref_trees, dist_threshold = 1, title=label, plot=True)
    #query_trees, R, t = align_3d_points(query_trees, ref_trees)
    #print('query_trees:{}, ref_trees:{}'.format(np.shape(query_trees), np.shape(ref_trees)))

    models[label] = (query_trees, ref_trees)

#plt.show()

# Containers for errors
abs_errors_dict = {}
rel_errors_dict = {}

# Error computation functions
def compute_absolute_error(query, reference):
    return np.linalg.norm(query - reference, axis=1)

def compute_relative_error(query, reference):
    N = query.shape[0]
    ref_distances = []
    est_distances = []

    for i in range(N):
        for j in range(i + 1, N):
            ref_d = np.linalg.norm(reference[i] - reference[j])
            est_d = np.linalg.norm(query[i] - query[j])
            ref_distances.append(ref_d)
            est_distances.append(est_d)

    return np.abs(np.array(est_distances) - np.array(ref_distances))

# Compute errors
for name, (query, reference) in models.items():
    abs_errors_dict[name] = compute_absolute_error(query, reference)
    rel_errors_dict[name] = compute_relative_error(query, reference)

# Summary printing
def summarize(errors, name):
    print(f"\n{name}")
    for model, errs in errors.items():
        mean = np.mean(errs)
        median = np.median(errs)
        rmse = np.sqrt(mean_squared_error(np.zeros_like(errs), errs))
        std = np.std(errs)
        print(f"{model}: Mean={mean:.4f}, Median={median:.4f}, RMSE={rmse:.4f}, Std={std:.4f}")

summarize(abs_errors_dict, "Absolute Errors")
summarize(rel_errors_dict, "Relative Errors")

# Boxplot plotting
def plot_boxplots(data, metric_name, colors):
    plt.figure(figsize=(10, 6))
    labels = list(data.keys())
    values = [data[label] for label in labels]

    box = plt.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False)

    legend_handles = []
    for patch, color, label in zip(box['boxes'], colors, labels):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        legend_handles.append(mpatches.Patch(color=color, label=label))

    for median_line in box['medians']:
        median_line.set_alpha(0)

    for mean_line in box['means']:
        mean_line.set(color='black', linewidth=2, linestyle='--')

    for line in box['whiskers'] + box['caps']:
        line.set(color='black', linewidth=1.2)

    plt.ylabel(metric_name)
    plt.legend(handles=legend_handles, title="Model", loc='best')
    plt.grid(False)
    plt.tight_layout()
    #plt.xticks([])
    plt.draw()

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:brown', 'skyblue', 'lightgoldenrodyellow', 'lightblue', 'lightgreen', 'lightcoral', 'lightblue', 'lightgreen', 'lightcoral' ][:len(models)]

# Plotting
plot_boxplots(abs_errors_dict, "Absolute Error (m)", colors)
plot_boxplots(rel_errors_dict, "Relative Error (m)", colors)

plt.show()