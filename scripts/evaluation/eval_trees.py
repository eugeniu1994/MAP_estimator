import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches
import seaborn as sns

font = 18


plt.rcParams.update({'font.size': font})
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.4)  # 1.6 × base font size (~10 by default)
font = 16

plt.rcParams.update({'font.size': font})
plt.rc('axes', titlesize=font)     # Set the font size of the title
#plt.rc('axes', labelsize=font)     # Set the font size of the x and y labels
#plt.rc('xtick', labelsize=12)    # Set the font size of the x tick labels
#plt.rc('ytick', labelsize=12)    # Set the font size of the y tick labels
plt.rc('legend', fontsize=14)    # Set the font size of the legend
plt.rc('font', size=font)          # Set the general font size'''


MLS_all_trees_xyz_path = "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/trees/Robust MLS + Dense ALS + Map Fusion + GNSS/MLS_all_trees_xyz.txt"
ALS_all_trees_xyz_path = "/media/eugeniu/T7/a_georeferenced_vux_tests/merged/trees/ALS_experimental/ALS_all_trees_xyz.txt"

from scipy import stats
from sklearn.utils import resample

# Function to calculate confidence intervals
def calculate_confidence_interval(mean, std, n, alpha=0.05):
    # t-critical value for 95% confidence (two-tailed test)
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
    # Standard error
    std_error = std / np.sqrt(n)
        
    # Margin of error
    margin_of_error = t_critical * std_error
        
    # Confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
        
    return lower_bound, upper_bound
    
def bootstrap_ci(data, n_resamples=1000, ci=95):
    means = []
    for _ in range(n_resamples):
        sample = resample(data)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


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

# print("ALS to MLS:\n", T_als2mls)
# print("MLS to ALS:\n", T_mls2als)


def apply_transformation(pcd, T):
    T = np.array(T, dtype=np.float64)
    
    #points = np.asarray(pcd, dtype=np.float64)
    points = pcd[:,:3].dot(T[:3, :3].T) + T[:3, 3]

    return points

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
    rv = rv[(rv[:,1] < 670), :]

    return rv[:,:2] #x y
    #return rv[:,:3] #x y height

def AssociateTrees(query_trees, reference_trees, dist_threshold = 1, title = '', plot = False):
    tree = cKDTree(reference_trees) # Build KD-tree from reference points

    distances, indices = tree.query(query_trees, k=1)

    # Mask of matches within 1 meter
    within_1m = distances < dist_threshold
    matched_indices = indices[within_1m]
    print('matched_indices:', np.shape(matched_indices),' with threshold of ',dist_threshold)

    #unique_matched_ref_points = reference_trees[np.unique(matched_indices)]

    
    colors = np.where(distances < dist_threshold, 'red', 'black')
    #alphas = np.full(query_trees.shape[0], 0.1)  # Default: faded out
    #alphas[distances < dist_threshold] = 1.0    # Highlight matched ones
    alphas = np.full(query_trees.shape[0], 1)  

    if plot:
        plt.figure(figsize=(8, 8))
        for i, point in enumerate(query_trees):
            plt.scatter(point[0], point[1], color=colors[i], alpha=alphas[i], label='VUX points', s=10)
        #plt.scatter(query_trees[:, 0], query_trees[:, 1], c=colors, s=10, label='VUX points')
        plt.scatter(reference_trees[:, 0], reference_trees[:, 1], c='green', s=60, alpha=0.3, label='Reference Points')
        #plt.scatter(unique_matched_ref_points[:, 0], unique_matched_ref_points[:, 1], c='blue', s=60, alpha=0.3, label='Reference trees')

        #plt.legend()

        # Add legend manually
        ref_handle = plt.Line2D([0], [0], marker='o', color='w', label='Reference Trees',
                                markerfacecolor='green', alpha=0.3, markersize=10)
        query_handle = plt.Line2D([0], [0], marker='o', color='w', label=title +' Query Trees',
                                markerfacecolor='red', markersize=8)

        plt.legend(handles=[ref_handle, query_handle], loc='best')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Nearest Neighbor Check (< {}m → red, ≥ {}m → black)'.format(dist_threshold,dist_threshold))
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

    return query_matched, ref_matched, final_ref_indices

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

def summarize(errors, name):
    print(f"\n{name}")
    for model, errs in errors.items():
        mean = np.mean(errs)
        median = np.median(errs)
        rmse = np.sqrt(mean_squared_error(np.zeros_like(errs), errs))
        std = np.std(errs)

        #ci = calculate_confidence_interval(mean, std, len(errs), alpha = 0.05)
        #print('\n \n prev ci:',ci)
        print('len errors:', len(errs))
        ci = bootstrap_ci(errs, n_resamples=00, ci=95)
        ci_mean = (mean - ci[0])

        print(f"{model}: Mean={mean:.4f}, Median={median:.4f}, RMSE={rmse:.4f}, Std={std:.4f}, ci={ci}, , ci_mean={ci_mean}")


ref_trees = "/media/eugeniu/T7/las_georeferenced/Ref_ALS_Hesai_fused/results/Results/Stem_curves/Found_stem_tree_attributes.txt"

ref_tree_data = read_query_tree_file(ref_trees)  
print('\nref_tree_data ', np.shape(ref_tree_data))
#print('ref_tree_data:\n', ref_tree_data[:3])

#np.savetxt('/media/eugeniu/T7/las_georeferenced/Ref_ALS_Hesai_fused/results/Results/Stem_curves/xyz_hesai0_estimated.txt', ref_tree_data, fmt='%.6f', delimiter=',')


#convert to ALS frame 
#ref_tree_data = apply_transformation(ref_tree_data, T_mls2als)
#print('ref_tree_data als :\n', ref_tree_data[:3])

methods = {
    'GNSS-IMU 0': '/media/eugeniu/T7/las_georeferenced/big_cov_init/gnss0/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'GNSS-IMU 1': '/media/eugeniu/T7/las_georeferenced/big_cov_init/gnss1/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'GNSS-IMU 2': '/media/eugeniu/T7/las_georeferenced/big_cov_init/gnss2/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'GNSS-IMU 3': '/media/eugeniu/T7/las_georeferenced/big_cov_init/gnss3/results/Results/Stem_curves/Found_stem_tree_attributes.txt',

    'Hesai 0': '/media/eugeniu/T7/las_georeferenced/big_cov_init/hesai0/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'Hesai 1': '/media/eugeniu/T7/las_georeferenced/big_cov_init/hesai1/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'Hesai 2': '/media/eugeniu/T7/las_georeferenced/big_cov_init/hesai2/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
    'Hesai 3': '/media/eugeniu/T7/las_georeferenced/big_cov_init/hesai3/results/Results/Stem_curves/Found_stem_tree_attributes.txt',
}

lab = ['A','B','C','D','E','F','G','H']

def plot_trees(tree, ref_tree, label, c='green', s=60, marker='o', alpha=0.5):
    plt.figure(figsize=(10, 9))
    
    plt.scatter(ref_tree[:, 0], ref_tree[:, 1], c='green', s=60, marker='o', alpha=0.5)
    plt.scatter(tree[:, 0], tree[:, 1], c=c, label=label, s=s, marker=marker, alpha=alpha)

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Tree Locations')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.draw()

plot = False 

matches = {}

for label, folder in methods.items():
    model_trees = read_query_tree_file(folder)

    print('{} - model_trees{}'.format(label, np.shape(model_trees)))

    #if plot:
    #    plot_trees(model_trees, ref_tree_data, label='{} tree'.format(label), c='blue', s=35, marker='*', alpha=1)

    #model_trees, ref_tree_data = model_trees[:,:2], ref_tree_data[:,:2]
    query_trees, ref_trees, ref_indices = AssociateTrees(model_trees, ref_tree_data, dist_threshold = 1, title = label, plot = plot)
    
    #if plot:
    #    plot_trees(query_trees, ref_trees, label='After DA {} tree'.format(label), c='blue', s=35, marker='*', alpha=1)

    matches[label] = {
        'matched_query': query_trees,
        'matched_ref': ref_trees,
        'ref_indices': ref_indices
    }

all_ref_sets = [set(match['ref_indices']) for match in matches.values()]
common_ref_indices = sorted(set.intersection(*all_ref_sets))
print('common_ref_indices:', np.shape(common_ref_indices))

aligned_matches = {}

for label, data in matches.items():
    ref_idx_array = np.array(data['ref_indices'])
    mask = np.isin(ref_idx_array, common_ref_indices)

    matched_query, matched_ref = data['matched_query'][mask], data['matched_ref'][mask]
    print("method: ",label,", matched_query:",np.shape(matched_query), ", matched_ref:",np.shape(matched_ref) )
    
    #plot_trees(query_trees, ref_trees, label='After DA {} tree'.format(label), c='blue', s=35, marker='*', alpha=1)
    
    # Compute errors
    abs_errors_dict[label] = compute_absolute_error(matched_query, matched_ref)
    rel_errors_dict[label] = compute_relative_error(matched_query, matched_ref)

    # if label == 'Hesai 0':
    #     np.savetxt('/media/eugeniu/T7/las_georeferenced/Ref_ALS_Hesai_fused/results/Results/Stem_curves/xyz_ref.txt', matched_ref, fmt='%.6f', delimiter=',')
    #     np.savetxt('/media/eugeniu/T7/las_georeferenced/Ref_ALS_Hesai_fused/results/Results/Stem_curves/xyz_model_hesai0.txt', matched_query, fmt='%.6f', delimiter=',')


summarize(abs_errors_dict, "Absolute Errors")
summarize(rel_errors_dict, "Relative Errors")

add_first_legend = True
# Boxplot plotting
def plot_boxplots(data, metric_name, colors, add_first_legend = True):
    plt.figure(figsize=(10, 6))
    labels = list(data.keys())
    values = [data[label] for label in labels]

    box = plt.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False)

    legend_handles = []
    ind = 0
    for patch, color, label in zip(box['boxes'], colors, labels):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        legend_handles.append(mpatches.Patch(color=color, label=lab[ind] + " : " + label))
        ind += 1

    for median_line in box['medians']:
        median_line.set_alpha(0)

    for mean_line in box['means']:
        mean_line.set(color='black', linewidth=2, linestyle='--')

    for line in box['whiskers'] + box['caps']:
        line.set(color='black', linewidth=1.2)

    plt.ylabel(metric_name)
    if add_first_legend:
        plt.legend(handles=legend_handles, title="Model", loc='best')
        add_first_legend = False
    plt.grid(False)
    plt.tight_layout()
    #plt.xticks([])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], lab) 
    plt.draw()

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:brown', 'skyblue', 'lightgoldenrodyellow', 'lightblue', 'lightgreen', 'lightcoral', 'lightblue', 'lightgreen', 'lightcoral' ][:len(methods)]

# Plotting
plot_boxplots(abs_errors_dict, "Absolute Error (m)", colors, True)
plot_boxplots(rel_errors_dict, "Relative Error (m)", colors, False)


plt.show()