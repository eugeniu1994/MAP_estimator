import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import matplotlib.patches as mpatches

colors = ['lightblue', 'lightgoldenrodyellow', 'skyblue', 'lightcoral', 'lightgreen', 'khaki', 'plum', 'lightgray', 'orange', 'skyblue', 'lightcoral']
colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:brown', 'skyblue', 'lightgoldenrodyellow', ]


plt.rcParams.update({'font.size': 16})
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.3)  # 1.6 × base font size (~10 by default)

# p2plane error, furtherst_d, closest_d, curvature, neighbours in a radius ball 
methods = {
    'GNSS-IMU 0': '/home/eugeniu/vux-georeferenced/No_refinement/gnss-imu0/surface-eval',
    'GNSS-IMU 1': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu1/surface-eval",
    'GNSS-IMU 2': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu2/surface-eval",
    'GNSS-IMU 3': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu3/surface-eval",

    'Hesai 0': '/home/eugeniu/vux-georeferenced/No_refinement/hesai0/surface-eval',
    'Hesai 1': "/home/eugeniu/vux-georeferenced/BA-2_iterations/hesai1/surface-eval",
    'Hesai 2': "/home/eugeniu/vux-georeferenced/BA-2_iterations/hesai2/surface-eval",
    'Hesai 3': "/home/eugeniu/vux-georeferenced/BA-2_iterations/hesai3/surface-eval",
    



    
    # 'GNSS-IMU1_2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu1/surface-eval",
    # 'Hesai1_2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/hesai1/surface-eval",

    # 'GNSS-IMU1_3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/gnss-imu1/surface-eval",
    # 'Hesai1_3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/hesai1/surface-eval",

    # 'GNSS-IMU2_2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu2/surface-eval",
    # 'Hesai2_2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/hesai2/surface-eval",

    # 'GNSS-IMU2_3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/gnss-imu2/surface-eval",
    # 'Hesai2_3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/hesai2/surface-eval",

    # 'GNSS-IMU3_2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu3/surface-eval",
    # 'Hesai3_2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/hesai3/surface-eval",

    # 'GNSS-IMU3_3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/gnss-imu3/surface-eval",
    # 'Hesai3_3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/hesai3/surface-eval",
    
}


'''
the conclusions so far:

-There is no corelation between point to plane error
    and the quality of the reference plane 

-gnss-imu is bad, the hesai is a good initial guess
-we can refine a bad initial guess via scan-to-map registration
-if the initial guess is good already - then results do not change much

-The reference map might make the results slightly worse 2cm 
    due to different FOV - or something else 

-Using too many BA iterations - makes it workse 2 iters better than 3




WE STICK WITH 2BA METHOD


'''

# Bootstrap parameters
n_bootstrap = 10000
rng = np.random.default_rng(seed=42)  # reproducible

data = {}
for label, folder in methods.items():
    all_data = []
    scans = 500
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            file_path = os.path.join(folder, fname)
            scan_data = np.loadtxt(file_path)
            
            valid_rows = (scan_data[:, 0] >= 0) # Filter out invalid rows (p2plane_error == -1)
            valid_rows = ((scan_data[:, 0] >= 0) & (scan_data[:, 3] > 0.0001))            
            
            all_data.append(scan_data[valid_rows])

            # scans-=1
            # if scans < 0:
            #     break

    data[label] = np.vstack(all_data)

    print(label,": has", np.shape(data[label]))

# Metrics to evaluate
metrics = {
    'Point-to-surface error': 0,
    #'Curvature': 3,
    'Neighbours in a 1 m radius ball': 4
}

def compute_stats(arr):
    return {
        'Mean': np.mean(arr),
        'Median': np.median(arr),
        'RMSE': np.sqrt(np.mean(np.square(arr))),
        'Std Dev': np.std(arr),
    }

def show_stats():
    print('\nshow_stats')

    print('draw box plots')
    for metric_name, col_idx in metrics.items():
        plt.figure(figsize=(10, 6))

        labels = list(data.keys())
        values = [data[label][:, col_idx] for label in labels]

        box = plt.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False, notch=False) #, 

        legend_handles = []
        for patch, color, label in zip(box['boxes'], colors, labels):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
            legend_handles.append(mpatches.Patch(color=color, label=label))

        for median_line in box['medians']:
            median_line.set_alpha(0)  # or median_line.set_visible(False)
            #median_line.set(color='grey', linewidth=2, linestyle='dotted')

        for mean_line in box['means']:
            mean_line.set(color='black', linewidth=2, linestyle='--')

        for line in box['whiskers'] + box['caps']:
            line.set(color='black', linewidth=1.2)

        plt.ylabel(metric_name)
        #plt.title(f'Distribution of {metric_name}')
        plt.xticks(np.arange(1, len(labels) + 1), labels)
        plt.legend(handles=legend_handles, title="Method", loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.xticks([])
        plt.draw()

        #break

    plt.draw()

    print('draw bar plots')
    for metric_name, col_idx in metrics.items():
        plt.figure(figsize=(10, 6))

        stats_summary = {label: compute_stats(d[:, col_idx]) for label, d in data.items()}
        stat_keys = list(next(iter(stats_summary.values())).keys())
        x = np.arange(len(stat_keys))
        width = 0.8 / len(data)
        print('\nmetric_name:',metric_name)
        for i, (label, stats) in enumerate(stats_summary.items()):
            values = [stats[k] for k in stat_keys]
            plt.bar(x + i * width - width * len(data)/2, values, width, label=label)

            print(label, ':', values)

        plt.ylabel(metric_name)
        #plt.title(f'Statistics for {metric_name}')
        plt.xticks(x, stat_keys)
        plt.legend(title="Method", loc='best')
        plt.tight_layout()
        plt.grid(False)
        plt.draw()
        #plt.show()

    
    # KDE plots for distribution
    print('draw KDE')
    for metric_name, col_idx in metrics.items():
        plt.figure(figsize=(10, 5))
        for label in data:
            sns.kdeplot(data[label][:, col_idx], label=label, fill=True, bw_adjust=0.5)
        plt.title(f'Distribution of {metric_name}')
        plt.xlabel(metric_name)
        plt.ylabel("Density")
        plt.legend(title="Method", loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.draw()

        break


    # Cumulative Distribution (CDF) plots for each metric
    print('draw CDF')
    for metric_name, col_idx in metrics.items():
        plt.figure(figsize=(10, 5))
        for label in data:
            values = np.sort(data[label][:, col_idx])
            cdf = np.linspace(0, 1, len(values))
            plt.plot(values, cdf, label=label)

        #plt.title(f'Cumulative Distribution of {metric_name}')
        plt.xlabel(metric_name)
        plt.ylabel("Cumulative Probability")
        plt.legend(title="Method", loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.draw()

        break

def show_correlation():
    print('show_correlation')
    
    #Pearson correlation coefficient matrix between x and y. 1 (perfect positive correlation), 0.1 – 0.3	Weak positive correlation
    #WE WANT TO SHOW THAT THERE IS NO CORRELATION - BETWEEN THE CURVATURE AND THE POINT TO PLANE ERROR 
    
    color_index = 0
    for label, d in data.items():
        #print('color_index:',color_index,",  colors:", np.shape(colors))
        c = colors[color_index]
        color_index += 1
        #plt.figure(figsize=(10, 6))
        errors = d[:, 0]  # Point-to-surface error 

        # # Bootstrap resampling
        # bootstrap_means = np.array([
        #     rng.choice(errors, size=int(len(errors)/100), replace=True).mean()
        #     for _ in range(n_bootstrap)
        # ])

        # # 95% confidence interval (change percentiles as needed)
        # lower = np.percentile(bootstrap_means, 2.5)
        # upper = np.percentile(bootstrap_means, 97.5)
        # mean = np.mean(errors)
        # ci_half_width = (upper - lower) / 2

        # print(f"Mean: {mean:.4f}, 95% CI: ({lower:.4f}, {upper:.4f})")
        # print(f"Mean: {mean:.4f} ± {ci_half_width:.4f} (95% CI)")

        x = d[::30, 0]  # Point-to-surface error 
        y = d[::30, 3]  # Curvature

        
        pearson = np.corrcoef(x, y)[0,1]
        print('{} correlation {}'.format(label, pearson))
        # plt.scatter(x, y, s=1, c=c, alpha=0.5, label=f'{label} (r={pearson:.2f})')

        # # Plot horizontal line: mean error
        # mean_error = np.mean(y)
        # # plt.axhline(mean_error, linestyle='--', color = 'black', linewidth=2, alpha=1)
        # # plt.text(np.max(x)*0.95, mean_error, f"mean={mean_error:.3f}", va='bottom', ha='right', fontsize=14, color='black')

        # # Optional: Fit a line (linear regression)
        # # slope, intercept, r_value, _, _ = linregress(x, y)
        # # x_fit = np.linspace(np.min(x), np.max(x), 500)
        # # y_fit = slope * x_fit + intercept
        # # plt.plot(x_fit, y_fit, linestyle='-', linewidth=5, label=f'{label} fit (slope={slope:.2e})')

        # #plt.title('Curvature vs Point-to-surface error')
        # plt.xlabel('Point-to-surface error')
        # plt.ylabel('Curvature')
        # plt.legend(title="Method", loc='best')
        # #plt.grid(False)
        # plt.tight_layout()
        # plt.draw()


        plt.figure(figsize=(10, 6))
        # 2D histogram
        plt.hist2d(x, y, bins=100)
        plt.colorbar(label=f'Bins for {label} (r={pearson:.2f})')

        # Add correlation in the title or label if desired
        r = np.corrcoef(x, y)[0, 1]
        #plt.title(f'{label} (r = {r:.2f})')
        #plt.legend(title="Method", loc='best')
        plt.xlabel('Point-to-surface error')
        plt.ylabel('Curvature')
        plt.draw()

show_stats()
plt.draw()

show_correlation()
plt.show()