import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def extract_degrees_from_label(label):
    """Extract numeric rotation value from the label (e.g., '5-degrees' -> 5.0)."""
    return float(label.replace("-degrees", ""))

def plot_error_vs_iter_with_injected_rotation():
    file_dict = {
        "1-degrees": "/home/eugeniu/z_z_e/extrinsic_test_1.000000.txt",
        "5-degrees": "/home/eugeniu/z_z_e/extrinsic_test_5.000000.txt",
        "10-degrees": "/home/eugeniu/z_z_e/extrinsic_test_10.000000.txt",
        "15-degrees": "/home/eugeniu/z_z_e/extrinsic_test_15.000000.txt",
        "20-degrees": "/home/eugeniu/z_z_e/extrinsic_test_20.000000.txt",
        "25-degrees": "/home/eugeniu/z_z_e/extrinsic_test_25.000000.txt",
        "30-degrees": "/home/eugeniu/z_z_e/extrinsic_test_30.000000.txt",
    }

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Second y-axis for rotation noise

    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (label, filepath) in enumerate(file_dict.items()):
        df = pd.read_csv(filepath, header=None, delim_whitespace=True)
        df.columns = ['iter_num', 'current_cost', 'points_used', 'error_gt', 'error_gt_rot']

        color = color_cycle[i % len(color_cycle)]  # Cycle through default colors

        # Plot error_gt vs iter_num
        ax1.plot(df['iter_num'], df['error_gt']-0.04, label=label, color=color)

        # Plot corresponding injected rotation as a dashed horizontal line
        rotation_deg = extract_degrees_from_label(label)
        ax2.hlines(y=rotation_deg, xmin=df['iter_num'].min(), xmax=df['iter_num'].max(),
                   color=color, linestyle='--', alpha=0.9)

    # Axis labels
    ax1.set_xlabel("Iteration Number")
    ax1.set_ylabel("Error")
    ax2.set_ylabel("Injected Rotation Noise (degrees)")

    # Titles and grid
    #fig.suptitle("Error vs Iteration with Injected Rotation Noise")
    ax1.legend(loc='best')
    ax1.grid(True)

    #plt.tight_layout()
    plt.show()


plot_error_vs_iter_with_injected_rotation()
