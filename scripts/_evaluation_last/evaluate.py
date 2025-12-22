#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from evo.core.trajectory import PoseTrajectory3D
import numpy as np
from evo.core import metrics
from evo.tools import log
import pprint
# from evo.tools import plot
import matplotlib.pyplot as plt
# temporarily override some package settings
from evo.tools.settings import SETTINGS
from evo.core import lie_algebra as lie

SETTINGS.plot_usetex = False
from evo.core import sync
from evo.tools import file_interface
import copy
from matplotlib import pyplot
from termcolor import colored
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams['lines.linewidth'] = 2  

from scipy.stats import norm
from scipy import stats
from scipy.stats import exponpow
from matplotlib.dates import date2num, DateFormatter
import datetime as dt
from matplotlib.lines import Line2D
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from evo.core import lie_algebra as lie
from evo.core.transformations import quaternion_from_matrix, quaternion_matrix
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import networkx as nx

max_diff = 0.000001  #s

max_diff = 0.1

import warnings

warnings.filterwarnings("ignore")

import matplotlib
from scipy.stats import linregress

#-----------------
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from scipy.ndimage import rotate
# Define custom tile provider
from matplotlib.ticker import FuncFormatter

from xyzservices import TileProvider
from PIL import Image

sns.set()


font_legend = 20-4
font = 22-4

font_legend = 14
font = 14

#PUT THESE BACK 
matplotlib.rc('xtick', labelsize=font)
matplotlib.rc('ytick', labelsize=font)
plt.rcParams.update({'font.size': font})
plt.rc('axes', titlesize=font)     # Set the font size of the title
plt.rc('axes', labelsize=font)     # Set the font size of the x and y labels
plt.rc('xtick', labelsize=font)    # Set the font size of the x tick labels
plt.rc('ytick', labelsize=font)    # Set the font size of the y tick labels
plt.rc('legend', fontsize=font_legend)    # Set the font size of the legend
plt.rc('font', size=font)          # Set the general font size'''

plt.rcParams['hatch.linewidth'] = 2.0  # default is ~1.0

markersize = 15
# plt.rcParams['lines.markersize'] = 10  # radius
sns.set_style("whitegrid")
#----------------------------------------------------------------


translation_ENU_to_origin = np.array([4525805.18109165318310260773, 5070965.88124799355864524841,  114072.22082747340027708560])
rotation_ENU_to_origin = np.array([[-0.78386212744029792887, -0.62091317757072628236,  0.00519529438949398702],
                                    [0.62058202788821892337, -0.78367126767620609584, -0.02715310076057271885],
                                    [0.02093112101431114647, -0.01806018100107483967,  0.99961778597386541367]])

R_origin_to_ENU = rotation_ENU_to_origin.T
t_origin_to_ENU = -R_origin_to_ENU @ translation_ENU_to_origin


bbox_to_anchor=(0.5, -0.12) #bottom
bbox_to_anchor=(0.5, 1.2)   #top
bbox_to_anchor=(0.5, 1)   #top

ncol = 3
# ncol = 4

# ncol = 2


#used so far
search_radius = 3.0
min_time_diff = 100
distance_upper_bound = 1.0 #distance_upper_bound xy max distance between the points 


class TrajectoryReader(object):
    def __init__(self, path_gt, path_model, model_name = '', path_time='', path_iterations='', align = True):
        self.path_gt = path_gt
        self.path_model = path_model
        self.model_name = model_name
        self.time = [0]
        self.iterations = [0]
        #the prev values
        # traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        # traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        traj_gt = file_interface.read_custom_trajectory_file2(self.path_gt)
        traj_model = file_interface.read_custom_trajectory_file2(self.path_model)

        # traj_gt = self.clip(traj_gt)

        # self.traj_gt, self.traj_model = traj_gt, traj_model
        self.traj_gt, self.traj_model = sync.associate_trajectories(traj_gt, traj_model, max_diff)

        # if align:
        self.traj_model.align(self.traj_gt, correct_scale=False, correct_only_scale=False, n=-1)     
        # self.traj_model.align_origin(traj_ref=self.traj_gt)
        # self.traj_model.align(self.traj_gt, correct_scale=False, correct_only_scale=False, n=20)     

        
        self.T_origin_to_ENU = np.eye(4)
        self.T_origin_to_ENU[:3, :3] = R_origin_to_ENU
        self.T_origin_to_ENU[:3, 3] = t_origin_to_ENU

        self.traj_gt = self.transform_trajectory_to_ENU(self.traj_gt, self.T_origin_to_ENU)
        self.traj_model = self.transform_trajectory_to_ENU(self.traj_model, self.T_origin_to_ENU)

        print("Reference trajectory:", self.traj_gt.positions_xyz.shape)
        print("Model:", self.traj_model.positions_xyz.shape)

        if path_time != '':
            self.time = np.loadtxt(path_time)[:,2]

        if path_iterations != '':
            self.iterations = np.loadtxt(path_iterations)[:,2]

    def transform_trajectory_to_ENU(self, traj: PoseTrajectory3D, T_origin_to_ENU: np.ndarray) -> PoseTrajectory3D:
        new_poses_se3 = []
        for pose in traj.poses_se3:
            T_local = pose
            T_enu = T_origin_to_ENU @ T_local
            new_poses_se3.append(T_enu)

        test_no_fail = 9999999
        #this is used to show when it fails the ablation study 
        
        # if self.model_name in ['LI', 'LI-VUX', '*-LI', '*-LI-VUX', 'test', 'test_now']:
        #     test_no_fail = 4650

        return PoseTrajectory3D(
            poses_se3=new_poses_se3[:test_no_fail],
            timestamps=traj.timestamps[:test_no_fail])

    def clip(self, traj: PoseTrajectory3D) -> PoseTrajectory3D:
        new_poses_se3 = []
        for pose in traj.poses_se3:
            new_poses_se3.append(pose)

        test_no_fail = 2400 # 5400  # 9999999
        
        return PoseTrajectory3D(
            poses_se3=new_poses_se3[:test_no_fail],
            timestamps=traj.timestamps[:test_no_fail])

    

    def traveled_distance(self, est_xyz: np.ndarray) -> float:
        # Compute differences between consecutive positions
        diffs = np.diff(est_xyz, axis=0)
        
        # Compute Euclidean distances between consecutive points
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        return np.sum(segment_lengths)

    def APE_translation(self):
        print('\nAPE_translation')
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        # ape_metric = metrics.APE(metrics.PoseRelation.full_transformation)
        # ape_metric = metrics.APE(metrics.PoseRelation.point_distance)

        ape_metric.process_data((self.traj_gt, self.traj_model))
        self.ape_statistics_t = ape_metric.get_all_statistics()
        print("Model ape_statistics_t:", self.ape_statistics_t)

        self.ape_t_error_vectors = ape_metric.error  # shape (N, 3)
    
    def APE_rotation(self):
        print('\nAPE_rotation')
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        ape_metric.process_data((self.traj_gt, self.traj_model))
        self.ape_statistics_r = ape_metric.get_all_statistics()
        print("Model ape_statistics_r:", self.ape_statistics_r)

        self.ape_r_error_vectors = ape_metric.error  # shape (N, 3)
    
    def RPE_translation(self):
        print('\nRPE_translation')
        
        # Calculate relative pose error
        all_pairs = True
        delta_unit = metrics.Unit.meters
        delta = 100
        rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
        # rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        # rpe_metric = metrics.RPE(metrics.PoseRelation.full_transformation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)




        rpe_metric.process_data((self.traj_gt, self.traj_model))
        self.rpe_statistics_t = rpe_metric.get_all_statistics()
        print("Model rpe_statistics_t:", self.rpe_statistics_t)

        self.rpe_t_error_vectors = rpe_metric.error  # shape (N, 3)

    def RPE_rotation(self):
        print('\nRPE_rotation')
        
        # Calculate relative pose error
        all_pairs = True
        delta_unit = metrics.Unit.meters
        delta = 100
                
        rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        rpe_metric.process_data((self.traj_gt, self.traj_model))
        self.rpe_statistics_r = rpe_metric.get_all_statistics()
        print("Model rpe_statistics_r:", self.rpe_statistics_r)

        self.rpe_r_error_vectors = rpe_metric.error


path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS_prev.txt"
path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS_gnss.txt"
path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS.txt"

methods = {
    #  'rko'             : '/home/eugeniu/zz_zx_final/rko',
    # 'point-lio'             : '/home/eugeniu/zz_zx_final/point-lio',

    ## 'Reference trajectory' : '/home/eugeniu/zz_zx_final/ref',
    # '0_LI'                     : '/home/eugeniu/zz_zx_final/0_LI',
    '1_LI_robust_adaptive' : '/home/eugeniu/zz_zx_final/1_LI_robust_adaptive',
    # '2_LI_robust_adaptive_g' : '/home/eugeniu/zz_zx_final/2_LI_robust_adaptive_g',
    '3_LI_robust_adaptive_p2p_p2pl' : '/home/eugeniu/zz_zx_final/3_LI_robust_adaptive_p2p_p2pl',
    # '4_LI_robust_adaptive_backwardPass' : '/home/eugeniu/zz_zx_final/4_LI_robust_adaptive_backwardPass',
    # '5_LI_robust_adaptive_backwardPass_p2p_p2pl' : '/home/eugeniu/zz_zx_final/5_LI_robust_adaptive_backwardPass_p2p_p2pl',


    # 'R-A LI + IMU_g-update'               : '/home/eugeniu/zz_zx_final/prev/li5', #robust adaptive li , + imu gravity update
    # # 'R-A LI + (planes + p2p)'             : '/home/eugeniu/zz_zx_final/prev/li2',
    # 'R-A LI'                              : '/home/eugeniu/zz_zx_final/prev/li1', #robust adaptive li , no rot-tran coupled
    # 'R-A LI + backwardPass'               : '/home/eugeniu/zz_zx_final/prev/li3',

    'madgwick_p2p_p2pl'             : '/home/eugeniu/zz_zx_final/madgwick',

    'g_p2p_p2pl'             : '/home/eugeniu/zz_zx_final/g_p2p_p2pl',



    # 'test-prev'             : '/home/eugeniu/zz_zx_final/test-prev',

    'test'             : '/home/eugeniu/zz_zx_final/test',

   

}

# raw_mat = file_interface.read_vel(path_gt)

def funct_debug():
    file_path = "/home/eugeniu/zz_zx_final/madgwick/debugMLS.txt"

    data = np.loadtxt(file_path)
    l = len(data)
    print("data:", np.shape(data))

    data = data[:100000, :]

    time = data[:, 0]

    vel = data[:, 1:4]   # vx, vy, vz
    ba  = data[:, 4:7]   # bax, bay, baz
    bw  = data[:, 7:10]  # bwx, bwy, bwz
    dt  = data[:, 10]

    # Create figure
    # plt.figure(figsize=(12, 8))

    # --- 1. Time ---
    # plt.subplot(2, 2, 1)
    # plt.plot(dt, linewidth=1.5)
    # plt.xlabel("Index")
    # plt.ylabel("Time [s]")
    # plt.title("dt")
    # plt.grid(True)

    # --- 2. Velocity ---
    # plt.subplot(2, 2, 2)
    # vel_vector = np.linalg.norm(vel, axis = 1)
    # print("vel_vector:", np.shape(vel_vector))
    # # plt.plot(time, vel[:, 0], 'r', label='v_x')
    # # plt.plot(time, vel[:, 1], 'g', label='v_y')
    # # plt.plot(time, vel[:, 2], 'b', label='v_z')
    # plt.plot(time, vel_vector, 'k', label='vel_vector')

    # max_vel_m_s = 20 / 3.6
    # vel_line = np.full_like(time, max_vel_m_s)
    # print("vel_line:", np.shape(vel_line))
    # plt.plot(time, vel_line, '--', label='20 km/h')

    # plt.xlabel("Time [s]")
    # plt.ylabel("Velocity [m/s]")
    # plt.title("Velocity")
    # plt.legend()
    # plt.grid(True)
    # plt.draw()

    vel_vector = np.linalg.norm(vel, axis=1)
    print("vel_vector:", vel_vector.shape)

    max_vel_m_s = 30.0 / 3.6  # 30 km/h in m/s
    vel_line = np.full_like(time, max_vel_m_s)

    # --- gain computation ---
    beta_min = 0.001
    beta_max = 0.1

    v_n = vel_vector / max_vel_m_s
    v_n = np.clip(v_n, 0.0, 1.0)

    beta = beta_min + (1.0 - v_n) * (beta_max - beta_min)

    # --- figure ---
    fig, ax1 = plt.subplots(figsize=(12, 8))


    # Differentiate to get acceleration
    acceleration = np.abs(np.diff(vel_vector) / dt[:-1])
    print("acceleration:", np.shape(acceleration))





    # Left Y-axis: velocity
    ax1.plot(time, vel_vector, 'k', label='|v| [m/s]')
    

    ax1.plot(time, vel_line, '--', label='30 km/h')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [m/s]")
    ax1.grid(True)

    # Right Y-axis: beta
    ax2 = ax1.twinx()
    ax2.plot(time, beta, 'r', label='β (Madgwick gain vel)')
    ax2.set_ylabel("Gain β")

    # v_n = acceleration / .1 #2 max acceleration
    # v_n = np.clip(v_n, 0.0, 1.0)
    # beta = beta_min + (1.0 - v_n) * (beta_max - beta_min)
    # ax2.plot(time[:-1], beta, 'b', label='β (Madgwick gain acc)')

    # --- combine legends ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Velocity and Adaptive Madgwick Gain")
    plt.tight_layout()
    plt.draw()

    plt.figure()
    plt.plot(time[:-1], acceleration, label="acc")
    plt.legend()
    plt.grid(True)

    # --- 3. Accelerometer bias ---
    # plt.subplot(2, 2, 3)
    # plt.figure(figsize=(12, 8))
    # beta_min = 0.001
    # beta_max = 0.1
    # v_n = vel_vector / max_vel_m_s
    # v_n = np.clip(v_n, 0.0, 1.0)

    # beta = beta_min + (1.0 - v_n) * (beta_max - beta_min)
    # plt.plot(time, beta, 'r', label='beta')
    # plt.plot(time, ba[:, 0], 'r', label='ba_x')
    # plt.plot(time, ba[:, 1], 'g', label='ba_y')
    # plt.plot(time, ba[:, 2], 'b', label='ba_z')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Accel Bias")
    # plt.title("Accelerometer Bias (ba)")
    # plt.legend()
    # plt.grid(True)

    # --- 4. Gyroscope bias ---
    # plt.subplot(2, 2, 4)
    # plt.plot(time, bw[:, 0], 'r', label='bw_x')
    # plt.plot(time, bw[:, 1], 'g', label='bw_y')
    # plt.plot(time, bw[:, 2], 'b', label='bw_z')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Gyro Bias")
    # plt.title("Gyroscope Bias (bw)")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()

# funct_debug()
# exit()

methods_data = {
    '0_LI' :  ["#1f77b4",'M'],
    '1_LI_robust_adaptive'      : ['#9467bd','D'],
    '2_LI_robust_adaptive_g'             : ['#2ca02c','A'],
    '3_LI_robust_adaptive_p2p_p2pl'                 : ['#ff7f0e','C'],
    '4_LI_robust_adaptive_backwardPass' : ['#17becf','F'],
    '5_LI_robust_adaptive_backwardPass_p2p_p2pl' : ['#e377c2','H'],





    'test' : ["#648099",'Z'],
    'madgwick_p2p_p2pl' : ['#8c564b','G'],

    'Reference trajectory' : ['#d62728','L'],
    

    'R-A LI + IMU_g-update'      : ['#9467bd','D'],

    'R-A LI + backwardPass' : ['#bcbd22','E'],
    'R-A LI' : ["#808019",'E'],

    'test-prev'      : ["#651e1e",'D'],
    'rko' : ["#808019",'E'],







    # 'GNSS-INS'             : ['#2ca02c','A'],

    # # # 'LI-VUX-(raw)GNSS'      : ['#7f7f7f','D'],

    


    

    'point-lio' : ["#ff7f0e",'N'], #['#04f810','N'],

    'g_p2p_p2pl' : ["#26391B",'Z'],
    'test_now' : ["#a86b40",'S'],
    '*-LI-VUX + D-ALS (t-coupled)' : ["#e377c2",'S'],
}

colors = ['tab:brown', 'tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'cyan', 'lime','orange','gray']

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', '-', '--', ':',]
lab = ['A','B','C','D','E','F','G','H','K','X']
lt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

obj_gt = TrajectoryReader(path_gt, path_gt)
all_traj = [obj_gt.traj_gt]
# 3D Plot
fig_3d = plt.figure(figsize=(10, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_title("3D Trajectories")
ax_3d.set_xlabel("East [m]")
ax_3d.set_ylabel("North [m]")
ax_3d.set_zlabel("Height [m]")


data_ape_t = {}
data_rpe_t = {}
data_ape_r = {}
data_rpe_r = {}
data_time = {}
data_iterations = {}

positions = all_traj[0].positions_xyz 
ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],label="Reference trajectory", color="black")
# ax_2d.plot(positions[:, 0], positions[:, 1],label="Reference trajectory", color="black",) 
x_limits = [np.min(positions[:, 0]), np.max(positions[:, 0])]
y_limits = [np.min(positions[:, 1]), np.max(positions[:, 1])]
z_limits = [np.min(positions[:, 2]), np.max(positions[:, 2])]

# Compute the global range and center
x_range = x_limits[1] - x_limits[0]
y_range = y_limits[1] - y_limits[0]
z_range = z_limits[1] - z_limits[0]
max_range = max(x_range, y_range, z_range)
x_center = np.mean(x_limits)
y_center = np.mean(y_limits)
z_center = np.mean(z_limits)
# Set equal limits for each axis
ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)


for idx, (label, path) in enumerate(methods.items()):
    #for label, path in methods.items():
    print('\n Model ',label)
    path_model=path+"/MLS.txt"
    path_time=path+"/MLS_time.txt"
    path_iterations=path+"/MLS_iter.txt"
    obj = TrajectoryReader(path_gt, path_model, label, path_time, path_iterations)

    obj.APE_translation()
    obj.RPE_translation()
    # obj.APE_rotation()
    # obj.RPE_rotation()
    all_traj.append(obj.traj_model)

    data_ape_t[label] = obj.ape_t_error_vectors
    data_rpe_t[label] = obj.rpe_t_error_vectors

    # data_ape_r[label] = obj.ape_r_error_vectors
    # data_rpe_r[label] = obj.rpe_r_error_vectors

    data_time[label] = obj.time
    data_iterations[label] = obj.iterations

    positions = obj.traj_model.positions_xyz  # N x 3 numpy array

    ax_3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        label=label, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)]
    )

ax_3d.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
          ncol = ncol, fancybox=True, shadow=True)
ax_3d.grid(True)
plt.draw()

# plt.show()
# exit()

def plot_box(data, metric = '',  show_legend = True, show_cumulative = False, add_arrows = False):
    print('plot_box for ',metric)
    labels = list(data.keys())

    #labels_local = lab[0:len(labels)]
    lt_local = lt[0:len(labels)]

    plt.figure(figsize=(10, 6))
    

    values = [data[label] for label in labels]

    box = plt.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False, notch=False) 
    ind = 0
    legend_handles = []
    labels_local = []
    colors_ = [methods_data[label][0] for label in labels]

    for patch, color, label in zip(box['boxes'], colors_, labels):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch_legend = mpatches.Patch(
            facecolor=color,
            # edgecolor='black',
            label=methods_data[label][1] + " : " + label
        )
        
        labels_local.append(methods_data[label][1])
        
        if '*' in label:
            patch.set_hatch('\\')  # or 'xx', '\\',  //etc.
            patch.set_edgecolor('white') 
            patch_legend = mpatches.Patch(
                facecolor=color,
                edgecolor='white',   
                hatch='\\',          
                label=methods_data[label][1] + " : " + label
            )
        
        # legend_handles.append(mpatches.Patch(color=color, label = methods_data[label][1]+" : "+label))
        legend_handles.append(patch_legend)
        if add_arrows:
            mean_value = np.mean(data[label])
            #If the mean is greater than 2, annotate it
            if mean_value > 0.9:
                box['boxes'][ind].set_visible(False)
                # Hide whiskers (2 per box)
                box['whiskers'][2*ind].set_visible(False)
                box['whiskers'][2*ind + 1].set_visible(False)

                box['caps'][2*ind].set_visible(False)
                box['caps'][2*ind + 1].set_visible(False)
                box['medians'][ind].set_visible(False)
                box['means'][ind].set_visible(False)


                x_pos = ind + 1  # boxplot x positions are 1-indexed
                y_base = 0.85       # arrow starts at y = y_base

                # Draw upward arrow and label
                plt.annotate("", xy=(x_pos, y_base + 0.15), xytext=(x_pos, y_base),
                            arrowprops=dict(arrowstyle="->", lw=3, color='black'),
                            xycoords='data', annotation_clip=False)

                plt.text(x_pos - 0.26, y_base + 0.15, f"{mean_value:.2f}",
                        fontsize=font-2, weight=600, color='black')

        ind += 1

    for median_line in box['medians']:
        median_line.set_alpha(0)  # or median_line.set_visible(False)
        #median_line.set(color='grey', linewidth=2, linestyle='dotted')

    for mean_line in box['means']:
        mean_line.set(color='black', linewidth=2, linestyle='--')

    for line in box['whiskers'] + box['caps']:
        line.set(color='black', linewidth=1.2)

    plt.title(f'Box plot of {metric}')
    


    # plt.xticks(np.arange(1, len(labels) + 1), labels, labelsize=font)
    if show_legend:
        plt.legend(handles=legend_handles,  loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
            ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)
    plt.grid(True)
    #plt.tight_layout()
    #plt.xticks(rotation=90)
    #plt.xticks([])
    plt.xticks(lt_local, labels_local, fontsize = font) 
    # if first_got_legend:
    #     plt.legend().set_visible(False)
    
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.2f}"))   
    
    plt.ylabel(metric, fontsize=font)
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.draw()


    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.2f}"))  

    if show_cumulative:
        plt.figure(figsize=(10, 5))

        values = [data[label] for label in labels]

        i=0
        for patch, color, label in zip(box['boxes'], colors_, labels):
            #for i, label in enumerate(data):
            values = np.sort(data[label])
            cdf = np.linspace(0, 1, len(values))
            plt.plot(values, cdf, label=label, color = color, linestyle = linestyles[i])

            #plt.title(f'Cumulative Distribution of {metric_name}')
            plt.xlabel(metric, fontsize=font)
            plt.ylabel("Cumulative Probability", fontsize=font)
            if show_legend:
                plt.legend( loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
                ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)
            plt.grid(True)
            #plt.tight_layout()
            plt.tick_params(axis='both', which='major', labelsize=font)
            plt.draw()
            i+=1

def plot_lines(data, metric = '',  show_legend = True):
    print('plot_lines for ',metric)
    labels = list(data.keys())

    lt_local = lt[0:len(labels)]
    plt.figure(figsize=(10, 6))
    
    for label in labels:
        mean = np.mean(data[label])
        std = np.std(data[label])
        plt.plot(data[label], color = methods_data[label][0], label = methods_data[label][1] + " : " + label+ " ( mean:" + str(round(mean, 2)) + " ) ( std:" + str(round(std, 2)) + " )")

    plt.title(f'Box plot of {metric}')
    
    # if show_legend:
    #     plt.legend(handles=legend_handles,  loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
    #         ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)
    plt.grid(True)
    plt.ylabel(metric, fontsize=font)
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.draw()
    plt.legend()

plot_box(data_ape_t, 'ATE translation (m)', show_legend = True, add_arrows = False)
plot_box(data_rpe_t, 'RTE translation (%)', show_legend = True)

# plot_box(data_ape_r, 'ATE rotation (m)', show_legend = True, add_arrows = False)
# plot_box(data_rpe_r, 'RTE rotation (%)', show_legend = True)


# plot_lines(data_time, 'Time (ms)', show_legend = True)
# plot_lines(data_iterations, 'Iterations', show_legend = True)

plt.show()