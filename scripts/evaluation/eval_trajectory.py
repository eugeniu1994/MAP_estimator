#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from evo.core.trajectory import PoseTrajectory3D
import numpy as np
from evo.core import metrics
from evo.tools import log
import pprint
from evo.tools import plot
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
import seaborn as sns

sns.set()
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

max_diff = 0.000001  #s

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

from scipy.stats import linregress

font = 16
plt.rcParams.update({'font.size': font})
plt.rc('axes', titlesize=font)     # Set the font size of the title
#plt.rc('axes', labelsize=font)     # Set the font size of the x and y labels
#plt.rc('xtick', labelsize=font)    # Set the font size of the x tick labels
#plt.rc('ytick', labelsize=font)    # Set the font size of the y tick labels
plt.rc('legend', fontsize=font)    # Set the font size of the legend
#plt.rc('font', size=font)          # Set the general font size'''

markersize = 8
sns.set_style("whitegrid")
#----------------------------------------------------------------

class TrajectoryReader(object):
    def __init__(self, path_gt, path_noise, path_model):
        self.path_gt = path_gt
        self.path_noise = path_noise
        self.path_model = path_model

    def plot_data(self):
        traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        traj_noisy = file_interface.read_custom_trajectory_file(self.path_noise)
        traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        print("GT:", traj_gt.positions_xyz.shape)
        print("Noisy:", traj_noisy.positions_xyz.shape)
        print("Model:", traj_model.positions_xyz.shape)

        fig = plt.figure(figsize=(10, 6))  
        plot.trajectories(
            fig,  
            [traj_gt, traj_noisy, traj_model], 
            plot.PlotMode.xyz,
            title="Aligned Trajectories")
        
        plt.draw()  

    def APE_translation(self):
        print('\nAPE_translation')
        traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        traj_noisy = file_interface.read_custom_trajectory_file(self.path_noise)
        traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)

        ape_metric.process_data((traj_gt, traj_model))
        ape_statistics = ape_metric.get_all_statistics()
        print("Model APE statistics:", ape_statistics)

        ape_metric.process_data((traj_gt, traj_noisy))
        ape_statistics = ape_metric.get_all_statistics()
        print("Noisy APE statistics:", ape_statistics)
    
    def APE_rotation(self):
        print('\nAPE_rotation')
        traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        traj_noisy = file_interface.read_custom_trajectory_file(self.path_noise)
        traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        ape_metric.process_data((traj_gt, traj_model))
        ape_statistics = ape_metric.get_all_statistics()
        print("Model APE statistics:", ape_statistics)

        ape_metric.process_data((traj_gt, traj_noisy))
        ape_statistics = ape_metric.get_all_statistics()
        print("Noisy APE statistics:", ape_statistics)
    
    def RPE_translation(self):
        print('\nRPE_translation')
        traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        traj_noisy = file_interface.read_custom_trajectory_file(self.path_noise)
        traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        
        # Calculate relative pose error
        all_pairs = True
        delta_unit = metrics.Unit.meters
        delta =  10
                
        rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        rpe_metric.process_data((traj_gt, traj_model))
        ape_statistics = rpe_metric.get_all_statistics()
        print("Model RPE statistics:", ape_statistics)

        rpe_metric.process_data((traj_gt, traj_noisy))
        ape_statistics = rpe_metric.get_all_statistics()
        print("Noisy RPE statistics:", ape_statistics)

    def RPE_rotation(self):
        print('\nRPE_rotation')
        traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        traj_noisy = file_interface.read_custom_trajectory_file(self.path_noise)
        traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        
        # Calculate relative pose error
        all_pairs = True
        delta_unit = metrics.Unit.meters
        delta =  10
                
        rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        rpe_metric.process_data((traj_gt, traj_model))
        ape_statistics = rpe_metric.get_all_statistics()
        print("Model RPE statistics:", ape_statistics)

        rpe_metric.process_data((traj_gt, traj_noisy))
        ape_statistics = rpe_metric.get_all_statistics()
        print("Noisy RPE statistics:", ape_statistics)

    def all(self):
        print()
        # Compute APE for each axis
        # ape_metric_x = metrics.APE(metrics.PoseRelation.translation_part, pose_relation=metrics.PoseRelation.x_axis)
        # ape_metric_x.process_data((traj_est_aligned, traj_ref))
        # print("X-axis error (m):", ape_metric_x.get_statistic(metrics.StatisticsType.rmse))

        # ape_metric_y = metrics.APE(metrics.PoseRelation.translation_part, pose_relation=metrics.PoseRelation.y_axis)
        # ape_metric_y.process_data((traj_est_aligned, traj_ref))
        # print("Y-axis error (m):", ape_metric_y.get_statistic(metrics.StatisticsType.rmse))

        # ape_metric_z = metrics.APE(metrics.PoseRelation.translation_part, pose_relation=metrics.PoseRelation.z_axis)
        # ape_metric_z.process_data((traj_est_aligned, traj_ref))
        # print``("Z-axis error (m):", ape_metric_z.get_statistic(metrics.StatisticsType.rmse))

        # from scipy.spatial.transform import Rotation

        # # Get quaternions (xyzw → wxyz for scipy)
        # quats_est = traj_est_aligned.orientations_quat_wxyz
        # quats_ref = traj_ref.orientations_quat_wxyz

        # # Convert to Euler angles (ZYX convention: yaw, pitch, roll)
        # euler_est = Rotation.from_quat(quats_est).as_euler('zyx', degrees=True)
        # euler_ref = Rotation.from_quat(quats_ref).as_euler('zyx', degrees=True)

        # # Compute errors (roll, pitch, yaw)
        # roll_error = np.abs(euler_est[:, 2] - euler_ref[:, 2])  # Roll (X-axis)
        # pitch_error = np.abs(euler_est[:, 1] - euler_ref[:, 1])  # Pitch (Y-axis)
        # yaw_error = np.abs(euler_est[:, 0] - euler_ref[:, 0])    # Yaw (Z-axis)

        # print(f"Roll (X) error (deg): {np.mean(roll_error):.3f} ± {np.std(roll_error):.3f}")
        # print(f"Pitch (Y) error (deg): {np.mean(pitch_error):.3f} ± {np.std(pitch_error):.3f}")
        # print(f"Yaw (Z) error (deg): {np.mean(yaw_error):.3f} ± {np.std(yaw_error):.3f}")
    
    def plot_errors(self):
        noise_levels = [0.1, 0.2, 0.3, 0.4]  # Your noise increments
        trans_ape_noisy = [0.12, 0.25, 0.38, 0.51]  # APE of noisy trajectories
        trans_ape_model = [0.08, 0.15, 0.22, 0.30]  # APE after model correction
        rot_ape_noisy = [5.2, 10.1, 15.3, 20.0]    # Rotation APE (deg)
        rot_ape_model = [3.1, 6.0, 8.5, 11.2]      # Corrected rotation APE
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Translation APE plot
        #ax1.plot(noise_levels, trans_ape_noisy, 'r-o', label='Noisy Input')
        ax1.plot(noise_levels, trans_ape_model, 'g-s', label='Model Output')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Translation APE (m)')
        ax1.set_title('Translation Error vs. Noise')
        ax1.grid(True)

        # Rotation APE plot
        #ax2.plot(noise_levels, rot_ape_noisy, 'r-o', label='Noisy Input')
        ax2.plot(noise_levels, rot_ape_model, 'g-s', label='Model Output')
        ax2.set_xlabel('Noise Level')
        #ax2.set_ylabel('Rotation APE (deg)')
        ax2.set_ylabel(r'Rotation APE ($^\circ$)')
        ax2.set_title('Rotation Error vs. Noise')
        ax2.grid(True)

        plt.legend()
        #plt.tight_layout()
        plt.draw()
        
        # Calculate improvement percentage
        trans_improvement = 100 * (1 - np.array(trans_ape_model) / np.array(trans_ape_noisy))
        rot_improvement = 100 * (1 - np.array(rot_ape_model) / np.array(rot_ape_noisy))

        plt.figure(figsize=(10, 5))
        plt.plot(noise_levels, trans_improvement, 'b-^', label='Translation Improvement')
        plt.plot(noise_levels, rot_improvement, 'm-^', label='Rotation Improvement')
        plt.xlabel('Noise Level')
        plt.ylabel('Error Reduction (%)')
        plt.title('Model Correction Effectiveness')
        plt.legend()
        plt.grid(True)
        plt.draw()



def get_errors_t(p,t,r):
    path_gt = p + 't_{}_r{}_GT.txt'.format(t,r)
    path_model = p + 't_{}_r{}_model.txt'.format(t,r)
    path_noise = p + 't_{}_r{}_noisy.txt'.format(t,r)

    traj_gt = file_interface.read_custom_trajectory_file(path_gt)
    traj_model = file_interface.read_custom_trajectory_file(path_model)
    traj_noise = file_interface.read_custom_trajectory_file(path_noise)
    
    ape_metric_t = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric_t.process_data((traj_gt, traj_model))
    model_stat = ape_metric_t.get_all_statistics()

    ape_metric_t.process_data((traj_gt, traj_noise))
    noise_stat = ape_metric_t.get_all_statistics()

    return model_stat, noise_stat

def get_errors_r(p,t,r):
    path_gt = p + 't_{}_r{}_GT.txt'.format(t,r)
    path_model = p + 't_{}_r{}_model.txt'.format(t,r)
    path_noise = p + 't_{}_r{}_noisy.txt'.format(t,r)

    traj_gt = file_interface.read_custom_trajectory_file(path_gt)
    traj_model = file_interface.read_custom_trajectory_file(path_model)
    traj_noise = file_interface.read_custom_trajectory_file(path_noise)
    
    ape_metric_r = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_metric_r.process_data((traj_gt, traj_model))
    model_stat = ape_metric_r.get_all_statistics()

    ape_metric_r.process_data((traj_gt, traj_noise))
    noise_stat = ape_metric_r.get_all_statistics()

    return model_stat, noise_stat

def plot_dual_axis_combined(noise_levels_t, noise_levels_r, trans_ape_model, rot_ape_model,
                            _noise_levels_t, _trans_ape_model, _noise_levels_r, _rot_ape_model):
    fig, ax1 = plt.subplots()

    # Plot translation error (left y-axis)
    line1, = ax1.plot(noise_levels_t, trans_ape_model, 'r-o', label='Combined Translation Error')
    ax1.set_xlabel('Translation Noise (cm)', color='r')
    ax1.set_ylabel('Translation Error (cm)', color='r')
    ax1.tick_params(axis='x', labelcolor='r')
    ax1.tick_params(axis='y', labelcolor='r')
    line3, = ax1.plot(_noise_levels_t, _trans_ape_model, color='red', linestyle='--', marker='x', label='Translation (only) Error')

    # Right y-axis for rotation error
    ax2 = ax1.twinx()
    line2, = ax2.plot(noise_levels_t, rot_ape_model, 'g-s', label='Combined Rotation Error')
    #ax2.set_ylabel('Rotation Error (deg)', color='g')
    ax2.set_ylabel(r'Rotation Error ($^\circ$)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    line4, = ax2.plot(_noise_levels_r, _rot_ape_model,color='green', linestyle='--', marker='d', label='Rotation (only) Error')
    
    # Top x-axis for rotation noise
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(noise_levels_t)
    ax_top.set_xticklabels([str(r) for r in noise_levels_r], color='g')
    #ax_top.set_xlabel('Rotation Noise (deg)', color='g')
    ax_top.set_ylabel(r'Rotation Noise ($^\circ$)', color='g')
    ax_top.tick_params(axis='x', labelcolor='g')

    ax1.grid(True)
    lines = [line1, line3, line2, line4]
    #lines = [line1,  line2]
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='lower right') 

    plt.draw()

def show_this():    

    c_noise_levels_t, c_noise_levels_r = [], []  
    c_trans_ape_model, c_trans_ape_model_noise = [],[]  
    c_rot_ape_model,c_rot_ape_model_noise = [],[]  
    #combined rotation noise around x and translation noise on z
    p = '/home/eugeniu/xz_final_clouds/Added_noise/x_rot_x_tran_vux_frame/'
    t = ['0.050000', '0.100000', '0.150000', '0.200000', '0.250000', '0.300000', '0.350000', '0.400000', '0.450000', '0.500000'] #m
    r = ['0.017444', '0.034889', '0.052333', '0.069778', '0.087222', '0.104667', '0.122111', '0.139556', '0.157000', '0.174444'] #radians

    c_noise_levels_t = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    c_noise_levels_r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for (ti,ri) in zip(t,r):
        t_error, t_error_noise = get_errors_t(p,ti,ri)
        r_error, r_error_noise = get_errors_r(p,ti,ri)
        print('\n\nt:{}, r:{} in degrees = {}'.format(ti,ri, float(float(ri) * 180. / np.pi)))
        #print('t_error:\n',t_error)
        #print('r_error:\n',r_error)
        print('\n----------------------') #
        print('t_error: ', t_error['mean'],' ',t_error['median'],' ',t_error['rmse'],' ',t_error['std'])
        print('r_error: ', r_error['mean'],' ',r_error['median'],' ',r_error['rmse'],' ',r_error['std'])

        row = ""
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(t_error['mean']* 100,3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(t_error['median']* 100,3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(t_error['rmse']* 100,3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(t_error['std']* 100,3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(r_error['mean'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(r_error['median'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(r_error['rmse'],3)} }} \n"
        row += f"& {{{round(r_error['std'],3)}}}" 

        print(row)


        c_trans_ape_model.append(t_error['mean'] * 100) 
        # c_noise_levels_t.append(float(ti) * 100 )

        c_rot_ape_model.append(r_error['mean'])   
        # c_noise_levels_r.append(float(float(ri) * 180. / np.pi))

        c_trans_ape_model_noise.append(t_error_noise['mean'] * 100)
        c_rot_ape_model_noise.append(r_error_noise['mean'])  

    noise_levels_t, noise_levels_r = [], []  
    trans_ape_model = []  
    rot_ape_model = []      

    #only translation noise added to z axis
    p = '/home/eugeniu/xz_final_clouds/Added_noise/z_only_translation_mls_frame/'
    t = ['0.050000', '0.100000', '0.150000', '0.200000', '0.250000', '0.300000', '0.350000', '0.400000'] #m
    r = ['0.001000', '0.001000', '0.001000', '0.001000', '0.001000', '0.001000', '0.001000', '0.001000'] #radians

    for (ti,ri) in zip(t,r):
        t_error, t_error_noise = get_errors_t(p,ti,ri)
        #r_error, r_error_noise = get_errors_r(p,ti,ri)
        #print('\n\nt:{}, r:{} in degrees = {}'.format(ti,ri, float(float(ri) * 180. / np.pi)))
        #print('t_error:\n',t_error)
        #print('r_error:\n',r_error)
        trans_ape_model.append(t_error['mean'] * 100)   #save the translation 
        noise_levels_t.append(float(ti) * 100 )
        #break

    #only rotation noise added on x axis
    p = '/home/eugeniu/xz_final_clouds/Added_noise/x_only_rotation_vux_frame/'
    t = ['0.001000', '0.001000', '0.001000', '0.001000'] #m
    r = ['0.008722', '0.017444', '0.034889', '0.087222'] #radians
    for (ti,ri) in zip(t,r):
        #t_error, t_error_noise = get_errors_t(p,ti,ri)
        r_error, r_error_noise = get_errors_r(p,ti,ri)
        #print('\n\nt:{}, r:{} in degrees = {}'.format(ti,ri, float(float(ri) * 180. / np.pi)))
        #print('t_error:\n',t_error)
        #print('r_error:\n',r_error)
        rot_ape_model.append(r_error['mean'])   #save the rotation - its already in degrees
        noise_levels_r.append(float(float(ri) * 180. / np.pi))
        #break
    
    plt.figure()
    #plt.plot(noise_levels_t, trans_ape_model, color='red', linestyle='--', marker='x', label='Translation (only) Error')

    slope_model_ape, intercept_a, r_value_a, p_value_a, std_err_a = linregress(c_noise_levels_t, c_trans_ape_model)
    slope_noise_ape, intercept_b, r_value_b, p_value_b, std_err_b = linregress(c_noise_levels_t, c_trans_ape_model_noise)

    plt.plot(c_noise_levels_t, c_trans_ape_model_noise, color='red', linestyle='--', marker='d', markersize=markersize, label=f'Noise translation error (slope={slope_noise_ape:.2f})')
    plt.plot(c_noise_levels_t, c_trans_ape_model, 'g-o', markersize=markersize,  label=f'Model translation error (slope={slope_model_ape:.2f})')
    plt.xticks(c_noise_levels_t)
    plt.xlabel('Noise level std (cm)')
    plt.ylabel('Translation APE (cm)')
    #plt.title('Translation Error vs. Noise')
    plt.grid(False)
    #plt.gca().set_aspect('equal')
    plt.legend()

    slope_model_rpe, intercept_a, r_value_a, p_value_a, std_err_a = linregress(c_noise_levels_r, c_rot_ape_model)
    slope_noise_rpe, intercept_b, r_value_b, p_value_b, std_err_b = linregress(c_noise_levels_r, c_rot_ape_model_noise)

    plt.figure()
    #plt.plot(noise_levels_r, rot_ape_model, color='green', linestyle='--', marker='d', label='Rotation (only) Error')
    plt.plot(c_noise_levels_r, c_rot_ape_model_noise, color='red', linestyle='--', marker='d', markersize=markersize, label=f'Noise rotation error (slope={slope_noise_rpe:.2f})')
    plt.plot(c_noise_levels_r, c_rot_ape_model, 'g-o', markersize=markersize, label=f'Model rotation error (slope={slope_model_rpe:.2f})')
    plt.xticks(c_noise_levels_r)
    plt.xlabel('Noise level std (deg)')
    plt.ylabel('Rotation APE (deg)')

    plt.xlabel(r'Noise level std ($^\circ$)')
    plt.ylabel(r'Rotation APE ($^\circ$)')

    #plt.title('Rotation Error vs. Noise')
    plt.grid(False)
    #plt.gca().set_aspect('equal')
    plt.legend()

    plt.draw()

    # plot_dual_axis_combined(c_noise_levels_t, c_noise_levels_r, c_trans_ape_model, c_rot_ape_model,
    #                         c_noise_levels_t, c_trans_ape_model_noise, c_noise_levels_r, c_rot_ape_model_noise)
    

    
    
    # The slope tells you how fast the error increases with noise.
    # A smaller slope means the model is more robust to increasing noise.
    # Model A is better than Model B by a factor of slope_b / slope_a.
    #units of slope = (units of error) / (units of noise)
    print('\n\nTranslation Model is better than Noise by a factor of slope_noise_ape / slope_model_ape = ', slope_noise_ape/slope_model_ape)
    print('Rotation Model is better than Noise by a factor of slope_noise_rpe / slope_model_rpe = ', slope_noise_rpe/slope_model_rpe)
    
    


show_this()





p = '/home/eugeniu/xz_final_clouds/Added_noise/z_only_translation_mls_frame/'
path_gt=p+"t_0.050000_r0.001000_GT.txt"
path_noise=p+"t_0.050000_r0.001000_noisy.txt"
path_model=p+"t_0.050000_r0.001000_model.txt"
obj = TrajectoryReader(path_gt, path_noise, path_model)
#obj.plot_data()
# obj.APE_translation()
# obj.APE_rotation()
#obj.RPE_translation()
#obj.plot_errors()
#plt.show()


#-----------------------------------------------------

def _2d_plot():
    # Data as floats
    t = np.array([0.001, 0.001, 0.001, 0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.05, 0.10, 0.15, 0.20, 0.25])
    r = np.array([0.008722, 0.017444, 0.034889, 0.087222, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.017444, 0.034889, 0.052333, 0.069778, 0.087222])

    err_t = np.array([1,1,1,1, 1,2,3,4,5,6,7, 1,2,3,4,5])  # translation error
    err_r = np.array([1,2,3,4, 1,1,1,1,1,1,1, 1,2,3,4,5])  # rotation error

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)

    # Translation error scatter plot
    sc1 = ax1.scatter(t, r, c=err_t, cmap='viridis', s=100, edgecolor='k')
    ax1.set_title("Translation Error vs Noise Levels")
    ax1.set_xlabel("Translation Noise (m)")
    ax1.set_ylabel("Rotation Noise (rad)")
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("Translation Error (m)")

    # Rotation error scatter plot
    sc2 = ax2.scatter(t, r, c=err_r, cmap='plasma', s=100, edgecolor='k')
    ax2.set_title("Rotation Error vs Noise Levels")
    ax2.set_xlabel("Translation Noise (m)")
    ax2.set_ylabel("Rotation Noise (rad)")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Rotation Error (rad)")

    plt.draw()

#_2d_plot()

plt.show()