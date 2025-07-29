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
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import networkx as nx

max_diff = 0.000001  #s

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

from xyzservices import TileProvider
from PIL import Image

font = 14

matplotlib.rc('xtick', labelsize=font)
matplotlib.rc('ytick', labelsize=font)
plt.rcParams.update({'font.size': font})
plt.rc('axes', titlesize=font)     # Set the font size of the title
plt.rc('axes', labelsize=font)     # Set the font size of the x and y labels
plt.rc('xtick', labelsize=font)    # Set the font size of the x tick labels
plt.rc('ytick', labelsize=font)    # Set the font size of the y tick labels
plt.rc('legend', fontsize=font)    # Set the font size of the legend
plt.rc('font', size=font)          # Set the general font size'''

markersize = 8
sns.set_style("whitegrid")
#----------------------------------------------------------------

translation_ENU_to_origin = np.array([4525805.18109165318310260773, 5070965.88124799355864524841,  114072.22082747340027708560])
rotation_ENU_to_origin = np.array([[-0.78386212744029792887, -0.62091317757072628236,  0.00519529438949398702],
                                    [0.62058202788821892337, -0.78367126767620609584, -0.02715310076057271885],
                                    [0.02093112101431114647, -0.01806018100107483967,  0.99961778597386541367]])

R_origin_to_ENU = rotation_ENU_to_origin.T
t_origin_to_ENU = -R_origin_to_ENU @ translation_ENU_to_origin


bbox_to_anchor=(0.5, -0.12)
ncol = 3
def plot_trajectory_(xyz_enu, etrs_tm35fin = 'EPSG:3067', epsg = 3067):
    start = xyz_enu[0]
    print('start:',start)
    end = xyz_enu[len(xyz_enu) - 1]
    # Create a GeoDataFrame with the EVO location point in the ETRS-TM35FIN projection

    middle_index = len(xyz_enu) // 2
    middle_point = xyz_enu[middle_index]

    evo_point = gpd.GeoDataFrame(
        #{'geometry': [Point(start[0],start[1])]},
        {'geometry': [Point(middle_point[0],middle_point[1])]},
        crs=etrs_tm35fin
    )

    evo_area = evo_point.buffer(500)  # 500 meters 

    trajectory_points_etrstm35fin = xyz_enu[::30,:2]

    # Create GeoDataFrame from the points (ETRS-TM35FIN)
    trajectory_gdf = gpd.GeoDataFrame(
        {'geometry': [Point(x, y) for x, y in trajectory_points_etrstm35fin]},
        crs=etrs_tm35fin
    )

    # Reproject the trajectory back to Web Mercator (EPSG:3857) for plotting with the basemap
    trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=epsg)  

    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the area boundary
    evo_area.boundary.plot(ax=axis, color='gray', linewidth=0)
    # Plot the trajectory points in Web Mercator (projected for plotting)
    trajectory_gdf_mercator.plot(ax=axis, color='red', marker='o', markersize=12, label='Trajectory', zorder=10)

    #ctx.add_basemap(axaxis, source=ctx.providers.Esri.WorldImagery, crs=etrs_tm35fin)
    #ctx.add_basemap(axis, source=ctx.providers.OpenStreetMap.Mapnik, crs=etrs_tm35fin)

    
    google_sat = TileProvider(
        name="Google Satellite",
        url="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attribution="Google",
    )
    ctx.add_basemap(axis, source=google_sat, crs=trajectory_gdf.crs)

    axis.set_xlabel('East (m)')
    axis.set_ylabel('North (m)')
    plt.legend()
    plt.grid(False)

    # Set the formatter for x and y axis to avoid scientific notation
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    # Specify the data coordinates where to place the image
    xy = (end[0]-5, end[1])

    image = mpimg.imread('/home/eugeniu/x_vux_mls_als_paper/car_small.png')  # Use a small PNG file
    image = rotate(image, angle=95, reshape=True)
    imagebox = OffsetImage(image, zoom=.2)  # Adjust zoom to make the image smaller or larger

    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        frameon=False,
                        zorder=10)  # No frame around image

    axis.add_artist(ab)

    return fig, axis

def plot_trajectory(xyz_enu, other_traj = [], other_labels=[], etrs_tm35fin = 'EPSG:3067', epsg = 3067):
    start = xyz_enu[0]
    print('start:',start)
    end = xyz_enu[len(xyz_enu) - 1]
    # Create a GeoDataFrame with the EVO location point in the ETRS-TM35FIN projection

    middle_index = len(xyz_enu) // 2
    middle_point = xyz_enu[middle_index]

    evo_point = gpd.GeoDataFrame(
        #{'geometry': [Point(start[0],start[1])]},
        {'geometry': [Point(middle_point[0],middle_point[1])]},
        crs=etrs_tm35fin
    )

    evo_area = evo_point.buffer(500)  # 500 meters 

    trajectory_points_etrstm35fin = xyz_enu[::30,:2]

    # Create GeoDataFrame from the points (ETRS-TM35FIN)
    trajectory_gdf = gpd.GeoDataFrame(
        {'geometry': [Point(x, y) for x, y in trajectory_points_etrstm35fin]},
        crs=etrs_tm35fin
    )

    # Reproject the trajectory back to Web Mercator (EPSG:3857) for plotting with the basemap
    trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=epsg)  

    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the area boundary
    evo_area.boundary.plot(ax=axis, color='gray', linewidth=0)
    # Plot the trajectory points in Web Mercator (projected for plotting)
    trajectory_gdf_mercator.plot(ax=axis, color='red', marker='o', markersize=12, label='GT', zorder=10)

    colors = ["#1f77b4",  
            '#ff7f0e',   
            "#EFD700",  
            "#04f810"]  
    
    markers = ['o','^','D','d']
    for i in range(len(other_traj)):
        traj = other_traj[i]
        l = other_labels[i]

        points_etrstm35fin = traj[::10,:2]
        trajectory_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(x, y) for x, y in points_etrstm35fin]},
            crs=etrs_tm35fin
        )
        trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=epsg)
        trajectory_gdf_mercator.plot(ax=axis, color=colors[i], marker=markers[i], markersize=8, alpha=.7, label=l, zorder=11)


    #ctx.add_basemap(axaxis, source=ctx.providers.Esri.WorldImagery, crs=etrs_tm35fin)
    #ctx.add_basemap(axis, source=ctx.providers.OpenStreetMap.Mapnik, crs=etrs_tm35fin)

    google_sat = TileProvider(
        name="Google Satellite",
        url="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attribution="Google",
    )
    ctx.add_basemap(axis, source=google_sat, crs=trajectory_gdf.crs)

    axis.set_xlabel('East (m)')
    axis.set_ylabel('North (m)')
    plt.legend()
    plt.grid(False)

    # Set the formatter for x and y axis to avoid scientific notation
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    # Specify the data coordinates where to place the image
    xy = (end[0]-5, end[1])

    image = mpimg.imread('/home/eugeniu/x_vux_mls_als_paper/car_small.png')  # Use a small PNG file
    image = rotate(image, angle=95, reshape=True)
    imagebox = OffsetImage(image, zoom=.2)  # Adjust zoom to make the image smaller or larger

    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        frameon=False,
                        zorder=10)  # No frame around image

    axis.add_artist(ab)

    return fig, axis

class TrajectoryReader(object):
    def __init__(self, path_gt, path_model, model_name = '', align = True):
        self.path_gt = path_gt
        self.path_model = path_model
        self.model_name = model_name

        traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        self.traj_gt, self.traj_model = sync.associate_trajectories(traj_gt, traj_model, max_diff)

        if align:
            self.traj_model.align(self.traj_gt, correct_scale=False, correct_only_scale=False, n=-1) 
            #self.traj_model.align_origin(traj_ref=self.traj_gt)
        
        self.T_origin_to_ENU = np.eye(4)
        self.T_origin_to_ENU[:3, :3] = R_origin_to_ENU
        self.T_origin_to_ENU[:3, 3] = t_origin_to_ENU

        self.traj_gt = self.transform_trajectory_to_ENU(self.traj_gt, self.T_origin_to_ENU)
        self.traj_model = self.transform_trajectory_to_ENU(self.traj_model, self.T_origin_to_ENU)

        print("GT:", self.traj_gt.positions_xyz.shape)
        print("Model:", self.traj_model.positions_xyz.shape)

    def transform_trajectory_to_ENU(self, traj: PoseTrajectory3D, T_origin_to_ENU: np.ndarray) -> PoseTrajectory3D:
        new_poses_se3 = []
        for pose in traj.poses_se3:
            T_local = pose
            T_enu = T_origin_to_ENU @ T_local
            new_poses_se3.append(T_enu)

        test_no_fail = 815 # 9999999# 4650
        # if self.model_name in ['LI', 'LI-VUX', 'before LI', 'before LI-VUX', 'test', 'test_now']:
        #     test_no_fail = 4650
        #     test_no_fail = 2500

        return PoseTrajectory3D(
            poses_se3=new_poses_se3[:test_no_fail],
            timestamps=traj.timestamps[:test_no_fail])

    def plot_data(self):
        # fig = plt.figure(figsize=(10, 6))  
        # plot.trajectories(
        #     fig,  
        #     [self.traj_gt, self.traj_model], 
        #     plot.PlotMode.xyz,
        #     title="Aligned Trajectories")
                
        ref = self.traj_gt.positions_xyz
        est = self.traj_model.positions_xyz
        print('self.ape_t_error_vectors:', np.shape(self.ape_t_error_vectors))
        print('ref:', np.shape(ref))

        # Create segments for the reference trajectory (colored by error)
        ref_points = ref[:, :2]  # (N, 2)
        ref_segments = np.stack([ref_points[:-1], ref_points[1:]], axis=1)  # (N-1, 2, 2)
        ref_colors = self.ape_t_error_vectors[:-1]  # (N-1,)

        # Line collection for the reference trajectory
        ref_lc = LineCollection(ref_segments, cmap='viridis', linewidth=2.5,
                                norm=plt.Normalize(vmin=ref_colors.min(), vmax=ref_colors.max()))
        ref_lc.set_array(ref_colors)

        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        est_points = est[:, :2]
        ax.plot(est_points[:, 0], est_points[:, 1], color='black', alpha=0.8, linewidth=1, label="{}".format(self.model_name))
        ax.add_collection(ref_lc)

        # Colorbar for error values
        cbar = fig.colorbar(ref_lc, ax=ax)
        cbar.set_label('Error Norm [m]')

        # Aesthetics
        ax.set_title("Trajectory with Absolute Errors (XY plane) for {}".format(self.model_name))
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        #ax.legend()
        ax.legend(title="Method", loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)

        plt.draw()

    def APE_translation(self):
        print('\nAPE_translation')
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)

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

    def overlap_error(self, est_xyz, label, search_radius = 3.0, min_time_diff = 100, plot = False):
        traj = est_xyz
        xy = traj[:, :2]
        # Build KD-tree on XY coordinates
        tree = cKDTree(xy)
        
        # Step 1: Find overlapping point pairs
        self.overlap_pairs = []
        for i, point in enumerate(xy):
            # Query neighbors within radius (exclude self)
            idxs = tree.query_ball_point(point, r=search_radius)
            # Filter neighbors that are too close in time (to avoid sequential points)
            idxs = [j for j in idxs if abs(j - i) > min_time_diff]
            for j in idxs:
                self.overlap_pairs.append((i, j))

        print('overlap_pairs:', len(self.overlap_pairs))

        # Step 2: Build graph and find connected components
        G = nx.Graph()
        G.add_edges_from(self.overlap_pairs)
        components = list(nx.connected_components(G))
        min_segment_size = 5
        overlap_segments = [sorted(list(c)) for c in components if len(c) >= min_segment_size]
        print(f"Found {len(overlap_segments)} overlapping segments")

        # Step 3: Plot trajectory with highlighted overlaps
        f_map,axis_map = plot_trajectory(est_xyz, etrs_tm35fin = 'EPSG:3067', epsg = 3067)

        plt.draw()

        cmap = plt.cm.get_cmap('tab10')

        cols_ = np.array(['tab:orange', 'tab:blue'])
        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(xy[:, 0], xy[:, 1], label='Trajectory: {}'.format(label), color='gray', alpha=0.5)
            colors = plt.cm.get_cmap('tab10', len(overlap_segments))
            i = 0
            for idx, segment in enumerate(overlap_segments):
                pts = xy[segment]
                plt.plot(pts[:, 0], pts[:, 1], '.', color=colors(idx), label=f'Overlapped segment {idx+1}')
                print('idx:', idx)
                axis_map.plot(pts[::30, 0], pts[::30, 1], '*', markersize=12, alpha=0.7, color=cols_[i], label=f'Overlapped segment {idx+1}')
                i+=1

            plt.xlabel('East [m]')
            plt.ylabel('North [m]')
            plt.title('Trajectory with Overlapping Segments Highlighted')
            #plt.legend()
            plt.legend(title="Method", loc='upper center', bbox_to_anchor=bbox_to_anchor,ncol = ncol, fancybox=True, shadow=True)
            plt.axis('equal')
            plt.grid(False)
            plt.draw()

            axis_map.legend()
            plt.draw()

        #-----------------------------------------------------------------------
        if plot: # and False:
            fig_3d = plt.figure(figsize=(12, 8))
            ax3d = fig_3d.add_subplot(111, projection='3d')

            # Plot full trajectory in gray
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='gray', alpha=0.3, label='Full Trajectory')

        
        self.segment_passes = []  # Stores tuples: (forward_pass_idxs, backward_pass_idxs)

        for idx, segment in enumerate(overlap_segments):
            segment = sorted(segment)
            # Use clustering in index space to separate passes
            diffs = np.diff(segment)
            split_idx = np.argmax(diffs) + 1  # split at largest gap

            forward_pass = segment[:split_idx]
            backward_pass = segment[split_idx:]

            self.segment_passes.append((forward_pass, backward_pass))
            print(f"Segment with {len(segment)} points -> Forward: {len(forward_pass)}, Backward: {len(backward_pass)}")

            pts1 = traj[forward_pass]
            pts2 = traj[backward_pass]

            #pts1 = pts1[::10]
            #pts2 = pts2[::10]

            if plot: # and False:
                color1 = cmap((2 * idx) % 10)
                color2 = cmap((2 * idx + 1) % 10)

                ax3d.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], 
                        color=color1, marker='o', alpha=0.7, s=3,
                        label=f'Segment {idx+1} - Forward Pass')

                ax3d.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], 
                                color=color2, marker='^', alpha=0.7, s=3,
                                label=f'Segment {idx+1} - Backward Pass')

                ax3d.set_xlabel("East [m]")
                ax3d.set_ylabel("North [m]")
                ax3d.set_zlabel("Height [m]")
                ax3d.set_title(f"3D Trajectory with Overlapping Segments ({label})")
                ax3d.grid(True)

        if plot: # and False:
            def set_axes_equal(ax):
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])

                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)

                radius = 0.5 * max(x_range, y_range, z_range)
                ax.set_xlim3d([x_middle - radius, x_middle + radius])
                ax.set_ylim3d([y_middle - radius, y_middle + radius])
                ax.set_zlim3d([z_middle - radius, z_middle + radius])

            set_axes_equal(ax3d)
            ax3d.legend(title="Method",loc='best')



            plt.draw()

        #--------------------------------------------------------------------------------
        # Step 4: Compare Z-values using NN match between forward/backward pass
        num_plots = len(self.segment_passes)
        if num_plots == 0:
            print("No valid segments for z-axis error plotting.")
            return 0

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=False)
        if num_plots == 1:
            axes = [axes]  # ensure it's iterable

        avg_z = 0
        number = 0
        all_z_diffs = []
        for idx, ((forward_pass, backward_pass), ax) in enumerate(zip(self.segment_passes, axes)):

            if len(forward_pass) < 5 or len(backward_pass) < 5:
                ax.set_title(f"Segment {idx+1}: Skipped (too small)")
                continue

            # Get 3D points
            fwd_pts = traj[forward_pass]
            bwd_pts = traj[backward_pass]

            # KD-Tree to find nearest backward point for each forward point
            bwd_tree = cKDTree(bwd_pts[:, :3])
            distances, indices = bwd_tree.query(fwd_pts[:, :3], distance_upper_bound=1.0)

            valid = distances != np.inf
            fwd_valid = fwd_pts[valid]
            bwd_valid = bwd_pts[indices[valid]]

            z1 = fwd_valid[:, 2]
            z2 = bwd_valid[:, 2]
            z_diff = z2 - z1

            rmse = np.sqrt(np.mean(z_diff ** 2))
            mean = np.mean(np.abs(z_diff))
            all_z_diffs.append(np.abs(z_diff))
            avg_z += mean
            number+= 1
            print(f"Segment {idx+1}: Matched {len(z_diff)} pairs — z-RMSE = {rmse:.2f} m, mean = {mean:.2f} m")

            ax.plot(z1, label='Forward Pass z-axis', linestyle='--', color='tab:blue')
            ax.plot(z2, label='Backward Pass z-axis', linestyle='-.', color='tab:orange')
            ax.fill_between(range(len(z_diff)), z1, z2, color='gray', alpha=0.3, label='z-axis Error')
            ax.set_ylabel("z-axis [m]")
            #ax.set_xlabel("Points")
            ax.set_title(f"Segment {idx+1} — Mean Δz-axis: {mean:.2f} m, RMSE: {rmse:.2f} m")
            ax.grid(True)
        
        axes[1].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)

        fig.suptitle(f"z-axis Comparison Across Overlapping Segments ({label})")
        #fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.draw()

        all_z_diffs_concatenated = np.concatenate(all_z_diffs)
        return avg_z/number, all_z_diffs_concatenated

def overlap_error(est_xyz, label, segment_passes,  plot = False):    
    # if label in ['LI', 'LI-VUX']:
    #     return 0, [0]
    
    #segment_passes = []  # Stores tuples: (forward_pass_idxs, backward_pass_idxs)

    num_plots = len(segment_passes)
    if num_plots == 0:
        print("No valid segments for Z error plotting.")
        return 0, [0]

    max_index = est_xyz.shape[0] 
    
    

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=False)
    if num_plots == 1:
        axes = [axes]  # ensure it's iterable

    avg_z = 0
    number = 0
    all_z_diffs = []
    for idx, ((forward_pass, backward_pass), ax) in enumerate(zip(segment_passes, axes)):

        if len(forward_pass) < 5 or len(backward_pass) < 5:
            ax.set_title(f"Segment {idx+1}: Skipped (too small)")
            continue
        
        if max(forward_pass) < len(est_xyz) and max(backward_pass) < len(est_xyz):
            fwd_pts = est_xyz[forward_pass]
            bwd_pts = est_xyz[backward_pass]
        else:
            print("One or more index lists contain out-of-bounds indices.")
            return 0, [0]
            continue


        # Get 3D points
        # fwd_pts = est_xyz[forward_pass]
        # bwd_pts = est_xyz[backward_pass]

        # KD-Tree to find nearest backward point for each forward point
        bwd_tree = cKDTree(bwd_pts[:, :3])
        distances, indices = bwd_tree.query(fwd_pts[:, :3], distance_upper_bound=1.0)

        valid = distances != np.inf
        fwd_valid = fwd_pts[valid]
        bwd_valid = bwd_pts[indices[valid]]

        z1 = fwd_valid[:, 2]
        z2 = bwd_valid[:, 2]
        z_diff = z2 - z1

        rmse = np.sqrt(np.mean(z_diff ** 2))
        mean = np.mean(np.abs(z_diff))
        all_z_diffs.append(np.abs(z_diff))
        avg_z += mean
        number+= 1
        print(f"Segment {idx+1}: Matched {len(z_diff)} pairs — z-RMSE = {rmse:.2f} m, mean = {mean:.2f} m")

        ax.plot(z1, label='Forward Pass z-axis',linestyle='--', color='tab:blue')
        ax.plot(z2, label='Backward Pass z-axis', linestyle='-.', color='tab:orange')
        ax.fill_between(range(len(z_diff)), z1, z2, color='gray', alpha=0.3, label='z-axis Error')
        ax.set_ylabel("z-axis [m]")
        ax.set_xlabel("Points")
        ax.set_title(f"Segment {idx+1} — Mean Δz-axis: {mean:.2f} m, RMSE: {rmse:.2f} m")
        ax.grid(True)
    
    axes[1].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)

    #fig.suptitle(f"z-axis Comparison Across Overlapping Segments ({label})")
    plt.draw()

    all_z_diffs_concatenated = np.concatenate(all_z_diffs)
    return avg_z/number, all_z_diffs_concatenated



path_gt =  "/home/eugeniu/z_tighly_coupled/ref/MLS.txt"

methods = {
    'GT' : '/home/eugeniu/z_tighly_coupled/ref',

    '(ppk)GNSS'             : '/home/eugeniu/z_tighly_coupled/0',
    'LI'                    : '/home/eugeniu/z_tighly_coupled/1',
    'LI-VUX'                : '/home/eugeniu/z_tighly_coupled/2',
    'LI-VUX-ALS(l-coupled)' : '/home/eugeniu/z_tighly_coupled/3',
    'LI-VUX-ALS(t-coupled)' : '/home/eugeniu/z_tighly_coupled/4',
    'LI-VUX-(raw)GNSS'      : '/home/eugeniu/z_tighly_coupled/5',
    'LI-VUX-(ppk)GNSS'      : '/home/eugeniu/z_tighly_coupled/6',
    'LI-VUX-sparse-ALS(l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    'LI-VUX-sparse-ALS(t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',
}

#-robust comes from plane uncertanties 

methods = {
    #'GT' : '/home/eugeniu/z_tighly_coupled/ref',
    'test'                         : '/home/eugeniu/z_tighly_coupled/test1',
    'test2'                         : '/home/eugeniu/z_tighly_coupled/test2',
    'test_now'                         : '/home/eugeniu/z_tighly_coupled/test',

    #'before LI'                    : '/home/eugeniu/z_tighly_coupled/1',
    #'before LI-VUX'                : '/home/eugeniu/z_tighly_coupled/2',
    
    'LI'                        : '/home/eugeniu/z_tighly_coupled/1.1',    #robust
    #'LI-VUX'                    : '/home/eugeniu/z_tighly_coupled/2.1', #robust

    # '(ppk)GNSS'             : '/home/eugeniu/z_tighly_coupled/0',
    # 'LI-VUX-(raw)GNSS'      : '/home/eugeniu/z_tighly_coupled/5',
    # 'LI-VUX-(ppk)GNSS'      : '/home/eugeniu/z_tighly_coupled/6',

    # 'LI-VUX-ALS(l-coupled)' : '/home/eugeniu/z_tighly_coupled/3',
    # 'LI-VUX-ALS(t-coupled)' : '/home/eugeniu/z_tighly_coupled/4',
    
    # 'LI-VUX-sparse-ALS(l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    # 'LI-VUX-sparse-ALS(t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',
}

methods_data = {
    'LI'                    : ['#1f77b4','A'],    #robust
    'LI-VUX'                 : ['#ff7f0e','B'],#robust

    '(ppk)GNSS'             : ['#2ca02c','C'],
    'LI-VUX-(raw)GNSS'      : ['#7f7f7f','D'],
    'LI-VUX-(ppk)GNSS'      : ['#9467bd','E'],

    'LI-VUX-ALS(l-coupled)' : ['#8c564b','F'],
    'LI-VUX-ALS(t-coupled)' : ['#e377c2','G'],
    
    'LI-VUX-sparse-ALS(l-coupled)' : ['#bcbd22','H'],
    'LI-VUX-sparse-ALS(t-coupled)' : ['#17becf','K'],

    'GT' : ['#d62728','L'],

    'before LI' : ['#EFD700','M'],
    'before LI-VUX' : ['#04f810','N'],

    'test' : ["#648099",'Z'],
    'test2' : ["#26391B",'Z'],
    'test_now' : ["#a86b40",'S'],
}


colors = ['tab:brown', 'tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'cyan', 'lime','orange','gray']
colors2 = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#7f7f7f',  # Gray
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#bcbd22',  # Yellow-green
    '#17becf'   # Cyan
    '#d62728',  # Red
]
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', '-', '--', ':',]
lab = ['A','B','C','D','E','F','G','H','K','X']
lt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

obj_gt = TrajectoryReader(path_gt, path_gt)


if True:
    est_xyz = obj_gt.traj_model.positions_xyz

    xyz_1 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/1.1/MLS.txt', 'LI', False).traj_model.positions_xyz
    xyz_2 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/2.1/MLS.txt', 'LI-VUX', False).traj_model.positions_xyz

    before_xyz_1 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/1/MLS.txt', 'before LI', False).traj_model.positions_xyz
    before_xyz_2 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/2/MLS.txt', 'before LI-VUX', False).traj_model.positions_xyz

    other_traj = [xyz_1, xyz_2, before_xyz_1, before_xyz_2]
    other_labels = ['LI', 'LI-VUX', 'before LI', 'before LI-VUX']

    f_map,axis_map = plot_trajectory(est_xyz, other_traj=other_traj, other_labels = other_labels)
    plt.show()


obj_gt.overlap_error(obj_gt.traj_model.positions_xyz, "GT", plot = True)
all_traj = [obj_gt.traj_gt]
# 3D Plot
fig_3d = plt.figure(figsize=(10, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_title("3D Trajectories")
ax_3d.set_xlabel("East [m]")
ax_3d.set_ylabel("North [m]")
ax_3d.set_zlabel("Height [m]")

# 2D Plot (XY Plane)
fig_2d = plt.figure(figsize=(10, 7))
ax_2d = fig_2d.add_subplot(111)
ax_2d.set_title("XY Plane Trajectories")
ax_2d.set_xlabel("East [m]")
ax_2d.set_ylabel("North [m]")

data_ape_t = {}
data_ape_r = {}
data_rpe_t = {}
data_rpe_r = {}
data_z_overlap = {}
data_z_overlap2 = {}

positions = all_traj[0].positions_xyz 
ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],label="GT", color="black")
ax_2d.plot(positions[:, 0], positions[:, 1],label="GT", color="black",) 
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
    obj = TrajectoryReader(path_gt, path_model, label)

    obj.APE_translation()
    obj.APE_rotation()
    obj.RPE_translation()
    obj.RPE_rotation()

    
    if False:    
        row = "" #table_data[label]
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['mean'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['median'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['rmse'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['std'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.rpe_statistics_t['mean'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.rpe_statistics_t['median'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.rpe_statistics_t['rmse'],3)} }} \n"
        row += f"& {{{round(obj.rpe_statistics_t['std'],3)}}}" 
        print('\n\n',label,'\n',row,'\n\n')


    #obj.plot_data()

    all_traj.append(obj.traj_model)

    data_ape_t[label] = obj.ape_t_error_vectors
    data_ape_r[label] = obj.ape_r_error_vectors
    data_rpe_t[label] = obj.rpe_t_error_vectors
    data_rpe_r[label] = obj.rpe_r_error_vectors

    positions = obj.traj_model.positions_xyz  # N x 3 numpy array

    ax_3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        label=label, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)]
    )

    ax_2d.plot(
        positions[:, 0], positions[:, 1],
        label=label, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)]
    )

    z_overlap_error, all_z_diffs = overlap_error(obj.traj_model.positions_xyz, label, obj_gt.segment_passes,  plot = False)
    data_z_overlap[label] = z_overlap_error
    data_z_overlap2[label] = all_z_diffs



#ax_3d.legend()
#ax_2d.legend()
ax_3d.legend(title="Method",loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)
ax_2d.legend(title="Method",loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)

ax_3d.grid(True)
ax_2d.grid(True)
plt.draw()

def plot_box(data, metric = '',  show_legend = True):
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
        legend_handles.append(mpatches.Patch(color=color, label = methods_data[label][1]+" : "+label))
        labels_local.append(methods_data[label][1])
        ind += 1
    for median_line in box['medians']:
        median_line.set_alpha(0)  # or median_line.set_visible(False)
        #median_line.set(color='grey', linewidth=2, linestyle='dotted')

    for mean_line in box['means']:
        mean_line.set(color='black', linewidth=2, linestyle='--')

    for line in box['whiskers'] + box['caps']:
        line.set(color='black', linewidth=1.2)

    plt.title(f'Box plot of {metric}')
    plt.ylabel(metric)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    if show_legend:
        plt.legend(handles=legend_handles, title="Method", loc='upper center', bbox_to_anchor=bbox_to_anchor,
            ncol = ncol, fancybox=True, shadow=True)
    plt.grid(True)
    #plt.tight_layout()
    #plt.xticks(rotation=90)
    #plt.xticks([])
    plt.xticks(lt_local, labels_local) 
    # if first_got_legend:
    #     plt.legend().set_visible(False)
    plt.draw()

    plt.figure(figsize=(10, 5))

    values = [data[label] for label in labels]

    i=0
    for patch, color, label in zip(box['boxes'], colors_, labels):
        #for i, label in enumerate(data):
        values = np.sort(data[label])
        cdf = np.linspace(0, 1, len(values))
        plt.plot(values, cdf, label=label, color = color, linestyle = linestyles[i])

        #plt.title(f'Cumulative Distribution of {metric_name}')
        plt.xlabel(metric)
        plt.ylabel("Cumulative Probability")
        if show_legend:
            plt.legend(title="Method", loc='upper center', bbox_to_anchor=bbox_to_anchor,
            ncol = ncol, fancybox=True, shadow=True)
        plt.grid(True)
        #plt.tight_layout()
        plt.draw()
        i+=1

def plot_z_overlap(data, metric_name=''):
    plt.figure(figsize=(10, 6))

    labels = list(data.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(data)
    
    for i, (label, value) in enumerate(data.items()):
        plt.bar(i , value, width, label=label)

        print(label, ':', value)

    plt.ylabel(metric_name)
    #plt.xticks(x, labels)
    plt.xticks([])
    plt.legend(title="Method", loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)
    #plt.xticks(rotation=90)
    #plt.tight_layout()
    plt.grid(True)
        
    plt.draw()


plot_box(data_ape_t, 'APE translation (m)', show_legend = False)
#plot_box(data_ape_r, 'APE rotation (deg)')
plot_box(data_rpe_t, 'RPE translation (m)', show_legend = True)
#plot_box(data_rpe_r, 'RPE rotation (deg)')

# label = "GT"
# z_overlap_error, all_z_diffs = overlap_error(obj_gt.traj_model.positions_xyz, label, obj_gt.segment_passes,  plot = False)
# data_z_overlap[label] = z_overlap_error
# data_z_overlap2[label] = all_z_diffs

# data_z_overlap2.pop('LI')
# data_z_overlap2.pop('LI-VUX')

plot_box(data_z_overlap2, 'Overlap z-axis error (m)')

#plot_z_overlap(data_z_overlap, 'Mean z error overlap')

plt.show()