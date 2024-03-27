import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D
from csbdeep.utils import Path, normalize
from tifffile import imread
from glob import glob
from tracking_utils import plot_img_label, random_fliprot, random_intensity_change, augmenter
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) # this goes *before* tf import
import tensorflow as tf
np.random.seed(42)
lbl_cmap = random_label_cmap()
from tracking_utils import *
import trackpy as tp
import pandas as pd
import scipy
import sklearn
import random

import yupi.stats as ys
from yupi import Trajectory, WindowType, DiffMethod
def get_trajs(nDrops, trajs, fps, subsample_factor):
    Trajs = []
    for i in range(0, nDrops):
        p = trajs.loc[trajs.particle == i, ['x','y']][::subsample_factor]
        Trajs.append(Trajectory(p.x, p.y, dt = 1/fps*subsample_factor, traj_id=i, diff_est={'method':DiffMethod.LINEAR_DIFF, 
                                                                                  'window_type': WindowType.CENTRAL}))
    return Trajs

run_detection_verb = False
run_linking_verb = False


resolution = 1000
sc = int(resolution/500)
fps = 100
simulated_trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_{fps}_fps_r_decay_r_gaussian_2.parquet')
imgs_path = f'./simulation/synthetic_dataset_{fps}_fps_r_decay_r_gaussian_2/image/'
frames = np.arange(0, 70000, 1)

simulated_trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_100_fps_r_decay_r_gaussian_2.parquet')
simulated_trajectories = simulated_trajectories.loc[simulated_trajectories.frame < 70000]
simulated_trajectories['particle'] = simulated_trajectories['label']
# offset the simulated trajectories to the center of the image
simulated_trajectories.loc[:, ['x','y']] = simulated_trajectories.loc[:, ['x','y']] + 250
simulated_trajectories.loc[:, ['x','y', 'r']] = simulated_trajectories.loc[:, ['x','y', 'r']] * sc

if run_detection_verb:
    model_name = 'modified_2D_versatile_fluo_synthetic_dataset_100_fps_r_decay_r_gaussian_only_optimization' 
    model = StarDist2D(None, name = model_name, basedir = './models/')
    raw_detection_df = detect_instances_from_images(frames, imgs_path, resolution, model)
    raw_detection_df.to_parquet(f'./simulation/stardist_detection_{fps}_fps_r_decay_r_gaussian2.parquet')
else:
    raw_detection_df = pd.read_parquet(f'./simulation/stardist_detection_{fps}_fps_r_decay_r_gaussian2.parquet')



n_feature_per_frame = raw_detection_df.groupby('frame').count().x.values
fig, ax = plt.subplots(2, 2, figsize = (8, 4))
ax[0, 0].plot(raw_detection_df.frame.unique(), n_feature_per_frame, '.')
ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
ax[0, 1].plot(raw_detection_df.r, '.')
ax[0, 1].set(xlabel = 'Instance index', ylabel = 'Radius [px]', title = 'Radius of instances detected')
ax[1, 0].scatter(raw_detection_df.r, raw_detection_df.eccentricity, s=0.1)
ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title='R-eccentricity correlation')
ax[1, 1].scatter(raw_detection_df.r, raw_detection_df.prob, s=0.1)
ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title='R-Probability correlation')
plt.tight_layout()
plt.show()

n_feature_per_frame = simulated_trajectories.groupby('frame').count().x.values
fig, ax = plt.subplots(2, 2, figsize = (8, 4))
ax[0, 0].plot(simulated_trajectories.frame.unique(), n_feature_per_frame, '.')
ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
ax[0, 1].plot(simulated_trajectories.r, '.')
ax[0, 1].set(xlabel = 'Instance index', ylabel = 'Radius [px]', title = 'Radius of instances detected')
ax[1, 0].axis('off')
ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title='R-eccentricity correlation')
ax[1, 1].axis('off')
ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title='R-Probability correlation')
plt.tight_layout()
plt.show()

if run_linking_verb:
    print('Linking stardist_trajectories...')
    cutoff = 100
    t = tp.link_df(raw_detection_df, cutoff, memory = 1, link_strategy = 'hybrid', neighbor_strategy = 'KDTree', adaptive_stop = 1)
    #print(t)
    t = t.sort_values(['frame', 'particle'])
    stardist_trajectories = t#tp.filter_stubs(t, 25)
    # CREATE COLOR COLUMN AND SAVE DF
    n = max(stardist_trajectories.particle)
    print(f'N of droplets: {n + 1}')
    random.seed(5)
    colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    for i in range(max(stardist_trajectories.particle)+1-n):
        colors.append('#00FFFF')
    c = []
    for p in t.particle:
        c.append(colors[p])
    stardist_trajectories['color'] = c
    stardist_trajectories = stardist_trajectories.reset_index(drop=True)
    stardist_trajectories.to_parquet(f'./simulation/stardist_trajectories_{fps}_fps_r_decay_r_gaussian2.parquet')
else:
    stardist_trajectories = pd.read_parquet(f'./simulation/stardist_trajectories_{fps}_fps_r_decay_r_gaussian2.parquet')

# match simulated and stardist trajectories particle ids
dist_matrix = scipy.spatial.distance_matrix(stardist_trajectories.loc[stardist_trajectories.frame == 0 ,['x','y']], simulated_trajectories.loc[simulated_trajectories.frame == 0, ['x','y']])                                    
id_assignment = scipy.optimize.linear_sum_assignment(dist_matrix)
stardist_trajectories['particle'] = np.array([id_assignment[1] for i in range(len(frames))]).flatten()
stardist_trajectories = stardist_trajectories.sort_values(['frame', 'particle'])
simulated_trajectories = simulated_trajectories.sort_values(['frame', 'particle'])

df1 = simulated_trajectories.loc[simulated_trajectories.frame == 0]
df2 = stardist_trajectories.loc[stardist_trajectories.frame == 0]
img = imread(imgs_path + f'frame_{0}_{resolution}_resolution.tif')

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 6))
ax.imshow(img, cmap='gray', vmin = 0, vmax = 255)
for i in range(len(df1)):
    ax.text(df1.iloc[i].x, df1.iloc[i].y, str(int(df1.iloc[i].particle)), color = 'w')
ax.set(xlabel = 'Simulated')
ax1.imshow(img, cmap='gray', vmin = 0, vmax = 255)
for i in range(len(df2)):
    ax1.text(df2.iloc[i].x, df2.iloc[i].y, str(df2.iloc[i].particle), color = 'w')
ax1.set(xlabel = 'Stardist')
plt.show()


turn_angles_bins = np.linspace(-np.pi, np.pi, 101)
speed_bins = np.arange(0, 100, .2)


subsample = False
if subsample:
    simulated_trajectories2 = simulated_trajectories.loc[simulated_trajectories.frame.isin(simulated_trajectories.frame.unique()[::3])]
    stardist_trajectories2 = stardist_trajectories.loc[stardist_trajectories.frame.isin(stardist_trajectories.frame.unique()[::3])]
else:
    simulated_trajectories2 = simulated_trajectories
    stardist_trajectories2 = stardist_trajectories

for polyorder in range(2, 10):
    windList = np.append(0, np.arange(polyorder + 1, 50, 1))
    MSE = np.zeros(len(windList))
    MSE_turn_angles = np.zeros(len(windList))
    MSE_speed = np.zeros(len(windList))

    simulated_positions = simulated_trajectories2.loc[:, ['x', 'y']].values.reshape(len(simulated_trajectories2.frame.unique()), 50, 2)
    stardist_positions = stardist_trajectories2.loc[:, ['x', 'y']].values.reshape(len(stardist_trajectories2.frame.unique()), 50, 2) 
    MSE[0] = np.mean((simulated_positions - stardist_positions)**2)

    simulation_turning_angles = ys.turning_angles_ensemble(get_trajs(50, simulated_trajectories2, 30, 1), centered = True)
    simulation_turning_angles_counts, _ = np.histogram(simulation_turning_angles, bins = turn_angles_bins, density = True)
    stardist_turning_angles = ys.turning_angles_ensemble(get_trajs(50, stardist_trajectories2, 30, 1), centered = True)
    stardist_turning_angles_counts, _ = np.histogram(stardist_turning_angles, bins = turn_angles_bins, density = True)
    MSE_turn_angles[0] = np.mean((simulation_turning_angles_counts - stardist_turning_angles_counts)**2)

    simulation_speed = ys.speed_ensemble(get_trajs(50, simulated_trajectories2, 30, 1), step=1)
    simulation_speed_counts, _ = np.histogram(simulation_speed, bins = speed_bins, density = True)
    stardist_speed = ys.speed_ensemble(get_trajs(50, stardist_trajectories2, 30, 1), step=1)
    stardist_speed_counts, _ = np.histogram(stardist_speed, bins = speed_bins, density = True)
    MSE_speed[0] = np.mean((simulation_speed_counts - stardist_speed_counts)**2)

    i = 1
    for wind in tqdm(windList[1:]):
        stardist_smooth_trajs = get_smooth_trajs(stardist_trajectories2, wind, polyorder)
        smoooth_stardist_positions = stardist_smooth_trajs.loc[:, ['x', 'y']].values.reshape(len(stardist_smooth_trajs.frame.unique()), 50, 2)
        MSE[i] = np.mean((simulated_positions - smoooth_stardist_positions)**2)
        
        smooth_stardist_turning_angles = ys.turning_angles_ensemble(get_trajs(50, stardist_smooth_trajs, 30, 1), centered= True)
        smooth_stardistis_turning_angles_counts, _ = np.histogram(smooth_stardist_turning_angles, bins = turn_angles_bins, density = True)
        MSE_turn_angles[i] = np.mean((simulation_turning_angles_counts - smooth_stardistis_turning_angles_counts)**2)

        smooth_stardist_speed = ys.speed_ensemble(get_trajs(50, stardist_smooth_trajs, 30, 1), step=1)
        smooth_stardistis_speed_counts, _ = np.histogram(smooth_stardist_speed, bins = speed_bins, density = True)

        MSE_speed[i] = np.mean((simulation_speed_counts - smooth_stardistis_speed_counts)**2)
        i += 1

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize = (14, 5))
    ax.plot(windList, MSE_turn_angles)
    ax.set(xlabel = 'Window size', ylabel = 'MSE', title = 'MSE of turning angles')
    ax.grid()
    ax1.plot(windList, MSE)
    ax1.set(xlabel = 'Window size', ylabel = 'MSE', title = 'MSE of positions')
    ax1.grid()
    ax1.set(ylim = (0, 1))
    ax2.plot(windList, MSE_speed)
    ax2.set(xlabel = 'Window size', ylabel = 'MSE', title = 'MSE of speed')
    ax2.grid()
    plt.suptitle(f'MSE of stardist smoothet with polynomial order: {polyorder}')
    plt.tight_layout()
    if subsample:
        plt.savefig(f'./simulation/smoothing_analysis/stardist_smoothed_MSE_polyorder_{polyorder}.png')
    else:
        plt.savefig(f'./simulation/smoothing_analysis/stardist_smoothed_MSE_polyorder_{polyorder}.png')
    plt.close()