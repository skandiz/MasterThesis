print("Importing libraries...")
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='gray')
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import random
from stardist import _draw_polygons
from stardist.models import StarDist2D
import trackpy as tp
from tifffile import imsave, imread
from utils import detect_features_frame, detect_features, test_detection, get_frame, interpolate_trajectory
import tensorflow as tf

model_name = 'modified_2D_versatile_fluo' # stardist model trained for 150 epochs on simulated dataset starting from the pretrained 2D versatile fluo model
model = StarDist2D(None, name = model_name, basedir = './models/')
print(model.config)
print(len(tf.config.list_physical_devices('GPU')))

video_selection =  '49b1r' #'25b25r-1'
if video_selection == '25b25r-1':
    xmin, ymin, xmax, ymax = 95, 30, 535, 470    
elif video_selection == '49b1r':
    xmin, ymin, xmax, ymax = 20, 50, 900, 930

save_path       = f'./{video_selection}/{model_name}/'
source_path     = f'./data/{video_selection}.mp4'
system_name     = f'{video_selection} system'
nDrops = 50

video = cv2.VideoCapture(source_path)
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video has {n_frames} frames with a resolution of {w}x{h} and a framerate of {fps} fps')

test_verb = False
detect_verb = False
link_verb = False
interp_verb = False

startFrame = 0
endFrame = n_frames

if test_verb: 
    n_samples = 100
    test_detection(n_samples, n_frames, nDrops, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, save_path)

if detect_verb:
    print(f'Processing from {int(startFrame/fps)} s to {int(endFrame/fps)} s')
    sample_frames = np.arange(startFrame, endFrame, 1, dtype=int)
    raw_detection_df = detect_features(sample_frames, False, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, save_path)
    
    n_feature_per_frame = raw_detection_df.groupby('frame').count().x.values
    fig, ax = plt.subplots(2, 2, figsize = (8, 4))
    ax[0, 0].plot(raw_detection_df.frame.unique(), n_feature_per_frame, '.')
    ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
    ax[0, 1].plot(raw_detection_df.r, '.')
    ax[0, 1].set(xlabel = 'Feature index', ylabel = 'Radius [px]', title = 'Radius of features detected')
    ax[1, 0].scatter(raw_detection_df.r, raw_detection_df.eccentricity, s=0.1)
    ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title='R-eccentricity correlation')
    ax[1, 1].scatter(raw_detection_df.r, raw_detection_df.prob, s=0.1)
    ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title='R-Probability correlation')
    plt.tight_layout()
    plt.savefig(save_path + f'raw_features_{model_name}_{startFrame}_{endFrame}.png', dpi = 500)
    plt.close()
else:
    raw_detection_df = pd.read_parquet(save_path + f'raw_detection_25b25r-1_modified_2D_versatile_fluo_{startFrame}_{endFrame}.parquet')
    sample_frames = raw_detection_df.frame.unique()

err_frames = np.where(raw_detection_df.loc[raw_detection_df.frame < 32269].groupby('frame').count().x.values != nDrops)[0]
err_frames.append(np.where(raw_detection_df.loc[raw_detection_df.frame >= 32269].groupby('frame').count().x.values != nDrops-1)[0] + 32269)
print(f'Number of errors: {len(err_frames)} / {len(sample_frames)} --> {len(err_frames)/len(sample_frames)*100:.2f}%')
condition = np.ediff1d(err_frames)
condition[condition == 1] = True
condition[condition != 1] = False
max_n_of_consecutive_errs = max(np.diff(np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:], [True])))[0])[::2])
print(f'Max number of consecutive errors: {max_n_of_consecutive_errs}')

if link_verb:
    print('Linking trajectories...')
    cutoff = 100
    mem = max_n_of_consecutive_errs + 1
    t = tp.link_df(raw_detection_df, cutoff, memory = mem, link_strategy = 'hybrid', neighbor_strategy = 'KDTree', adaptive_stop = 1)
    #print(t)
    t = t.sort_values(['frame', 'particle'])
    trajectories = tp.filter_stubs(t, 25)
    # CREATE COLOR COLUMN AND SAVE DF
    n = max(trajectories.particle)
    print(f'N of droplets: {n + 1}')
    random.seed(5)
    colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    for i in range(max(trajectories.particle)+1-n):
        colors.append('#00FFFF')
    c = []
    for p in t.particle:
        c.append(colors[p])
    trajectories['color'] = c
    trajectories = trajectories.reset_index(drop=True)
    trajectories.to_parquet(save_path + f'raw_tracking_25b25r-1_modified_2D_versatile_fluo_{startFrame}_{endFrame}.parquet', index = False)
else:
    print('Importing linked trajectories...')
    trajectories = pd.read_parquet(save_path + f'raw_tracking_25b25r-1_modified_2D_versatile_fluo_{startFrame}_{endFrame}.parquet')

if interp_verb:
    print('Interpolating trajectories...')
    interp_trajectories = trajectories.groupby('particle').apply(interpolate_trajectory)
    interp_trajectories = interp_trajectories.reset_index(drop=True)
    interp_trajectories['particle'] = interp_trajectories['particle'].astype(int)
    interp_trajectories = interp_trajectories.sort_values(['frame', 'particle'])
    interp_trajectories.to_parquet(save_path + f'interpolated_tracking_25b25r-1_modified_2D_versatile_fluo_{startFrame}_{endFrame}.parquet', index=False)
else:
    print('Importing interpolated trajectories...')
    interp_trajectories = pd.read_parquet(save_path + f'interpolated_tracking_25b25r-1_modified_2D_versatile_fluo_{startFrame}_{endFrame}.parquet')