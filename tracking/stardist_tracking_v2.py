import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='gray')
import joblib
parallel = joblib.Parallel(n_jobs = -1)
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import random
from stardist import _draw_polygons
from stardist.models import StarDist2D

from tifffile import imsave, imread
from utils import detect_features_frame, detect_features, test_detection, get_frame

model_name = 'modified_2D_versatile_fluo' # stardist model trained for 150 epochs on simulated dataset starting from the pretrained 2D versatile fluo model
model = StarDist2D(None, name = model_name, basedir = './models/')

video_selection = "25b25r-1" #"49b1r"
if video_selection == "25b25r-1":
    xmin, ymin, xmax, ymax = 95, 30, 535, 470    
    save_path = './25b25r/synthetic_simulation/'
elif video_selection == "49b1r":
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

print(f"Video has {n_frames} frames with a resolution of {w}x{h} and a framerate of {fps} fps")

#test_detection(100, n_frames)
detect_features([0, 1], False, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, save_path)

if 0:
    startFrame = 10000
    endFrame = startFrame + 1
    print(f"Processing from {int(startFrame/fps)} s to {int(endFrame/fps)} s")
    sample_frames = np.arange(startFrame, endFrame, 1, dtype=int)

    raw_detection_df = detect_features(frames = sample_frames, test_verb=False)
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