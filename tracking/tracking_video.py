import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib ipympl
import h5py 

plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['font.size'] = 8
mpl.rc('image', cmap='gray')
import trackpy as tp
tp.quiet()

import joblib 

import numpy as np
import pandas as pd
import csv, json
import pims
from PIL import Image, ImageDraw
import cv2

from scipy.optimize import dual_annealing, linear_sum_assignment
from scipy.spatial import distance_matrix
from tqdm import tqdm
import random

import skimage
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois

video_selection = "25b25r"

if video_selection == "49b1r":
    print("Import data 49b_1r ...")
    system_name = "49b-1r system"
    source_path = './data/49b1r.mp4'
    path = './25b_25r'
    data_preload_path = '/Volumes/ExtremeSSD/UNI/h5_data_thesis/49b-1r/part1.h5'
    ref = pims.open(source_path)
    h = 920
    w = 960
    xmin = 55
    ymin = 55
    xmax = 880
    ymax = 880
    fps = 10
    nDrops = 50


elif video_selection == "25b25r":
    print("Import data 25b_25r ...")
    system_name = "25b-25r system"
    source_path = './data/25b25r-1.mp4'
    path = './25b_25r/part1'
    data_preload_path = '/Volumes/ExtremeSSD/UNI/h5_data_thesis/25b-25r/part1.h5'
    ref = pims.open(source_path)
    h = 480
    w = 640
    xmin = 100
    ymin = 35 
    xmax = 530
    ymax = 465
    fps = 30
    nDrops = 50
else:
    raise ValueError("No valid video selection")




@pims.pipeline
def preprocessing(image, w, h, x1, y1, x2, y2):
    """
    Preprocessing function for the data.

    Parameters
    ----------
    image : pims.Frame
        Frame of the video.
    x1 : int
        x coordinate of the top left corner of the ROI. (region of interest)
    y1 : int
        y coordinate of the top left corner of the ROI.
    x2 : int    
        x coordinate of the bottom right corner of the ROI.
    y2 : int    
        y coordinate of the bottom right corner of the ROI.

    Returns
    -------
    npImage : np.array
        Preprocessed image.
    """
    npImage = np.array(image)
    alpha = Image.new('L', (h, w), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)
    npAlpha = np.array(alpha)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)*npAlpha
    ind = np.where(npImage == 0)
    npImage[ind] = npImage[200, 200]
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    # sharpen image https://en.wikipedia.org/wiki/Kernel_(image_processing)
    image_sharp = cv2.filter2D(src=npImage, ddepth=-1, kernel=kernel)
    return npImage

data  = preprocessing(ref, h, w, xmin, ymin, xmax, ymax) 

def get_frame(frame, data_preload_path, dataset_name):
    with h5py.File(data_preload_path, 'r') as f:
        dataset = f[dataset_name]
        frameImg = dataset[frame]
    return frameImg


if 0:
    preprocessed_data = np.zeros((nFrames, data[0].shape[0], data[0].shape[1]), dtype=data[0].dtype)
    for i in tqdm(range(nFrames)):
        preprocessed_data[i] = data[i]
else:
    # Open the HDF5 file in read mode
    with h5py.File(data_preload_path, 'r') as f:
        # Access the dataset within the HDF5 file
        dataset = f['dataset_name']
        preprocessed_data = dataset[framesList[0]:framesList[-1]+1]


fig = plt.figure(figsize = (8, 8))
def update_graph(frame):
    df = trajectories.loc[(trajectories.frame == frame), ["x", "y", "color", "d"]]
    for i in range(len(df)):
        graph[i].center = (df.x.values[i], df.y.values[i])
        graph[i].radius = df.d.values[i]
    graph2.set_data(data[frame])
    title.set_text(f'frame = {frame} --> {int(frame/fps)} s')
    return graph

ax = fig.add_subplot(111)
title = ax.set_title(f'frame = {0} --> {0} s')
ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
df = trajectories.loc[(trajectories.frame == 0), ["x", "y", "color", "radius"]]

graph = []
for i in range(len(df)):
    graph.append(ax.add_artist(plt.Circle((df.x.values[i], df.y.values[i]), df.d.values[i], color = df.color.values[i],\
                                           fill = False, linewidth=1)))
graph2 = ax.imshow(data[0])

ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(0, int(max(trajectories.frame)), 1), interval = 5, blit=False)
writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
ani.save('./results/video2.mp4', writer=writer, dpi = 300)
plt.close()