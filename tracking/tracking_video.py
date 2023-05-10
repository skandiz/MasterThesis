import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['font.size'] = 8
mpl.rc('image', cmap='gray')
import trackpy as tp
tp.quiet()

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
import stardist as star

# Import data
@pims.pipeline
def hough_preprocessing(image, x1, y1, x2, y2):    
    """
    Pims pipeline preprocessing of the image for the HoughCircles function.
    Crops the image to remove the petri dish, converts the image to grayscale and applies a median filter.

    Parameters
    ----------
    image: image
        image to preprocess.
    x1, y1, x2, y2: int
        coordinates of the circle to crop.
    Returns
    -------
    npImage: array
        image to be analyzed.
    """
    npImage = np.array(image)
    #npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2HSV)
    # Create same size alpha layer with circle
    alpha = Image.new('L', (920, 960), 0)

    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)

    # Convert alpha Image to numpy array
    npAlpha = np.array(alpha)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)*npAlpha #npImage[:, :, 1] * npAlpha
    
    ind = np.where(npImage == 0)
    # npImage[200, 200] color of the border to swap with the black
    npImage[ind] = npImage[200, 200]
    npImage = cv2.medianBlur(npImage, 5)
    return npImage

data = hough_preprocessing(pims.open('./data/movie.mp4'), 40, 55, 895, 910)

print("preloading data...")
fig, ax = plt.subplots()
ax.imshow(data[0])
plt.show()
"""
data = [frame for frame in data] # data preload
traj_part = "pre_merge"
hough_df = pd.read_parquet(f"./results/tracking_data/hough/linked_{traj_part}.parquet")

fig = plt.figure(figsize = (8, 8))
def update_graph(frame):
    df = hough_df.loc[(hough_df.frame == frame), ["x", "y", "color", "d"]]
    for i in range(len(df)):
        graph[i].center = (df.x.values[i], df.y.values[i])
        graph[i].radius = df.d.values[i]
    graph2.set_data(data[frame])
    title.set_text('Hough features location & Trackpy linking - frame = {}'.format(frame))
    return graph

ax = fig.add_subplot(111)
title = ax.set_title('Hough features location & Trackpy linking - frame = 0')
ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
df = hough_df.loc[(hough_df.frame == 0), ["x","y","color","d"]]

graph = []
for i in range(len(df)):
    graph.append(ax.add_artist(plt.Circle((df.x.values[i], df.y.values[i]), df.d.values[i], color = df.color.values[i],\
                                           fill = False, linewidth=1)))
graph2 = ax.imshow(data[0])

ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(0, int(max(hough_df.frame)), 1), interval = 5, blit=False)
writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
ani.save('./results/video2.mp4', writer=writer, dpi = 300)
plt.close()
"""