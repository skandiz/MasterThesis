import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import matplotlib as mpl
import matplotlib.animation
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


import skimage
from csbdeep.utils import normalize
save_path = './data/trial.npz'

@pims.pipeline
def preprocessing(image, x1, y1, x2, y2):
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
    alpha = Image.new('L', (920, 960), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)
    npAlpha = np.array(alpha)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)*npAlpha
    ind = np.where(npImage == 0)
    npImage[ind] = npImage[200, 200]
    npImage = cv2.medianBlur(npImage, 5)
    npImage = normalize(npImage)
    return npImage

data = preprocessing(pims.open('./data/movie.mp4'), 40, 55, 895, 910)
data_preload = list(data[:1000])
run = False
if run:
    from stardist.models import StarDist2D
    from stardist.data import test_image_nuclei_2d
    from stardist.plot import render_label
    from csbdeep.utils import normalize
    from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
    np.random.seed(6)
    lbl_cmap = random_label_cmap()

    # initialize model with versatile fluorescence pretrained weights
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    print(model)

    
    ## TEST
    frame = 0
    labels_test, dict_test = model.predict_instances(data[frame], predict_kwargs = {'verbose':False}) 

    plt.figure(figsize = (10, 5))
    coord, points, prob = dict_test['coord'], dict_test['points'], dict_test['prob']
    ax = plt.subplot(121)
    ax.imshow(img, cmap='gray'); plt.axis('off')
    _draw_polygons(coord, points, prob, show_dist=True)
    ax.set(title = f'Frame {frame}', xlabel='x', ylabel='y')
    ax1 = plt.subplot(122, sharex=ax, sharey=ax)
    ax1.imshow(img, cmap='gray'); plt.axis('off')
    ax1.imshow(labels_test, cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    plt.show()
    

    ## SEGMENT ALL FRAMES AND SAVE THEM IN A NPZ FILE 
    ## COMPUTE THE FEATURES AND SAVE THEM IN A DATAFRAME
    nFrames = 10000
    segm_preload = np.zeros((nFrames, 960, 920), dtype=np.int8)
    area, x, y, prob = [], [], [], []

    for frame in tqdm(range(nFrames)):
        segm_preload[frame], dict_test = model.predict_instances(data[frame], predict_kwargs = {'verbose':False})
        test = skimage.measure.regionprops_table(segm_preload[frame], properties=('centroid', 'area'))
        area += list(test['area'])
        x += list(test['centroid-0'])
        y += list(test['centroid-1'])
        prob += list(dict_test['prob'])
        frames += list(np.ones(len(test))*frame)

    df = pd.DataFrame({'x':x, 'y':y, 'area':area, 'prob':prob})
    df.to_parquet('./data/df.parquet')
    print(df)

    # Save the labeled elements using numpy.savez_compressed
    np.savez_compressed(save_path, labeled_elements=segm_preload)

    loaded_data = np.load(save_path)
    # Access the labeled elements array
    labeled_elements = loaded_data['labeled_elements']

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(labeled_elements[0], cmap='gray')
    plt.show()

    df = pd.read_parquet('./data/df.parquet')
    df = df.loc[(df.prob > 0.88) & (df.area < 3000)]
    if not run: df["frame"] = np.array([frame*np.ones(50) for frame in range(10000)]).flatten()
    print(df)

    #############################################################################################################
    #                                         LINK FEATURES WITH TRACKPY                                        #
    #############################################################################################################

    t = tp.link_df(df, 150, memory = 0, link_strategy = 'hybrid', neighbor_strategy = 'KDTree', adaptive_stop = 1)
    print(t)
    t = t.sort_values(['frame', 'particle'])

    # CREATE COLOR COLUMN AND SAVE DF
    n = max(t.particle)
    print(n)
    random.seed(5)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    for i in range(max(t.particle)+1-n):
        colors.append("#00FFFF")
    c = []
    for p in t.particle:
        c.append(colors[p])
    t["color"] = c
    trajectory = t.copy()
    trajectory.to_parquet('./data/df_linked.parquet')
    print(trajectory)

trajectory = pd.read_parquet('./data/df_linked.parquet')
print(trajectory)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(trajectory.area, trajectory.prob)
ax.set(xlabel='area', ylabel='prob')
plt.show()

fig = plt.figure(figsize = (5, 5))
anim_running = True

def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

def update_graph(frame):
    df = trajectory.loc[(trajectory.frame == frame) , ["x","y","color","area"]]
    for i in range(50):
        graph[i].center = (df.y.values[i], df.x.values[i])
        graph[i].radius = np.sqrt(df.area.values[i]/np.pi)
    graph2.set_data(data_preload[frame])
    title.set_text('Tracking raw - frame = {}'.format(frame))
    return graph

ax = fig.add_subplot(111)
title = ax.set_title('Tracking stardist + trackpy - frame = 0')
ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
df = trajectory.loc[(trajectory.frame == 0), ["x","y","color","area"]]

graph = []
for i in range(50):
    graph.append(ax.add_artist(plt.Circle((df.y.values[i], df.x.values[i]), np.sqrt(df.area.values[i]/np.pi), color = df.color.values[i],\
                                           fill = False, linewidth=1)))
graph2 = ax.imshow(data_preload[0])

fig.canvas.mpl_connect('button_press_event', onClick)
ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(0, max(trajectory.frame), 1), interval = 5, blit=False)
if 1: 
    writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
    ani.save('./results/tracking_stardist.mp4', writer=writer, dpi = 300)
plt.show()

"""
fig = plt.figure(figsize = (8, 8))
def update_graph(frame):
    graph2.set_data(labeled_elements[frame])
    title.set_text('Stardist location - frame = {}'.format(frame))
    return graph2

ax = fig.add_subplot(111)
title = ax.set_title('Stardist location - frame = 0')
ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
graph2 = ax.imshow(labeled_elements[0])

ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(0, nFrames, 1), interval = 50, blit=False)
#writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
#ani.save('./results/video2.mp4', writer=writer, dpi = 300)
plt.show()


"""
