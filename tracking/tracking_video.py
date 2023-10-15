import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation
import matplotlib as mpl
font = {'size'   : 12}
mpl.rc('font', **font)
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
import pims
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw
import cv2

def get_data_preload(startFrame, endFrame, data_preload_path, dataset_name):
    with h5py.File(data_preload_path, 'r') as f:
        dataset = f[dataset_name]
        frameImg = dataset[startFrame:endFrame]
    return frameImg


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
    return image_sharp

def get_smooth_trajs(trajs, nDrops, windLen, orderofPoly):
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    ret = trajs.copy()
    for i in range(nDrops):
        ret.loc[ret.particle == i, "x"] = savgol_filter(trajs.loc[trajs.particle == i].x.values, windLen, orderofPoly)
        ret.loc[ret.particle == i, "y"] = savgol_filter(trajs.loc[trajs.particle == i].y.values, windLen, orderofPoly)    
    return ret

#video_selection = "25b25r"
video_selection = "49b1r"

if video_selection == "49b1r":
    print("Import data 49b_1r ...")
    system_name = "49b-1r system"
    ref = pims.open('../tracking/data/49b1r.mp4')
    h = 920
    w = 960
    xmin = 55
    ymin = 55
    xmax = 880
    ymax = 880
    pxDimension = 90/875 # 90 mm / 875 mm = 0.10285714285714285 mm/mm
    data_preload_path = f'/Volumes/ExtremeSSD/UNI/h5_data_thesis/49b-1r/part1.h5'

    data_path = "../tracking/49b_1r/part1"
    res_path = "./49b_1r/results"
    pdf_res_path = "../../thesis_project/images/49b_1r"
    analysis_data_path = "./49b_1r/analysis_data"
    
    red_particle_idx = np.array([8]).astype(int)
    fps = 10
    maxLagtime = 100*fps # maximum lagtime to be considered in the analysis, 100 seconds

elif video_selection == "25b25r":
    print("Import data 25b_25r ...")
    system_name = "25b-25r system"
    ref = pims.open('../tracking/data/25b25r-1.mp4')
    h = 480
    w = 640
    xmin = 100
    ymin = 35 
    xmax = 530
    ymax = 465
    pxDimension = 90/435 # 90 mm / 435 mm = 0.20689655172413793 mm/mm
    data_preload_path = f'/Volumes/ExtremeSSD/UNI/h5_data_thesis/25b-25r/part1.h5'
    data_path = "../tracking/25b_25r/part1/df_linked.parquet"
    pdf_res_path = "../../thesis_project/images/25b_25r"
    res_path = "./25b_25r/results"
    analysis_data_path = "./25b_25r/analysis_data"
    red_particle_idx = np.sort(np.array([27, 24, 8, 16, 21, 10, 49, 14, 12, 9, 7, 37, 36, 40, 45, 42, 13, 20, 26, 2, 39, 5, 11, 22, 44])).astype(int)
    fps = 30
    maxLagtime = 100*fps # maximum lagtime to be considered in the analysis, 100 seconds = 100 * fps
else:
    raise ValueError("No valid video selection")
    
original_trajectories = pd.read_parquet(data_path + "/df_linked.parquet")
# set radius in mm
original_trajectories.r = original_trajectories.r * pxDimension
nDrops = int(len(original_trajectories.loc[original_trajectories.frame==0]))
frames = original_trajectories.frame.unique().astype(int)
nFrames = len(frames)
print(f"Number of Droplets: {nDrops}")
print(f"Number of Frames: {nFrames} at {fps} fps --> {nFrames/fps:.2f} s")

red_mask = np.zeros(nDrops, dtype=bool)
red_mask[red_particle_idx] = True
colors = np.array(['b' for i in range(nDrops)])
colors[red_particle_idx] = 'r'

# ANALYSIS PARAMETERS

x_diffusive = np.linspace(1, maxLagtime/fps, int((maxLagtime/fps + 1/fps - 1)*fps)) 
x_ballistic = np.linspace(1/fps, 1, int((1-1/fps)*fps)+1)

# WINDOWED ANALYSIS PARAMETERS
window = 300*fps # 320 s
stride = 10*fps # 10 s
print("Windowed analysis args:")
startFrames = np.arange(0, nFrames-window, stride, dtype=int)
endFrames = startFrames + window
nSteps = len(startFrames)
print(f"window of {window/fps} s, stride of {stride/fps} s --> {nSteps} steps")

speed_units = "mm/s"
dimension_units = "mm"
default_kwargs_blue = {"color": "#00FFFF", "ec": (0, 0, 0, 0.6), "density": True}
default_kwargs_red = {"color": "#EE4B2B", "ec": (0, 0, 0, 0.6), "density": True}

if 1:
    trajectories = get_smooth_trajs(original_trajectories, nDrops, int(fps/2), 4)
else:
    trajectories = original_trajectories

data  = preprocessing(ref, w, h, xmin, ymin, xmax, ymax) 

fig = plt.figure(figsize = (8, 8))
def update_graph(frame):
    df = trajectories.loc[(trajectories.frame == frame), ["x", "y", "color", "r"]]
    for i in range(len(df)):
        graph[i].center = (df.x.values[i], df.y.values[i])
        graph[i].radius = df.r.values[i]/pxDimension
    graph2.set_data(data[frame])
    title.set_text(f'49b1r system tracking -- frame = {frame}')
    return graph

ax = fig.add_subplot(111)
title = ax.set_title(f'49b1r system tracking -- frame = {0}')
ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
df = trajectories.loc[(trajectories.frame == 0), ["x", "y", "color", "r"]]

graph = []
for i in range(len(df)):
    graph.append(ax.add_artist(plt.Circle((df.x.values[i], df.y.values[i]), df.r.values[i]/pxDimension, color = df.color.values[i],\
                                           fill = False, linewidth=1)))
graph2 = ax.imshow(data[0])

ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(0, int(max(trajectories.frame)), 1), interval = 5, blit=False)
#ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(0, 100, 1), interval = 5, blit=False)
writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
ani.save(f'./{data_path}/video.mp4', writer=writer, dpi = 300)
plt.close()