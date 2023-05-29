import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib.animation
plt.rcParams.update({'font.size': 8})

import numpy as np
import pandas as pd
import random

from scipy.spatial import KDTree, cKDTree
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import joblib
import time
from numba_progress import ProgressBar
from tqdm import tqdm
import trackpy as tp
from numba import njit, prange

from yupi import Trajectory
import yupi.graphics as yg
import yupi.stats as ys

from utility import get_imsd, get_imsd_windowed, get_emsd, get_emsd_windowed, fit_hist, MB_2D,\
                    normal_distr, lorentzian_distr, get_trajs, speed_windowed, theta_windowed, \
                    get_smooth_trajs, get_velocities

from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx 
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN

show_verb = False
save_verb = True
anim_show_verb = False

traj_verb = "stardist"

if traj_verb == "trackpy": 
    rawTrajs = pd.read_parquet("../tracking/results/tracking_data/trackpy_pre_merge.parquet")
    nDrops = int(len(rawTrajs.loc[rawTrajs.frame==0]))
    red_particle_idx = 17
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    smoothTrajs = get_smooth_trajs(rawTrajs, nDrops, 30, 2)

    res_path = "results"
    analysis_data_path = "analysis_data"

elif traj_verb == "hough":
    rawTrajs = pd.read_parquet("../tracking/results/tracking_data/hough/linked_pre_merge.parquet")#pd.read_parquet("../tracking/results/tracking_data/tracking_hough_trackpy_linking.parquet")
    nDrops = int(len(rawTrajs.loc[rawTrajs.frame==0]))
    red_particle_idx = 17
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    smoothTrajs = get_smooth_trajs(rawTrajs, nDrops, 30, 2)
    
    res_path = "hough_results"
    analysis_data_path = "hough_analysis_data"

elif traj_verb == "stardist":
    rawTrajs = pd.read_parquet("../tracking/stardist_res/sharp/df_linked.parquet")
    nDrops = int(len(rawTrajs.loc[rawTrajs.frame==0]))
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    smoothTrajs = get_smooth_trajs(rawTrajs, nDrops, 30, 2)
    red_particle_idx = 8

    res_path = "stardist_results" 
    analysis_data_path = "stardist_analysis_data"
else:
    raise ValueError("traj_verb must be either 'trackpy' or 'hough'")


colors = rawTrajs.loc[rawTrajs.frame == 0, 'color'].values
nFrames = int(max(rawTrajs.frame) + 1)
fps = 10
print(f"Number of Droplets: {nDrops}")
print(f"Number of Frames: {nFrames} at {fps} fps --> {nFrames/10:.2f} s")

# WINDOWED ANALYSIS PARAMETERS
window = 3200 # 320 s
stride = 100 # 10 s
print("Windowed analysis args:")

startFrames = np.arange(0, nFrames-window, stride, dtype=int)
endFrames = startFrames + window
nSteps = len(startFrames)
print(f"window of {window/10} s, stride of {stride/10} s --> {nSteps} steps")

# step 10 with a 10 fps video --> 1 s
units = "px/s"
default_kwargs_blue = {"color": "#00FFFF", "ec": (0, 0, 0, 0.6), "density": True}
default_kwargs_red = {"color": "#EE4B2B", "ec": (0, 0, 0, 0.6), "density": True}


#preprocessed_data = np.load('../tracking/stardist_res/sharp/preprocessed_pre_merge.npz')['data']

n_fames = 30000
frames = np.arange(0, n_fames, 1)
# cutoff distance is 2 times the mean radius the droplets have at that frame
mean_d = 2*rawTrajs.groupby("frame").mean().r.values

n = 50
random.seed(5)
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]

centrality = np.zeros((n_fames, 50))
betweenness = np.zeros((n_fames, 50))
clustering = np.zeros((n_fames, 50))
random_geometric_graphs = []
node_colors = []
for frame in tqdm(frames):
    X = np.array(rawTrajs.loc[rawTrajs.frame == frame, ['x', 'y']])
    # create dictionary with positions of the agents
    dicts = {}
    for i in range(len(X)):
        dicts[i] = (X[i, 0], X[i, 1])
    # generate random geometric graph with cutoff distance 2.2 times the mean diameter the droplets have at that frame
    temp = nx.random_geometric_graph(len(dicts), mean_d[frame]*2.2, pos=dicts, dim=2)
    random_geometric_graphs.append(temp)    
    """
    centrality[frame] = list(nx.degree_centrality(temp).values())
    betweenness[frame] = list(nx.betweenness_centrality(temp).values())
    clustering[frame] = list(nx.clustering(temp).values()) 
    

    communities = greedy_modularity_communities(temp, resolution = 0.1)
    community_colors = {}
    for i, community_set in enumerate(communities):
        for node in community_set:
            community_colors[node] = i
    node_colors.append([community_colors[node] for node in temp.nodes])
    """ 

if 0:
    initFrame = 0
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
        ax.clear()
        nx.draw_networkx(random_geometric_graphs[frame], pos=nx.get_node_attributes(random_geometric_graphs[frame], 'pos'),\
                             node_size = 50, node_color = node_colors[frame], ax = ax, with_labels=False)
        ax.set(xlim=(0, 900), ylim=(900, 0), title=f"frame {frame}, communities", xlabel="x [px]", ylabel="y [px]")
        return ax

    ax = fig.add_subplot(111)
    title = ax.set_title('DBSCAN - frame = 0')
    nx.draw_networkx(random_geometric_graphs[initFrame], pos=nx.get_node_attributes(random_geometric_graphs[initFrame], 'pos'),\
                             node_size = 50, node_color = node_colors[initFrame], ax = ax, with_labels=False)
    ax.set(xlim=(0, 900), ylim=(900, 0), title=f"frame {initFrame}, communities", xlabel="x [px]", ylabel="y [px]")

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(initFrame, rawTrajs.frame.max(), 2), interval = 5, blit=False)
    if 1:
        writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
        ani.save(f'{res_path}/clustering/clustering_greedy2.mp4', writer=writer, dpi = 300)
    plt.close()

if 1:
    cluster_centroids = {}
    labels = np.ones((n_fames, 50), dtype=int)*-1
    for frame in tqdm(rawTrajs.frame.unique()):
        positions = rawTrajs.loc[rawTrajs.frame == frame, ['x', 'y']].values
        labels[frame] = DBSCAN(eps=mean_d[frame]*2.2, min_samples=2).fit_predict(positions)
clustered_df = rawTrajs.copy()
clustered_df['cluster_id'] = labels.flatten()
clustered_df['cluster_color'] = [colors[i] for i in labels.flatten()]
#clustered_df.to_parquet(f"{res_path}/clustering/df_clustered.parquet")


node_colors = clustered_df.cluster_color.values.reshape((n_fames, 50))

initFrame = 0

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
    ax.clear()
    nx.draw_networkx(random_geometric_graphs[frame], pos=nx.get_node_attributes(random_geometric_graphs[frame], 'pos'),\
                     node_size = 50, node_color = node_colors[frame], ax = ax, with_labels=False)
    ax.set_title(f'DBSCAN + random geometric graph - frame = {frame}')
    ax.set(xlabel = 'X [px]', ylabel = 'Y [px]', xlim=(0, 900), ylim=(900, 0))
    return ax

ax = fig.add_subplot(111)
title = ax.set_title('DBSCAN + random geometric graph - frame = 0')
ax.set(xlabel = 'X [px]', ylabel = 'Y [px]', xlim=(0, 900), ylim=(900, 0))
nx.draw_networkx(random_geometric_graphs[initFrame], pos=nx.get_node_attributes(random_geometric_graphs[initFrame], 'pos'),\
                     node_size = 50, node_color = node_colors[initFrame], ax = ax, with_labels=False)
fig.canvas.mpl_connect('button_press_event', onClick)
ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(initFrame, clustered_df.frame.max(), 2), interval = 5, blit=False)
if 1: 
    writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
    ani.save(f'{res_path}/clustering/clustering_dbscan2.mp4', writer=writer, dpi = 300)
plt.close()







