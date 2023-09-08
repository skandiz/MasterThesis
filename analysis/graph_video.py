import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py 
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from tqdm import tqdm
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx 
import random

def get_frame(frame):
    with h5py.File(data_preload_path, 'r') as f:
        # Access the dataset within the HDF5 file
        dataset = f['dataset_name']
        frameImg = dataset[frame]
    return frameImg

print("Import data...")
if 0:
    data_path = "../tracking/49b_1r/49b_1r_pre_merge/df_linked.parquet"
    tracking_path = "../tracking/49b_1r/49b_1r_pre_merge"
    res_path = "./49b_1r/results" 
    analysis_data_path = "./49b_1r/analysis_data"
    red_particle_idx = 8
    fps = 10
    v_step = 10
    xmin = 55
    xmax = 880
    ymin = 55
    ymax = 880
else:
    part = 1
    data_path = f"../tracking/25b_25r/part{part}/df_linked.parquet"
    tracking_path = f"../tracking/25b_25r/part{part}/"
    data_preload_path = '/Volumes/ExtremeSSD/UNI/h5_data_thesis/data.h5'
    res_path = "./25b_25r/results" 
    analysis_data_path = "./25b_25r/analysis_data"
    red_particle_idx = np.sort(np.array([27, 24, 8, 16, 21, 10, 49, 14, 12, 9, 7, 37, 36, 40, 45, 42, 13, 20, 26, 2, 39, 5, 11, 22, 44])).astype(int)
    fps = 30
    v_step = 30
    xmin = 100
    xmax = 530
    ymin = 35
    ymax = 465

trajectories = pd.read_parquet(data_path)
nDrops = int(len(trajectories.loc[trajectories.frame==0]))
chars = '0123456789ABCDEF'
colors = ['#'+''.join(random.sample(chars,6)) for i in range(100)]
frames = trajectories.frame.unique().astype(int)
nFrames = len(frames)
print(f"Number of Droplets: {nDrops}")
print(f"Number of Frames: {nFrames} at {fps} fps --> {nFrames/fps:.2f} s")

if 0:
    fig = plt.figure(figsize = (5, 5))
    def update_graph(frame):
        df = trajectories.loc[(trajectories.frame == frame) , ["x", "y", "color", "r"]]
        for i in range(50):
            graph[i].center = (df.x.values[i], df.y.values[i])
            graph[i].radius = df.r.values[i]
        graph2.set_data(get_frame(frame))
        title.set_text('Tracking raw - frame = {}'.format(frame))
        return graph

    ax = fig.add_subplot(111)
    title = ax.set_title('Tracking stardist + trackpy - frame = 0')
    ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
    df = trajectories.loc[(trajectories.frame == 0), ["x", "y", "color", "r"]]
    graph = []
    for i in range(50):
        graph.append(ax.add_artist(plt.Circle((df.x.values[i], df.y.values[i]), df.r.values[i], color = df.color.values[i],\
                                               fill = False, linewidth=1)))
    graph2 = ax.imshow(get_frame(0))
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames, interval = 5, blit=False)
    writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
    ani.save(tracking_path + 'tracking_stardist.mp4', writer=writer, dpi = 300)
    plt.close()

if 1:
    factor = 2.2
    mean_d = 2*trajectories.groupby("frame").mean().r.values

    

    if 1:
        clust_id_list = []
        random_geometric_graphs = []
        for frame in tqdm(frames):
            X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
            dicts = {}
            for j in range(len(X)):
                dicts[j] = (X[j, 0], X[j, 1])
            temp = nx.random_geometric_graph(len(dicts), mean_d[frame]*factor, pos=dicts, dim=2)
            random_geometric_graphs.append(temp)
            clust_id = np.ones(len(X), dtype=int)*-1
            for id, c in enumerate(list(nx.connected_components(temp))):
                clust_id[list(c)] = id
            clust_id_list += list(clust_id)
        clustered_trajs = trajectories.copy()
        clustered_trajs = clustered_trajs.loc[clustered_trajs.frame.between(frames[0], frames[-1])]
        clustered_trajs["cluster_id"] = clust_id_list
        clustered_trajs["cluster_color"] = [colors[i] for i in clust_id_list]
        clustered_trajs.to_parquet(f"{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet")
    else:
        clustered_trajs = pd.read_parquet(f"{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet")
    
    node_colors = clustered_trajs.cluster_color.values.reshape(len(frames), nDrops)

    fig = plt.figure(figsize = (10, 10))
    def update_graph(frame):
        ax.clear()
        nx.draw_networkx(random_geometric_graphs[frame], pos=nx.get_node_attributes(random_geometric_graphs[frame], 'pos'),\
                             node_size = 50, node_color = node_colors[frame], ax = ax, with_labels=False)
        ax.set(xlim=(xmin, xmax), ylim=(ymax, ymin), title=f"frame {frame}", xlabel="x [px]", ylabel="y [px]")
        return ax

    ax = fig.add_subplot(111)
    title = ax.set_title('frame = 0')
    nx.draw_networkx(random_geometric_graphs[0], pos=nx.get_node_attributes(random_geometric_graphs[0], 'pos'),\
                             node_size = 50, node_color = node_colors[0], ax = ax, with_labels=False)
    ax.set(xlim=(xmin, xmax), ylim=(ymax, ymin), title=f"frame {0}", xlabel="x [px]", ylabel="y [px]")
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames[::5], interval = 5, blit=False)
    writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args=['-vcodec', 'libx264'])
    ani.save(f'{analysis_data_path}/clustering/graph_video_{factor}.mp4', writer=writer, dpi = 300)
    plt.close()
