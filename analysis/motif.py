import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib.animation
import matplotlib.gridspec as gridspec

import pims

import numpy as np
import pandas as pd
import random

from scipy.spatial import KDTree, cKDTree, Voronoi, voronoi_plot_2d, ConvexHull
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import graph_tool.all as gt

show_verb = False
run_windowed_analysis = True
plot_verb = True
animated_plot_verb = True
save_verb = True
run_analysis_verb = True

msd_run = False
speed_run = False
turn_run = False
autocorr_run = True
rdf_run = False


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def compute_eccentricity(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    eccentricity = np.std(distances) / np.mean(distances)
    return eccentricity

def motif_search(mofitList, sizeList, g):
    counts = np.zeros(len(mofitList), dtype=int)
    for i, motif in enumerate(mofitList):
        _, temp = gt.motifs(g, sizeList[i], motif_list = [motif])
        counts[i] = temp[0]
    return counts


video_selection = "25b25r"
#video_selection = "49b1r"

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
    data_path = "../tracking/49b_1r/49b_1r_pre_merge/df_linked.parquet"
    res_path = "./49b_1r/results"
    pdf_res_path = "../../thesis_project/images/49b_1r"
    analysis_data_path = "./49b_1r/analysis_data"
    
    red_particle_idx = np.array([8]).astype(int)
    fps = 10
    maxLagtime = 100*fps # maximum lagtime to be considered in the analysis, 100 seconds
    v_step = fps

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
    data_path = "../tracking/25b_25r/part1/df_linked.parquet"
    pdf_res_path = "../../thesis_project/images/25b_25r"
    res_path = "./25b_25r/results"
    analysis_data_path = "./25b_25r/analysis_data"
    red_particle_idx = np.sort(np.array([27, 24, 8, 16, 21, 10, 49, 14, 12, 9, 7, 37, 36, 40, 45, 42, 13, 20, 26, 2, 39, 5, 11, 22, 44])).astype(int)
    fps = 30
    maxLagtime = 100*fps # maximum lagtime to be considered in the analysis, 100 seconds = 100 * fps
    v_step = fps
else:
    raise ValueError("No valid video selection")
    
original_trajectories = pd.read_parquet(data_path)
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
pxDimension = 1 # has to be defined 
x = np.arange(1, maxLagtime/fps + 1/fps, 1/fps) 

# WINDOWED ANALYSIS PARAMETERS
window = 300*fps # 320 s
stride = 10*fps # 10 s
print("Windowed analysis args:")
startFrames = np.arange(0, nFrames-window, stride, dtype=int)
endFrames = startFrames + window
nSteps = len(startFrames)
print(f"window of {window/fps} s, stride of {stride/fps} s --> {nSteps} steps")
units = "px/s"
default_kwargs_blue = {"color": "#00FFFF", "ec": (0, 0, 0, 0.6), "density": True}
default_kwargs_red = {"color": "#EE4B2B", "ec": (0, 0, 0, 0.6), "density": True}

if 1:
    trajectories = get_smooth_trajs(original_trajectories, nDrops, int(fps/2), 4)
else:
    trajectories = original_trajectories


factor = 2.2
droplets_diameter = 2*trajectories.groupby("frame").r.mean()
cutoff_distance = factor*droplets_diameter

if 0:
    clust_id_list = []
    for frame in tqdm(frames):
        X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
        dicts = {}
        for j in range(len(X)):
            dicts[j] = (X[j, 0], X[j, 1])
        temp = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos=dicts, dim=2)
        clust_id = np.ones(len(X), dtype=int)*-1
        for id, c in enumerate(list(nx.connected_components(temp))):
            clust_id[list(c)] = id
        clust_id_list += list(clust_id)
    clustered_trajs = trajectories.copy()
    clustered_trajs["cluster_id"] = clust_id_list
    clustered_trajs["cluster_color"] = [colors[i] for i in clust_id_list]
    clustered_trajs.to_parquet(f"{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet")
else:
    clustered_trajs = pd.read_parquet(f"{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet")

if 0:
    # CHOOSE MOTIFS TO SEARCH FOR BY HAND
    for frame in np.random.choice(frames, 10):
        X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
        g, pos = gt.geometric_graph(X, cutoff_distance[frame])
        gt.graph_draw(g, pos=pos, output=f'{res_path}/graph/motif/search/graph_{frame}.png', vertex_size=10, vertex_pen_width=0.5, edge_pen_width=0.5, )

        for n_vertices in range(3, 10):
            motifs, counts = gt.motifs(g, n_vertices)
            # print(motifs)
            for i in range(len(motifs)):
                gt.graph_draw(motifs[i], output=f'{res_path}/graph/motif/search/motif_{frame}_{n_vertices}_{i}.png', vertex_size=10, vertex_pen_width=0.5, edge_pen_width=0.5)
            #print(counts)
    # build by hand the motif list array:
    motifList = []
    # example
    motifList.append(motif[1])
    # write the sizes of the motifs in the sizeList array
    sizeList = []
    # save the motifList 
    for i, motif in enumerate(motifList):
        gt.graph_draw(motif, output=f'{res_path}/graph/motif_{i}.pdf', vertex_size=10)
        motif.save(f'{res_path}/graph/motif_{i}.graphml')
else:
    sizeList = [3, 4, 5, 5]
    motifList = []
    for i in range(len(sizeList)):
        motifList.append(gt.load_graph(f'{res_path}/graph/motif/selected/motif_{i}.graphml'))
if 0:
    motif_array = np.zeros((len(frames), len(motifList)+1), dtype=int)
    for frame in tqdm(frames):
        X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
        g, pos = gt.geometric_graph(X, cutoff_distance[frame])
        motif_array[frame] = np.concatenate(([frame], list(motif_search(motifList, sizeList, g))))
    motif_df = pd.DataFrame(motif_array, columns=["frame"] + [f"motif_{i}" for i in range(len(motifList))])
    motif_df.to_parquet(f"{analysis_data_path}/graph/motifs_factor{factor}.parquet")

verb_mean = True
if 1:
    result = []
    for frame in tqdm(clustered_trajs.frame.unique()):
        df = clustered_trajs.loc[clustered_trajs.frame == frame]
        labels = df.cluster_id.values
        unique_labels, counts = np.unique(labels, return_counts=True)

        for j, cluster_id in enumerate(unique_labels[counts>2]):
            # create subgrah with positions of the agents in the cluster
            df_cluster = df.loc[df.cluster_id == cluster_id]
            X = np.array(df_cluster[['x', 'y']])
            dicts = {}
            for i in range(len(X)):
                dicts[i] = (X[i, 0], X[i, 1])
            temp = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos=dicts, dim=2)

            # compute area and eccentricity of the subgraph
            hull = ConvexHull(X)
            area = hull.area
            eccentricity = compute_eccentricity(X)

            # compute degree, degree centrality, betweenness centrality, clustering coefficient, number of cycles,
            # dimension of cycles and first and eigenvalue
            degree = np.mean([val for (_, val) in temp.degree()])
            degree_centrality = np.mean(list(nx.degree_centrality(temp).values()))
            betweenness_centrality = np.mean(list(nx.betweenness_centrality(temp).values()))
            clustering = np.mean(list(nx.clustering(temp).values()))
            cycles = nx.cycle_basis(temp)
            n_cycles = int(len(cycles))
            dim_cycles = [len(cycles[i]) for i in range(int(len(cycles)))]
            if n_cycles == 0:
                mean_dim_cycles = 0
            else:
                mean_dim_cycles = np.mean(dim_cycles)

            # laplacian eigenvalues
            lap_eigenvalues = nx.laplacian_spectrum(temp)

            # append to list
            result.append(np.concatenate(([frame], [len(unique_labels[counts>2])], [n_cycles], [degree], [degree_centrality],\
                                    [betweenness_centrality], [clustering], [mean_dim_cycles], [area], [eccentricity],\
                                    [lap_eigenvalues[1]], [lap_eigenvalues[2]])))

    label = np.array(['frame', 'n_clusters', 'n_cycles', 'degree', 'degree_centrality', 'betweenness', 'clustering',\
            'd_cycles', 'area', 'eccentricity', 'first_lapl_eigv', 'second_lapl_eigv'])
    df_graph = pd.DataFrame(result, columns=label)
    display(df_graph)
    df_graph.to_parquet(f"{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet")
else:
    print(f"Import data with factor {factor}")
    df_graph = pd.read_parquet(f"{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet")
    label = np.array(df_graph.columns)