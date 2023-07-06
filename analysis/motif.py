
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib.animation
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 8})

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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def compute_eccentricity(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    eccentricity = np.std(distances) / np.mean(distances)
    return eccentricity

def motif_search(mofitList, sizeList, g):
    counts = np.zeros(len(mofitList), dtype=int)
    for i, mofit in enumerate(mofitList):
        _, temp = gt.motifs(g, sizeList[i], motif_list=[mofit])
        counts[i] = temp[0]
    return counts

print("Import data...")
if 1:
    data_path = "../tracking/49b_1r/49b_1r_pre_merge/df_linked.parquet"
    res_path = "./49b_1r/results" 
    analysis_data_path = "./49b_1r/analysis_data"
    red_particle_idx = 8
    red_mask = np.zeros(50, dtype=bool)
    red_mask[red_particle_idx] = True
    colors = np.array(['b' for i in range(50)])
    colors[red_particle_idx] = 'r'
    fps = 10
else:
    data_path = "../tracking/25b_25r/df_linked.parquet"
    res_path = "./25b_25r/results" 
    analysis_data_path = "./25b_25r/analysis_data"
    red_particle_idx = np.sort(np.array([27, 24, 8, 16, 21, 10, 49, 14, 12, 9, 7, 37, 36, 40, 45, 42, 13, 20, 26, 2, 39, 5, 11, 22, 44])).astype(int)
    red_mask = np.zeros(50, dtype=bool)
    red_mask[red_particle_idx] = True
    colors = np.array(['b' for i in range(50)])
    colors[red_particle_idx] = 'r'
    fps = 30

rawTrajs = pd.read_parquet(data_path)
nDrops = int(len(rawTrajs.loc[rawTrajs.frame==0]))
frames = rawTrajs.frame.unique().astype(int)
nFrames = len(frames)
print(f"Number of Droplets: {nDrops}")
print(f"Number of Frames: {nFrames} at {fps} fps --> {nFrames/fps:.2f} s")


sizeList = [3, 4, 5, 5]
motifList = []
for i in range(len(sizeList)):
    motifList.append(gt.load_graph(f'{res_path}/graph/motif/selected/motif_{i}.graphml'))

factorList = np.round(np.arange(1.4, 3, 0.1),1)
mean_d = 2*rawTrajs.groupby("frame").mean().r.values
if 0:   
    for factor in factorList:
        motif_results = np.zeros((nFrames, len(motifList)), dtype=int)
        for frame in tqdm(frames):
            X = np.array(rawTrajs.loc[rawTrajs.frame == frame, ['x', 'y']])
            g, pos = gt.geometric_graph(X, mean_d[frame]*factor)
            motif_results[frame] = motif_search(motifList, sizeList, g)
        df_motif = pd.DataFrame(motif_results, columns=[f'motif_{i}' for i in range(len(motifList))])
        df_motif.to_parquet(f"{analysis_data_path}/graph/motif/motif_analysis_factor{factor}.parquet")

if 0:
    for factor in factorList:
        clust_id_list = []
        for frame in tqdm(frames):
            X = np.array(rawTrajs.loc[rawTrajs.frame == frame, ['x', 'y']])
            dicts = {}
            for j in range(len(X)):
                dicts[j] = (X[j, 0], X[j, 1])
            temp = nx.random_geometric_graph(len(dicts), mean_d[frame]*factor, pos=dicts, dim=2)
            clust_id = np.ones(len(X), dtype=int)*-1
            for id, c in enumerate(list(nx.connected_components(temp))):
                clust_id[list(c)] = id
            clust_id_list += list(clust_id)
        clustered_trajs = rawTrajs.copy()
        clustered_trajs["cluster_id"] = clust_id_list
        clustered_trajs["cluster_color"] = [colors[i] for i in clust_id_list]
        clustered_trajs.to_parquet(f"{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet")

verb_mean = False       
if 1:
    for factor in factorList:
        clustered_trajs = pd.read_parquet(f"{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet")
        if verb_mean:
            if 1:
                res = np.zeros((len(clustered_trajs.frame.unique()), 12))
                for frame in tqdm(clustered_trajs.frame.unique()):
                    df = clustered_trajs.loc[clustered_trajs.frame == frame]
                    labels = df.cluster_id.values
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    degree, degree_centrality, betweenness_centrality, clustering, dim_cycles, eccentricities, area = [], [], [], [], [], [], []
                    first_eigenvalue, second_eigenvalue = [], []
                    tot_n_cycles = 0

                    for j, cluster_id in enumerate(unique_labels[counts>2]):
                        # create subgrah with positions of the agents in the cluster
                        df_cluster = df.loc[df.cluster_id == cluster_id]
                        X = np.array(df_cluster[['x', 'y']])
                        dicts = {}
                        for i in range(len(X)):
                            dicts[i] = (X[i, 0], X[i, 1])
                        temp = nx.random_geometric_graph(len(dicts), mean_d[frame]*factor, pos=dicts, dim=2)

                        # compute area and eccentricity of the subgraph
                        hull = ConvexHull(X)
                        area.append(hull.area)
                        eccentricities.append(compute_eccentricity(X))

                        # compute degree, degree centrality, betweenness centrality, clustering coefficient, number of cycles,
                        # dimension of cycles and first and eigenvalue
                        degree += [val for (_, val) in temp.degree()]
                        degree_centrality += list(nx.degree_centrality(temp).values())
                        betweenness_centrality += list(nx.betweenness_centrality(temp).values())
                        clustering += list(nx.clustering(temp).values())
                        cycles = nx.cycle_basis(temp)
                        tot_n_cycles += int(len(cycles))
                        dim_cycles += [len(cycles[i]) for i in range(int(len(cycles)))]
                        lap_eigenvalues = nx.laplacian_spectrum(temp)
                        first_eigenvalue.append(lap_eigenvalues[1])
                        second_eigenvalue.append(lap_eigenvalues[2])

                    mean_degree = np.mean(degree)
                    mean_degree_centrality = np.mean(degree_centrality)
                    mean_betweenness_centrality = np.mean(betweenness_centrality)
                    mean_clustering = np.mean(clustering)
                    if tot_n_cycles == 0:
                        mean_dim_cycles = 0
                    else:
                        mean_dim_cycles = np.mean(dim_cycles)
                    mean_area = np.mean(area)
                    mean_eccentricity = np.mean(eccentricities)
                    mean_first_eigenvalue = np.mean(first_eigenvalue)
                    mean_second_eigenvalue = np.mean(second_eigenvalue)

                    res[frame] = np.concatenate(([frame], [len(unique_labels[counts>2])], [tot_n_cycles], [mean_degree], [mean_degree_centrality],\
                                                [mean_betweenness_centrality], [mean_clustering], [mean_dim_cycles], [mean_area], [mean_eccentricity],\
                                                [mean_first_eigenvalue], [mean_second_eigenvalue]))

                label = np.array(['frame', 'n_clusters', 'n_cycles', 'degree', 'degree_centrality', 'betweenness', 'clustering',\
                        'd_cycles', 'area', 'eccentricity', 'first_eigv', 'second_eigv'])
                df_graph = pd.DataFrame(res, columns=label)
                df_graph.to_parquet(f"{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet")
            else:
                df_graph = pd.read_parquet(f"{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet")
                label = np.array(df_graph.columns.values, dtype=str)
        else:
            hist_bins_degree = np.arange(0, 30, 1)
            hist_bins_01 = np.arange(0, 1.01, 0.05)
            hist_bins_dim_cycles = np.arange(0, 51, 1)
            area_bins = np.arange(100, 2100, 100)

            if 1:
                res = np.zeros((len(clustered_trajs.frame.unique()), 201))
                for frame in tqdm(clustered_trajs.frame.unique()):
                    df = clustered_trajs.loc[clustered_trajs.frame == frame]
                    labels = df.cluster_id.values
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    degree, degree_centrality, betweenness_centrality, clustering, area, eccentricities, dim_cycles = [], [], [], [], [], [], []
                    lap_eigenvalues = []
                    tot_n_cycles = 0

                    for j, cluster_id in enumerate(unique_labels[counts>2]):
                        df_cluster = df.loc[df.cluster_id == cluster_id]
                        X = np.array(df_cluster[['x', 'y']])
                        # create dictionary with positions of the agents
                        dicts = {}
                        for i in range(len(X)):
                            dicts[i] = (X[i, 0], X[i, 1])
                        temp = nx.random_geometric_graph(len(dicts), mean_d[frame]*factor, pos=dicts, dim=2)

                        # compute area and eccentricity of the subgraph
                        hull = ConvexHull(X)
                        area.append(hull.area)
                        eccentricities.append(compute_eccentricity(X))
                        
                        degree += [val for (_, val) in temp.degree()]
                        degree_centrality += list(nx.degree_centrality(temp).values())
                        betweenness_centrality += list(nx.betweenness_centrality(temp).values())
                        clustering += list(nx.clustering(temp).values())

                        cycles = nx.cycle_basis(temp)
                        tot_n_cycles += int(len(cycles))
                        dim_cycles += [len(cycles[i]) for i in range(int(len(cycles)))]
                        lap_eigenvalues += list(nx.laplacian_spectrum(temp))
                
                    degree_count,             _ = np.histogram(degree, bins=hist_bins_degree)
                    centrality_count,         _ = np.histogram(degree_centrality, bins=hist_bins_01) 
                    betweenness_count,        _ = np.histogram(betweenness_centrality, bins=hist_bins_01)
                    clustering_count,         _ = np.histogram(clustering, bins=hist_bins_01)
                    dim_cycles_count,         _ = np.histogram(dim_cycles, bins=hist_bins_dim_cycles)
                    area,                     _ = np.histogram(area, bins=area_bins)
                    eccentricity,             _ = np.histogram(eccentricities, bins=hist_bins_01)
                    laplacian_spectrum_count, _ = np.histogram(lap_eigenvalues, bins=hist_bins_01)
                    
                    res[frame] =  np.concatenate(([frame], [len(unique_labels[counts>2])], [tot_n_cycles], degree_count, centrality_count,\
                                                betweenness_count, clustering_count, dim_cycles_count, area, eccentricity,\
                                                laplacian_spectrum_count))
                label = np.array(['frame'] + ['nClusters'] + ['nCycles'] + [f'degree_{i}' for i in range(len(hist_bins_degree[1:]))] + \
                        [f'centrality_{i}' for i in range(len(hist_bins_01[1:]))] + [f'betweenness_{i}' for i in range(len(hist_bins_01[1:]))] + \
                        [f'clustering_{i}' for i in range(len(hist_bins_01[1:]))] + [f'dim_cycles_{i}' for i in range(len(hist_bins_dim_cycles[1:]))] +\
                        [f'area_{i}' for i in range(len(area_bins[1:]))] + [f'eccentricity_{i}' for i in range(len(hist_bins_01[1:]))] +\
                        [f'laplacian_{i}'for i in range(len(hist_bins_01[1:]))])
                df_graph = pd.DataFrame(res, columns=label)
                df_graph.to_parquet(f'{analysis_data_path}/graph/graph_analysis_factor{factor}.parquet')
            else:
                df_graph = pd.read_parquet(f'{analysis_data_path}/graph/graph_analysis_factor{factor}.parquet')
                label = np.array(df_graph.columns.values, dtype=str)
                display(df_graph)

            # how to index the array
            id_nClusters = np.argwhere(np.char.startswith(label, 'nClusters')==True)[0][0]
            id_nCycles = np.argwhere(np.char.startswith(label, 'nCycles')==True)[0][0]
            id_degree = np.argwhere(np.char.startswith(label, 'degree')==True)[0][0], np.argwhere(np.char.startswith(label, 'degree')==True)[-1][0]
            id_centrality = np.argwhere(np.char.startswith(label, 'centrality')==True)[0][0], np.argwhere(np.char.startswith(label, 'centrality')==True)[-1][0]
            id_betweenness = np.argwhere(np.char.startswith(label, 'betweenness')==True)[0][0], np.argwhere(np.char.startswith(label, 'betweenness')==True)[-1][0]
            id_clustering = np.argwhere(np.char.startswith(label, 'clustering')==True)[0][0], np.argwhere(np.char.startswith(label, 'clustering')==True)[-1][0]
            id_dim_cycles = np.argwhere(np.char.startswith(label, 'dim_cycles')==True)[0][0], np.argwhere(np.char.startswith(label, 'dim_cycles')==True)[-1][0]
            id_area = np.argwhere(np.char.startswith(label, 'area')==True)[0][0], np.argwhere(np.char.startswith(label, 'area')==True)[-1][0]
            id_eccentricity = np.argwhere(np.char.startswith(label, 'eccentricity')==True)[0][0], np.argwhere(np.char.startswith(label, 'eccentricity')==True)[-1][0]
            id_laplacian = np.argwhere(np.char.startswith(label, 'laplacian')==True)[0][0], np.argwhere(np.char.startswith(label, 'laplacian')==True)[-1][0]


