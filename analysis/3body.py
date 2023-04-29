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

show_verb = False
save_verb = True
anim_show_verb = False

traj_verb = "hough"

if traj_verb == "trackpy": 
    rawTrajs = pd.read_parquet("../tracking/results/tracking_data/trackpy_pre_merge.parquet")
    res_path = "results"
    analysis_data_path = "analysis_data"

elif traj_verb == "hough":
    rawTrajs = pd.read_parquet("../tracking/results/tracking_data/tracking_hough_trackpy_linking.parquet")
    res_path = "hough_results"
    analysis_data_path = "hough_analysis_data"
else:
    raise ValueError("traj_verb must be either 'trackpy' or 'hough'")

red_particle_idx = 17
colors = rawTrajs.loc[rawTrajs.frame == 0, 'color'].values
nDrops = int(len(rawTrajs.loc[rawTrajs.frame==0]))
nFrames = int(max(rawTrajs.frame) + 1)
print(f"nDrops:{nDrops}")
print(f"nFrames:{nFrames} --> {nFrames/10:.2f} s")

# Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
smoothTrajs = get_smooth_trajs(rawTrajs, nDrops, 30, 2)

# WINDOWED ANALYSIS PARAMETERS
window = 3200 # 320 s
stride = 100 # 10 s
print(f"window of {window/10} s, stride of {stride/10} s")
startFrames = np.arange(0, nFrames-window, stride, dtype=int)
endFrames = startFrames + window
nSteps = len(startFrames)
print(f"number of steps: {nSteps}")

# step 10 with a 10 fps video --> 1 s
units = "px/s"
default_kwargs_blue = {"color": "#00FFFF", "ec": (0, 0, 0, 0.6), "density": True}
default_kwargs_red = {"color": "#EE4B2B", "ec": (0, 0, 0, 0.6), "density": True}

@njit
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin
    bin = int(n * (x - a_min) / (a_max - a_min))
    if bin < 0 or bin >= n:
        return None
    else:
        return bin
@njit
def numba_histogram(a, bin_edges, weights):
    hist = np.zeros((len(bin_edges)-1,), dtype=np.intp)
    for ind, x in enumerate(a):
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += weights[ind]
    return hist

@njit
def numba_mean_ax0(a):
    res = []
    for i in prange(a.shape[1]):
        res.append(a[:, i].mean())
    return np.array(res)


# this doesn't work
@njit(parallel = True)
def three_body_frame(coords, bList, sigma, hist_bins):
    """
    Computes the three body distribution for a single frame
    coords : array of shape (nDrops, 3)
        The coordinates of the drops
    bList : array of shape (nB)
        The b values to compute the three body distribution for
    sigma : float   
        The standard deviation of the gaussian used to weight the three body distribution
    hist_bins : array of shape (nBins)
        The bins to compute the three body distribution for
    Returns 
    ------- 
    mean_3_body : array of shape (nB, nBins)
        The mean three body distribution for each b value
    """
    res = np.ones((len(bList), len(hist_bins)-1))
    three_body = np.ones((nDrops, len(hist_bins)-1))
    angles = np.ones(int((nDrops-1)*(nDrops-2)/2))
    gauss_weights = np.zeros(int((nDrops-1)*(nDrops-2)/2))
    for b_ind in prange(len(bList)):
        b = bList[b_ind]
        for i in prange(nDrops):
            count = 0
            r_i = coords[i]
            for j in range(nDrops):
                if i == j:
                    continue
                r_ij = coords[j] - r_i
                for k in range(j+1, nDrops):
                    if k == i:
                        continue
                    r_ik = coords[k] - r_i
                    angles[count] = np.arccos(np.dot(r_ij, r_ik) / (np.linalg.norm(r_ij) * np.linalg.norm(r_ik)))
                    gauss_weights[count] = np.exp(-0.5*((np.linalg.norm(r_ij)-b)/sigma)**2) * np.exp(-0.5*((np.linalg.norm(r_ik)-b)/sigma)**2)
                    count += 1
            three_body[i] = numba_histogram(angles, hist_bins, gauss_weights)
        res[b_ind] = numba_mean_ax0(three_body)
    return res

@njit(parallel = True, fastmath = True)
def three_body_frame_modified(coords, bList, sigma, hist_bins):
    angles = np.ones((len(bList), nDrops, int((nDrops-1)*(nDrops-2)/2)))
    gauss_weights = np.ones((len(bList), nDrops, int((nDrops-1)*(nDrops-2)/2)))
    for b_ind in prange(len(bList)):
        b = bList[b_ind]
        for i in range(nDrops):
            count = 0
            r_i = coords[i]
            for j in range(nDrops):
                if i == j:
                    continue
                r_ij = coords[j] - r_i
                for k in range(j+1, nDrops):
                    if k == i:
                        continue
                    r_ik = coords[k] - r_i
                    angles[b_ind, i, count] = np.arccos(np.dot(r_ij, r_ik) / (np.linalg.norm(r_ij) * np.linalg.norm(r_ik)))
                    gauss_weights[b_ind, i, count] = np.exp(-0.5*((np.linalg.norm(r_ij)-b)/sigma)**2) * np.exp(-0.5*((np.linalg.norm(r_ik)-b)/sigma)**2)
                    count += 1
    return angles, gauss_weights


# test 2
COORDS = np.array(rawTrajs.loc[:,["x","y"]])
sigma = 38
bList = np.arange(0, 300, 300/50) # step 0.2 too little --> 448 hours to complete 30k frames
print("bList length: ", bList.shape[0])
hist_bins = np.arange(0, np.pi, np.pi/100)
hist_centers = (hist_bins[:-1] + hist_bins[1:])/2
frames = np.arange(0, 30000, 100)

counts = np.zeros((len(frames), len(bList), nDrops, len(hist_bins)-1))
for frame_ind in tqdm(range(len(frames))):
    test_ang, test_weights = three_body_frame_modified(COORDS[frames[frame_ind]:frames[frame_ind]+50], bList, sigma, hist_bins)
    for b_ind in range(len(bList)):
        for i in range(nDrops):
            counts[frame_ind, b_ind, i] = np.histogram(test_ang[b_ind, i], bins = hist_bins, weights = test_weights[b_ind, i])[0]
test_reshape = counts.reshape((len(bList)*nDrops*(len(hist_bins)-1), len(frames)))
test_df = pd.DataFrame(test_reshape)
test_df.columns = frames.astype(str)
test_df.to_parquet(f"./{analysis_data_path}/3_body/test_3body2.parquet")