import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation
import matplotlib as mpl
font = {'size'   : 12}
mpl.rc('font', **font)

import numpy as np
import pandas as pd
import pims
import networkx as nx

from scipy.spatial import KDTree, cKDTree, Voronoi, voronoi_plot_2d, ConvexHull
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import time
import joblib
from tqdm import tqdm
import trackpy as tp
from numba import njit, prange

from yupi import Trajectory, WindowType, DiffMethod
import yupi.graphics as yg
import yupi.stats as ys
from yupi.transformations import subsample

from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx 
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import graph_tool.all as gt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

show_verb = False
run_windowed_analysis = True
plot_verb = True
animated_plot_verb = False
save_verb = True
run_analysis_verb = True

# IMPORT FUNCTIONS
def get_smooth_trajs(trajs, nDrops, windLen, orderofPoly):
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    ret = trajs.copy()
    for i in range(nDrops):
        ret.loc[ret.particle == i, "x"] = savgol_filter(trajs.loc[trajs.particle == i].x.values, windLen, orderofPoly)
        ret.loc[ret.particle == i, "y"] = savgol_filter(trajs.loc[trajs.particle == i].y.values, windLen, orderofPoly)    
    return ret

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

def powerLawFit(f, x, nDrops, yerr):
    if nDrops == 1:
        ret = np.zeros((2, 2))
        ret[0], pcov = curve_fit(powerLaw, x, f, p0 = [1., 1.])
        ret[1] = np.sqrt(np.diag(pcov))
        fit = ret[0, 0] * x**ret[0, 1]
    else:
        fit = np.zeros((nDrops, f.shape[0])) 
        ret = np.zeros((nDrops, 2, 2))
        for i in range(nDrops):
            if yerr is None:
                ret[i, 0], pcov = curve_fit(powerLaw, x, f[i], p0 = [1., 1.])
            else:
                ret[i, 0], pcov = curve_fit(powerLaw, x, f[i], p0 = [1., 1.], sigma = yerr)
            ret[i, 1] = np.sqrt(np.diag(pcov))
            fit[i] = ret[i, 0, 0] * x**ret[i, 0, 1]
    return fit, ret 

def get_imsd(trajs, pxDimension, fps, maxLagtime, fit_range):
    imsd = tp.imsd(trajs, mpp = pxDimension, fps = fps, max_lagtime = maxLagtime)
    # fit the IMSD in the fit_range
    id_start = np.where(imsd.index == fit_range[0])[0][0]
    id_end = np.where(imsd.index == fit_range[-1])[0][0] + 1
    imsd_to_fit = imsd.iloc[id_start:id_end]
    fit, pw_exp = powerLawFit(imsd_to_fit, imsd_to_fit.index, nDrops, None)
    return imsd, fit, pw_exp


def get_emsd(imsd, fps, red_mask, nDrops, fit_range):
    id_start = np.where(imsd.index == fit_range[0])[0][0]
    id_end = np.where(imsd.index == fit_range[-1])[0][0] + 1
    MSD = np.array(imsd)
    MSD_b = [MSD[:, ~red_mask].mean(axis = 1),
                MSD[:, ~red_mask].std(axis = 1)]
    MSD_r = [MSD[:, red_mask].mean(axis = 1),
                MSD[:, red_mask].std(axis = 1)]
    # fit the EMSD in the fit_range
    fit_b, pw_exp_b = powerLawFit(MSD_b[0][id_start:id_end], fit_range, 1, MSD_b[1][id_start:id_end])
    fit_r, pw_exp_r = powerLawFit(MSD_r[0][id_start:id_end], fit_range, 1, MSD_r[1][id_start:id_end])
    results = {"fit_b": fit_b, "pw_exp_b": pw_exp_b, "fit_r": fit_r, "pw_exp_r": pw_exp_r}
    return MSD_b, MSD_r, results


def get_imsd_windowed(nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, fit_range):
    MSD_wind = []
    # fit region of the MSD
    fit_wind = np.zeros((nSteps, nDrops, len(fit_range)))
    pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
    for i in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
        temp, fit_wind[i], pw_exp_wind[i], = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, fit_range)
        MSD_wind.append(temp)
    return MSD_wind, fit_wind, pw_exp_wind


def get_emsd_windowed(imsd, x, fps, red_particle_idx, nSteps, maxLagtime, fit_range):
    id_start = np.where(imsd[0].index == fit_range[0])[0][0]
    id_end = np.where(imsd[0].index == fit_range[-1])[0][0] + 1
    EMSD_wind = np.array(imsd)
    EMSD_wind_b = [EMSD_wind[:, :, ~red_mask].mean(axis = 2), 
                    EMSD_wind[:, :, ~red_mask].std(axis = 2)]
    EMSD_wind_r = [EMSD_wind[:, :, red_mask].mean(axis = 2), 
                    EMSD_wind[:, :, red_mask].std(axis = 2)]

    # diffusive region of the MSD
    fit_wind_b = np.zeros((nSteps, len(fit_range)))
    pw_exp_wind_b = np.zeros((nSteps, 2, 2))
    fit_wind_r = np.zeros((nSteps, len(fit_range)))
    pw_exp_wind_r = np.zeros((nSteps, 2, 2))
    
    for i in tqdm(range(nSteps)):
        fit_wind_b[i], pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, id_start:id_end], x, 1, EMSD_wind_b[1][i, id_start:id_end])
        fit_wind_r[i], pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[0][i, id_start:id_end], x, 1, EMSD_wind_r[1][i, id_start:id_end])
    
    results = {"fit_wind_b":fit_wind_b, "pw_exp_wind_b":pw_exp_wind_b, "fit_wind_r":fit_wind_r,\
                            "pw_exp_wind_r":pw_exp_wind_r}

    return EMSD_wind_b, EMSD_wind_r, results


# get trajectories
def get_trajs(nDrops, red_particle_idx, trajs, subsample_factor, fps):
    blueTrajs = []
    redTrajs = []
    for i in range(0, nDrops):
        if i in red_particle_idx:
            p = trajs.loc[trajs.particle == i, ["x","y"]][::subsample_factor]
            redTrajs.append(Trajectory(p.x, p.y, dt = 1/fps*subsample_factor, traj_id=i, diff_est={"method":DiffMethod.LINEAR_DIFF, 
                                                                                  "window_type": WindowType.CENTRAL}))
        else:
            p = trajs.loc[trajs.particle == i, ["x","y"]][::subsample_factor]
            blueTrajs.append(Trajectory(p.x, p.y, dt = 1/fps*subsample_factor, traj_id=i))
    return blueTrajs, redTrajs


# get speed distributions windowed in time
def speed_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs, subsample_factor, fps):
    v_blue_wind = []
    v_red_wind = []
    for k in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
        blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajs_wind, subsample_factor, fps)
        v_blue_wind.append(ys.speed_ensemble(blueTrajs, step=1))
        v_red_wind.append(ys.speed_ensemble(redTrajs, step=1))
    return v_blue_wind, v_red_wind
    

# get turning angles distributions windowed in time
def turning_angles_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs, subsample_factor, fps):
    theta_blue_wind = []
    theta_red_wind = []
    for k in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
        blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajs_wind, subsample_factor, fps)
        theta_blue_wind.append(ys.turning_angles_ensemble(blueTrajs, centered= True))
        theta_red_wind.append(ys.turning_angles_ensemble(redTrajs, centered= True))
    return theta_blue_wind, theta_red_wind


# 2D Maxwell-Boltzmann distribution
def MB_2D(v, sigma):
    return v/(sigma**2) * np.exp(-v**2/(2*sigma**2))

# Normal distribution
def normal_distr(x, sigma, mu):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

def lorentzian_distr(x, gamma, x0):
    return 1/np.pi * gamma / ((x-x0)**2 + gamma**2)

# Generalized 2D Maxwell-Boltzmann distribution
def MB_2D_generalized(v, sigma, beta, A):
    return A*v * np.exp(-v**beta/(2*sigma**beta))

# Power Law 
def powerLaw(x, a, k):
    return a*x**k

# Histogram fit
def fit_hist(y, bins_, distribution, p0_):
    bins_c = bins_[:-1] + np.diff(bins_) / 2
    bin_heights, _ = np.histogram(y, bins = bins_, density = True)
    ret, pcov = curve_fit(distribution, bins_c, bin_heights, p0 = p0_)
    ret_std = np.sqrt(np.diag(pcov))
    return ret, ret_std

def vacf_windowed(trajectories, nDrops, red_particle_idx, trajs_wind, subsample_factor, fps):        
    vacf_b_wind = []
    vacf_b_std_wind = []
    vacf_r_wind = []
    vacf_r_std_wind = []
    
    for k in tqdm(range(nSteps)):
        trajs_wind = trajectories.loc[trajectories.frame.between(startFrames[k], endFrames[k])]
        blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajs_wind, subsample_factor, fps)
        temp = ys.vacf(blueTrajs, time_avg = True, lag = maxLagtime)
        vacf_b_wind.append(temp[0])
        vacf_b_std_wind.append(temp[1])
        temp = ys.vacf(redTrajs, time_avg = True, lag = maxLagtime)
        vacf_r_wind.append(temp[0])
        vacf_r_std_wind.append(temp[1])

    vacf_b_wind = pd.DataFrame(vacf_b_wind)
    vacf_b_std_wind = pd.DataFrame(vacf_b_std_wind)
    vacf_r_wind = pd.DataFrame(vacf_r_wind)
    vacf_r_std_wind = pd.DataFrame(vacf_r_std_wind)
    vacf_r_wind.columns = vacf_r_wind.columns.astype(str)
    vacf_b_wind.columns = vacf_b_wind.columns.astype(str)
    vacf_b_std_wind.columns = vacf_b_std_wind.columns.astype(str)
    vacf_r_std_wind.columns = vacf_r_std_wind.columns.astype(str)
    vacf_b_wind.to_parquet(f"./{analysis_data_path}/vacf/vacf_b_wind.parquet")
    vacf_b_std_wind.to_parquet(f"./{analysis_data_path}/vacf/vacf_b_std_wind.parquet")
    vacf_r_wind.to_parquet(f"./{analysis_data_path}/vacf/vacf_r_wind.parquet")
    vacf_r_std_wind.to_parquet(f"./{analysis_data_path}/vacf/vacf_r_std_wind.parquet")
    return vacf_b_wind, vacf_b_std_wind, vacf_r_wind, vacf_r_std_wind
""" 
@joblib.delayed
def rdf_frame(frame, COORDS, rList, dr, rho):
    coords = COORDS[frame*nDrops:(frame+1)*nDrops,:]
    kd = KDTree(coords)
    avg_n = np.zeros(len(rList))
    for i, r in enumerate(rList):
        a = kd.query_ball_point(coords, r + 20)
        b = kd.query_ball_point(coords, r)
        n1 = 0
        for j in a:
            n1 += len(j) - 1
        n2 = 0
        for j in b:
            n2 += len(j) - 1
        avg_n[i] = n1/len(a) - n2/len(b)
    rdf = avg_n/(np.pi*(dr**2 + 2*rList*dr)*rho)
    return rdf


def get_rdf(run_analysis_verb, nFrames, trajectories, rList, dr, rho):
    
    if run_analysis_verb:
        COORDS = np.array(trajectories.loc[:, ["x","y"]])
        parallel = joblib.Parallel(n_jobs = -2)
        rdf = parallel(
            rdf_frame(frame, COORDS, rList, dr, rho)
            for frame in tqdm(range(nFrames))
        )
        rdf = np.array(rdf)
        rdf_df = pd.DataFrame(rdf)
        # string columns for parquet filetype
        rdf_df.columns = [f"{r}" for r in rList]
        rdf_df.to_parquet(f"./{analysis_data_path}/rdf/rdf.parquet")

    elif not run_analysis_verb :
        try:
            rdf = np.array(pd.read_parquet(f"./{analysis_data_path}/rdf/rdf.parquet"))
        except: 
            raise ValueError("rdf data not found. Run analysis verbosely first.")
    else: 
        raise ValueError("run_analysis_verb must be True or False")
    return rdf
"""

@joblib.delayed
def rdf_frame(frame, COORDS_blue, n_blue, COORDS_red, n_red, rList, dr, rho_b, rho_r):
    coords_blue = COORDS_blue[frame*n_blue:(frame+1)*n_blue, :]
    coords_red = COORDS_red[frame*n_red:(frame+1)*n_red, :]
    kd_blue = KDTree(coords_blue)
    kd_red = KDTree(coords_red)

    avg_b = np.zeros(len(rList))
    avg_r = np.zeros(len(rList))
    avg_br = np.zeros(len(rList))
    avg_rb = np.zeros(len(rList))

    for i, r in enumerate(rList):
        # radial distribution function for Blue droplets
        a = kd_blue.query_ball_point(coords_blue, r + dr)
        b = kd_blue.query_ball_point(coords_blue, r)
        avg_b[i] = (sum(len(j) - 1 for j in a) / len(a)) - (sum(len(j) - 1 for j in b) / len(b))

        # radial distribution function for Red droplets
        a = kd_red.query_ball_point(coords_red, r + dr)
        b = kd_red.query_ball_point(coords_red, r)
        avg_r[i] = (sum(len(j) - 1 for j in a) / len(a)) - (sum(len(j) - 1 for j in b) / len(b))

        # radial distribution function for blue-Red droplets
        a = kd_blue.query_ball_point(coords_red, r + dr)
        b = kd_blue.query_ball_point(coords_red, r)
        avg_br[i] = (sum(len(j) - 1 for j in a) / len(a)) - (sum(len(j) - 1 for j in b) / len(b))

        # radial distribution function for red-Blue droplets
        a = kd_red.query_ball_point(coords_blue, r + dr) 
        b = kd_red.query_ball_point(coords_blue, r)
        avg_rb[i] = (sum(len(j) - 1 for j in a) / len(a)) - (sum(len(j) - 1 for j in b) / len(b))

    rdf_b = avg_b/(np.pi*(dr**2 + 2*rList*dr)*rho_b)
    rdf_r = avg_r/(np.pi*(dr**2 + 2*rList*dr)*rho_r)
    rdf_br = avg_br/(np.pi*(dr**2 + 2*rList*dr)*rho_b)
    rdf_rb = avg_rb/(np.pi*(dr**2 + 2*rList*dr)*rho_r)
    return rdf_b, rdf_r, rdf_br, rdf_rb

def get_rdf(frames, trajectories, red_particle_idx, rList, dr, rho_b, rho_r, n_blue, n_red):
    COORDS_blue = np.array(trajectories.loc[~trajectories.particle.isin(red_particle_idx), ["x","y"]])
    COORDS_red = np.array(trajectories.loc[trajectories.particle.isin(red_particle_idx), ["x","y"]])
    parallel = joblib.Parallel(n_jobs = -2)
    rdf = parallel(
        rdf_frame(frame, COORDS_blue, n_blue, COORDS_red, n_red, rList, dr, rho_b, rho_r)
        for frame in tqdm(frames)
    )
    return np.array(rdf)

@joblib.delayed
def rdf_center_frame(frame, COORDS, r_c, rList, dr, rho):
    coords = COORDS[frame*nDrops:(frame+1)*nDrops,:]
    kd = KDTree(coords)
    avg_n = np.zeros(len(rList))
    for i, r in enumerate(rList):
        # find all the points within r + dr
        a = kd.query_ball_point(r_c, r + dr)
        n1 = len(a) 
        # find all the points within r + dr
        b = kd.query_ball_point(r_c, r)
        n2 = len(b)
        avg_n[i] = n1 - n2
    rdf = avg_n/(np.pi*(dr**2 + 2*rList*dr)*rho)
    return rdf

def get_rdf_center(frames, trajectories, r_c, rList, dr, rho):
    COORDS = np.array(trajectories.loc[:,["x","y"]])
    parallel = joblib.Parallel(n_jobs = -2)
    rdf_c = parallel(
        rdf_center_frame(frame, COORDS, r_c, rList, dr, rho)
        for frame in tqdm( frames )
    )
    rdf_c = np.array(rdf_c)
    return rdf_c

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
    pxDimension = 90/875 # 90 mm / 875 mm = 0.10285714285714285 mm/mm
    data_preload_path = f'/Volumes/ExtremeSSD/UNI/h5_data_thesis/49b-1r/part1.h5'

    data_path = "../tracking/49b_1r/49b_1r_pre_merge/df_linked.parquet"
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
    
original_trajectories = pd.read_parquet(data_path)
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

factors = np.linspace(.5, 3, 20)
n_conn_components = np.zeros(len(factors))
for id, factor in enumerate(factors):
    cutoff_distance = factor * 2 * np.mean(trajectories.r.values.reshape(nFrames, nDrops), axis = 1)/pxDimension
    frame = 1500 * fps
    X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
    # create dictionary with positions of the agents
    dicts = {}
    for i in range(len(X)):
        dicts[i] = (X[i, 0], X[i, 1])
    # generate random geometric graph with cutoff distance 2.2 times the mean diameter the droplets have at that frame
    G = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos = dicts, dim = 2)
    n_conn_components[id] = nx.number_connected_components(G)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(factors, n_conn_components)
ax.set(xlabel = "Factor", ylabel = "Number of connected components", title = f"Number of connected components vs cutoff factor")
ax.grid(linewidth = 0.2)
ax.axvline(x = 1.5, color = 'r', linestyle = '--')
plt.savefig(f"./{pdf_res_path}/connected_components.pdf", bbox_inches='tight')
plt.close()

factor = 1.5
cutoff_distance = factor * 2 * np.mean(trajectories.r.values.reshape(nFrames, nDrops), axis = 1)/pxDimension

frame = 1500 * fps
X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
# create dictionary with positions of the agents
dicts = {}
for i in range(len(X)):
    dicts[i] = (X[i, 0], X[i, 1])
# generate random geometric graph with cutoff distance 2.2 times the mean diameter the droplets have at that frame
G = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos = dicts, dim = 2)
node_pos = nx.get_node_attributes(G, 'pos')
vor = Voronoi(np.asarray(list(node_pos.values())))

fig, ax = plt.subplots(figsize=(6, 6))
voronoi_plot_2d(vor, ax = ax, show_vertices = False, line_colors = 'orange', line_width = 2, line_alpha = 0.6, point_size = 2)
nx.draw(G, pos = node_pos, node_size = 25, node_color = colors, with_labels = False, ax=ax)
ax.set(xlim = (xmin, xmax), ylim = (ymax, ymin), title = f"Random Geometric Graph at {int(frame/fps)} s -- {system_name}", xlabel="x [px]", ylabel="y [px]")
if save_verb:
    plt.savefig(f"./{res_path}/graph/random_geometric_graph_{factor}.png", bbox_inches='tight')
    plt.savefig(f"./{pdf_res_path}/graph/random_geometric_graph_{factor}.pdf", bbox_inches='tight')
if show_verb:
    plt.show()
else:
    plt.close()

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

if 1:
    result = []
    for frame in tqdm(frames):
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
    df_graph.to_parquet(f"{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet")
else:
    print(f"Import data with factor {factor}")
    df_graph = pd.read_parquet(f"{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet")
    label = np.array(df_graph.columns)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(df_graph.frame.unique()/fps, df_graph.groupby("frame").n_clusters.mean())
ax.set(xlabel="Time [s]", ylabel="n_clusters", title=f"Number of clusters - {system_name}")
ax.grid(linewidth=0.2)
if save_verb:
    plt.savefig(f"./{res_path}/graph/n_clusters.png", bbox_inches='tight')
    plt.savefig(f"./{pdf_res_path}/graph/n_clusters.pdf", bbox_inches='tight')
if show_verb:
    plt.show()
else:
    plt.close()

x = df_graph.iloc[:, 1:].values # to skip the frame column
x = StandardScaler().fit_transform(x) # normalizing the features
x = x/np.std(x)
print(x.shape, np.mean(x), np.std(x))
normalized_df = pd.DataFrame(x, columns=label[1:])

# explained variance ratio with PCA
pca = PCA(n_components=label.shape[0]-2)
p_components = pca.fit_transform(x)
fig, ax = plt.subplots(1, 1, figsize = (6, 3))
ax.plot(np.arange(1, p_components.shape[1]+1, 1), pca.explained_variance_ratio_.cumsum())
ax.set(xlabel = 'N of components', ylabel = f'Cumulative explained variance ratio', title = f'PCA scree plot - {system_name}') 
ax.grid(linewidth = 0.2)
if save_verb:
    plt.savefig(f'{res_path}/graph/scree_plot.png', bbox_inches='tight')
    plt.savefig(f'{pdf_res_path}/graph/scree_plot.pdf', bbox_inches='tight')
if show_verb:
    plt.show()
else:
    plt.close()

pca = PCA(n_components=3)
p_components = pca.fit_transform(x)
expl_variance_ratio = np.round(pca.explained_variance_ratio_, 2)
print('Explained variation per principal component: {}'.format(expl_variance_ratio))
principal_df = pd.DataFrame(data = p_components, columns = ['pc1', 'pc2', 'pc3'])
principal_df['frame'] = df_graph.frame.values.astype(int)


fig, axs = plt.subplots(1, 3, figsize = (12, 4))
scatt = axs[0].scatter(principal_df.pc1, principal_df.pc2, c = principal_df.frame, cmap='viridis', s=10)
axs[0].set(xlabel = f'PC1 ({expl_variance_ratio[0]})', ylabel = f'PC2 ({expl_variance_ratio[1]})')
axs[1].scatter(principal_df.pc2, principal_df.pc3, c=principal_df.frame, cmap='viridis', s=10)
axs[1].set(xlabel = f'PC2 ({expl_variance_ratio[1]})', ylabel = f'PC3 ({expl_variance_ratio[2]})')
axs[2].scatter(principal_df.pc1, principal_df.pc3, c=principal_df.frame, cmap='viridis', s=10)
axs[2].set(xlabel = f'PC1 ({expl_variance_ratio[0]})', ylabel = f'PC3 ({expl_variance_ratio[2]})')
plt.suptitle(f'PCA projections - {system_name}')
axs[0].grid()
axs[1].grid()
axs[2].grid()
#fig.colorbar(scatt, ax=axs, label='frame', orientation='horizontal', shrink=0.5)
plt.tight_layout()
if save_verb:
    plt.savefig(f'{res_path}/graph/pca.png', bbox_inches='tight')
    plt.savefig(f'{pdf_res_path}/graph/pca.png', bbox_inches='tight', dpi = 500)
if show_verb:
    plt.show()
else:
    plt.close()

loadings = pca.components_

fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize = (10, 6), sharex=True, sharey=True)
ax.plot(loadings[0], 'r', label = 'PC1')
ax1.plot(loadings[1], 'b', label = 'PC2')
ax2.plot(loadings[2], 'y', label = 'PC3')
ax.set(ylabel = 'PC1', title = f'PCA components - {system_name}', ylim=(-1, 1))
ax1.set(ylabel = 'PC2', ylim=(-1,1))
ax2.set(ylabel = 'PC3', xlabel = 'features', ylim=(-1,1), xticks = np.arange(loadings.shape[1]))
ax2.set_xticklabels(df_graph.columns[1:], rotation = 45)
ax.grid(linewidth = 0.5)
ax1.grid(linewidth = 0.5)
ax2.grid(linewidth = 0.5)
fig.tight_layout()
if save_verb:
    plt.savefig(f'{res_path}/graph/pca_components.png', bbox_inches='tight')
    plt.savefig(f'{pdf_res_path}/graph/pca_components.pdf', bbox_inches='tight')
if show_verb:
    plt.show()
else:
    plt.close()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
scatterplot = ax.scatter(principal_df.pc1, principal_df.pc2, principal_df.pc3, c = principal_df.frame, cmap = 'viridis', s = 1, rasterized=True)
ax.grid(linewidth = 0.2)
fig.colorbar(scatterplot, ax=ax, shrink=0.6, label = 'frame', orientation = 'horizontal')
ax.set(xlabel = f'PC1 ({expl_variance_ratio[0]})', ylabel = f'PC2 ({expl_variance_ratio[1]})', zlabel = f'PC3 ({expl_variance_ratio[2]})')
if 0:
    # save to gif
    import imageio
    images = []
    for n in range(0, 250):
        if n >= 5:
            ax.azim = ax.azim+1.1
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        images.append(image.reshape(1000, 1000, 3)) ## 
    imageio.mimsave(f'{res_path}/graph/pca.gif', images)
if save_verb:
    plt.savefig(f'{res_path}/graph/pca_3d.png', bbox_inches='tight')
    plt.savefig(f'{pdf_res_path}/graph/pca_3d.png', bbox_inches='tight', dpi = 500)
if show_verb:
    plt.show()
else:
    plt.close()