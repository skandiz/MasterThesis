#%% IMPORTS
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib ipympl
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['font.size'] = 8
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
import csv, json
import pims
from PIL import Image, ImageDraw
import cv2

from scipy.optimize import dual_annealing, linear_sum_assignment
from scipy.spatial import distance_matrix
from tqdm import tqdm

from utility import hough_preprocessing, hough_loc_frame, hough_feature_location,\
                    optimize_params, plot_opt_results

#%%
# SETUP
preload_load_data = False # takes 20 min
merge_frame = 32269
data = hough_preprocessing(pims.open('./data/movie.mp4'), 40, 55, 895, 910)
if preload_load_data: 
    data_preload = list(data[:merge_frame])
#%%
class TrajectoryProcessing:
    def __init__(self, traj_part, data_preload, correct_n):
        self.traj_part = traj_part
        self.frames = np.arange(0, len(data_preload), 1)
        self.correct_n = 50
        self.parameters = {"dp": 1.5, "minDist": 15, "param1": 100, "param2": 0.8, "minRadius": 15, "maxRadius": 25}

    def parameters_optimization(self, nSample, run_verb = False):
        if run_verb:
            if not os.path.exists(f"./results/tracking_data/hough/{self.traj_part}_optimization.csv"):
                # initialize the CSV file with the header if it does not exist
                with open(f"./results/tracking_data/hough/{self.traj_part}_opt.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['loss', 'dp', 'minDist', 'param1', 'param2', 'minRadius', 'maxRadius'])

            self.frames_opt = np.sort(random.sample(list(frames), nSample)) # randomly select 5000 frames to optimize --> change this number if needed
            # paramters of HoughCircles --> dp, minDist, param1, param2, minRadius, maxRadius
            self.init_guess =  [2, 8, 20, 0.8, 10, 35] # initial guess for the parameters
            self.params_bounds = [(1, 3), (5, 20), (20, 200), (0.3, 1), (5, 20), (20, 40)] # bounds for the parameters
            print("Starting optimization...")
            opt_result = dual_annealing(self.optimize_params, x0 = self.init_guess, \
                                        args = (data_preload, self.frames_opt, self.correct_n, self.traj_part),\
                                        bounds=self.params_bounds, maxfun=100)
            print(opt_result)
            plot_opt_results(opt_result, self.traj_part)
            self.parameters = {"dp": opt_result.x[0], "minDist": opt_result.x[1], "param1": opt_result.x[2],\
                                        "param2": opt_result.x[3], "minRadius": opt_result.x[4], "maxRadius": opt_result.x[5]}
        else:
            try:
                opt_result_df = pd.read_csv(f"./results/tracking_data/hough/{traj_part}_opt.csv").sort_values("loss", ascending=False)
                # the best set of parameters is the one that minimizes the loss
                optimized_parameters = {"dp": opt_result_df.iloc[-1].dp, "minDist": opt_result_df.iloc[-1].minDist,\
                                        "param1": opt_result_df.iloc[-1].param1, "param2": opt_result_df.iloc[-1].param2,\
                                        "minRadius": int(opt_result_df.iloc[-1].minRadius), "maxRadius": int(opt_result_df.iloc[-1].maxRadius)}
                parameters = optimized_parameters
                print("Optimized parameters:", parameters)
                plot_opt_results(opt_result_df, "param1")
            except:
                raise Exception("No optimization results found")

    def run_location(self, data_preload, frames, correct_n, parameters):
        with open(f'./results/tracking_data/hough/hough/hough_{self.traj_part}.txt', 'w') as f:
            f.write(json.dumps(parameters))
        hough_df, err_frames, error = self.hough_feature_location(data_preload, frames, correct_n, parameters, False)
        hough_df.to_parquet(f"./results/tracking_data/hough/hough_{self.traj_part}.parquet")

    def import_trackpy_trajectory(self):
        try:
            trackpy_df = pd.read_parquet(f"./results/tracking_data/trackpy_{self.traj_part}_sorted_and_colored.parquet")
        except:
            raise Exception(f"No {self.traj_part} trackpy data found, run analysis first")
        colors = trackpy_df.loc[trackpy_df.frame == 0].color.values
        return trackpy_df, colors

    def setup_hough_features_dataframe(self, hough_df, trackpy_df):
        hough_df = hough_df.replace(0, np.nan)
        hough_df.loc[:49, ["frame"]] = 0

        hough_df = hough_df.loc[hough_df.frame.between(0, max(trackpy_df.frame)), :]
        hough_df["particle"] = np.ones(len(hough_df), dtype=int)*(-1) # initialize particle id to -1
        hough_df["flag"] = np.zeros(len(hough_df), dtype=int) # add flag column to keep record of error frames in which trackpy position 
        err_frames = np.where(hough_df.groupby("frame").mean().x.isna())[0] # detect the frames in which hough circle detection failed

#%%
#trial = TrajectoryProcessing("pre_merge", data_preload, 50)
trial.parameters_optimization(5000)
#%%
startFrame = 0
endFrame = merge_frame
frames = np.arange(startFrame, endFrame, 1)
correct_n = 50
default_parameters = {"dp": 1.5, "minDist": 15, "param1": 100, "param2": 0.8, "minRadius": 15, "maxRadius": 25}
# choose which trajectory part to run --> "pre_merge" or "post_merge"
traj_part = "pre_merge"

optimization_verb = True
run_optimization_verb = False 
if optimization_verb:
    if run_optimization_verb:
        if not os.path.exists(f"./results/tracking_data/hough/{traj_part}_optimization.csv"):
            # initialize the CSV file with the header if it does not exist
            with open(f"./results/tracking_data/hough/{traj_part}_opt.csv", mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['loss', 'dp', 'minDist', 'param1', 'param2', 'minRadius', 'maxRadius'])

        frames_opt = np.sort(random.sample(list(frames), 5000)) # randomly select 5000 frames to optimize --> change this number if needed
        # paramters of HoighCircles --> dp, minDist, param1, param2, minRadius, maxRadius
        init_guess =  [2, 8, 20, 0.8, 10, 35] # initial guess for the parameters
        params_bounds = [(1, 3), (5, 20), (20, 200), (0.3, 1), (5, 20), (20, 40)] # bounds for the parameters
        print("Starting optimization...")
        opt_result = dual_annealing(optimize_params, x0 = init_guess, args = (data_preload, frames_opt, correct_n, traj_part),\
                                    bounds = params_bounds, maxfun=100)
        print(opt_result)
    else:
        try:
            opt_result_df = pd.read_csv(f"./results/tracking_data/hough/{traj_part}_opt.csv").sort_values("loss", ascending=False)
            # the best set of parameters is the one that minimizes the loss
            optimized_parameters = {"dp": opt_result_df.iloc[-1].dp, "minDist": opt_result_df.iloc[-1].minDist,\
                                    "param1": opt_result_df.iloc[-1].param1, "param2": opt_result_df.iloc[-1].param2,\
                                    "minRadius": int(opt_result_df.iloc[-1].minRadius), "maxRadius": int(opt_result_df.iloc[-1].maxRadius)}
            parameters = optimized_parameters
            print("Optimized parameters:", parameters)
            plot_opt_results(opt_result_df, "param1")
        except:
            raise Exception("No optimization results found")
else:
    parameters = default_parameters
    print("Default parameters:", parameters)

############################################################################################################
#                                      hough feature location                                              #
############################################################################################################

run_location = False
if run_location:
    # save to txt parameters:
    with open(f'./results/tracking_data/hough/hough/hough_{traj_part}.txt', 'w') as f:
        f.write(json.dumps(parameters))
    hough_df, err_frames, error = hough_feature_location(data_preload, frames, correct_n, parameters, False)
    hough_df.to_parquet(f"./results/tracking_data/hough/hough_{traj_part}.parquet")
else:
    try:
        parameters = json.load(open(f'./results/tracking_data/hough/hough_{traj_part}.txt'))
        hough_df = pd.read_parquet(f"./results/tracking_data/hough/hough_{traj_part}.parquet")
        print(parameters)
        display(hough_df)
    except:
        raise Exception(f"No {traj_part} data found, run analysis first")

############################################################################################################
#                                       import trackpy trajectory                                          #
############################################################################################################
try:
    trackpy_df = pd.read_parquet(f"./results/tracking_data/trackpy_{traj_part}_sorted_and_colored.parquet")
except:
    raise Exception(f"No {traj_part} trackpy data found, run analysis first")
colors = trackpy_df.loc[trackpy_df.frame == 0].color.values


############################################################################################################
#                                  setup hough features dataframe                                          #
############################################################################################################
hough_df = hough_df.replace(0, np.nan)
hough_df.loc[:49, ["frame"]] = 0
if traj_part == "pre_merge":
    correct_n = 50
elif traj_part == "post_merge":
    correct_n = 49
hough_df = hough_df.loc[hough_df.frame.between(0, max(trackpy_df.frame)), :]
hough_df["particle"] = np.ones(len(hough_df), dtype=int)*(-1) # initialize particle id to -1
hough_df["flag"] = np.zeros(len(hough_df), dtype=int) # add flag column to keep record of error frames in which trackpy position 
err_frames = np.where(hough_df.groupby("frame").mean().x.isna())[0] # detect the frames in which hough circle detection failed


############################################################################################################
#                            link hough features with trackpy results                                      #
############################################################################################################
# associate at each frame droplet ID from trackpy to the one from hough circles by linear sum assignment on the distance matrix
link_verb = False
if link_verb:
    print("Starting linking procedure...")
    hough_trackpy_df = hough_df.copy()
    for frame in tqdm(range(max(trackpy_df.frame)+1)):
        # frames with error in hough circle detection --> use trackpy result
        if frame in err_frames:
            # change flag to 1 to denote the fact that trackpy result is used
            hough_trackpy_df.loc[hough_trackpy_df.frame == frame, "flag"] = np.ones(correct_n, dtype=int)
            hough_trackpy_df.loc[hough_trackpy_df.frame == frame, ["x", "y", "frame", "particle"]] = trackpy_df.loc[trackpy_df.frame == frame, ["x", "y", "frame", "particle"]]

        hough_frame = hough_trackpy_df.loc[hough_df.frame == frame]
        trackpy_frame = trackpy_df.loc[trackpy_df.frame == frame]
        # compute distance matrix between hough and trackpy positions
        dist = distance_matrix(hoguh_frame[["x", "y"]].values, trackpy_frame[["x", "y"]].values)
        # solve assignment problem
        row_ind, col_ind = linear_sum_assignment(dist)
        # set particle id in hough_trackpy_df to the one from trackpy
        hough_trackpy_df.loc[hough_trackpy_df.frame == frame, ["particle"]] = trackpy_frame.loc[:, ["particle"]].values[col_ind,:]

    # set colors to hough_trackpy_df
    c = []
    for p in hough_trackpy_df.particle:
        c.append(colors[p])
    hough_trackpy_df["color"] = c
    hough_trackpy_df.sort_values(by=["frame", "particle"], inplace=True)
    hough_trackpy_df = hough_trackpy_df.reset_index().drop(columns=["index"])
    display(hough_trackpy_df)
    hough_trackpy_df.to_parquet(f"./results/tracking_data/hough/tracking_ht_{traj_part}.parquet")
else: 
    try:
        hough_trackpy_df = pd.read_parquet(f"./results/tracking_data/hough/tracking_ht_{traj_part}.parquet")
    except:
        print("No parquet file found, run the linking code")
# %%
