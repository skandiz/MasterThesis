import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from tqdm import tqdm

from yupi import Trajectory
import yupi.stats as ys

from utility import get_imsd, get_imsd_windowed, get_emsd, get_emsd_windowed, fit_hist, MB_2D,\
                    normal_distr, get_trajs, speed_windowed, theta_windowed


rawTrajs = pd.read_parquet("../data/tracking/pre_merge_tracking.parquet")
red_particle_idx = 17
rawTrajs.loc[rawTrajs.particle != red_particle_idx, ["color"]] = "#00007F"
rawTrajs.loc[rawTrajs.particle == red_particle_idx, ["color"]] = "#FF0000"
colors = rawTrajs.loc[rawTrajs.frame == 0, 'color'].values
nDrops = len(rawTrajs.loc[rawTrajs.frame==0])
nFrames = max(rawTrajs.frame) + 1
print(f"nDrops:{nDrops}")
print(f"nFrames:{nFrames} --> {nFrames/10:.2f} s")


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

# Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
smoothTrajs = rawTrajs.copy()
windLen = 30
orderofPoly = 2
for i in range(nDrops):
    smoothTrajs.loc[smoothTrajs.particle == i, "x"] = savgol_filter(rawTrajs.loc[rawTrajs.particle == i].x.values,
                                                                    windLen, orderofPoly)
    smoothTrajs.loc[smoothTrajs.particle == i, "y"] = savgol_filter(rawTrajs.loc[rawTrajs.particle == i].y.values,
                                                                    windLen, orderofPoly)        
maxLagtime = 1000
x = np.arange(0, 100, 0.1) # with initial point

blueTrajs, redTraj = get_trajs(nDrops, red_particle_idx, rawTrajs)
blueTrajs_smooth, redTraj_smooth = get_trajs(nDrops, red_particle_idx, smoothTrajs)

def vacf_vindowed(run_verb, save_verb, trajectories, raw):        
    if run_verb:
        vacf_b_wind = []
        vacf_b_std_wind = []
        vacf_r_wind = []
        vacf_sigmas = np.zeros((nSteps, 2))

        for k in tqdm(range(nSteps)):
            trajs = trajectories.loc[trajectories.frame.between(startFrames[k], endFrames[k])]
            blueTrajs, redTraj = get_trajs(nDrops, red_particle_idx, trajs)
            temp = ys.vacf(blueTrajs, time_avg = True, lag = maxLagtime)
            vacf_b_wind.append(temp[0])
            vacf_b_std_wind.append(temp[1])
            #vacf_sigmas[k, 0] = np.mean(temp[2])

            temp  = ys.vacf(redTraj, time_avg = True, lag = maxLagtime)
            vacf_r_wind.append(temp[0])
            #vacf_sigmas[k, 1] = np.mean(temp[2])

        
        #vacf_sigmas = pd.DataFrame(vacf_sigmas)
        vacf_b_wind = pd.DataFrame(vacf_b_wind)
        vacf_b_std_wind = pd.DataFrame(vacf_b_std_wind)
        vacf_r_wind = pd.DataFrame(vacf_r_wind)

        if save_verb:
            if raw: 
                path = "../data/analysis/vacovf/raw/"
            else: 
                path = "../data/analysis/vacovf/smooth/"
        
            vacf_b_wind.to_csv(path + "vacf_blue_wind.csv")
            vacf_b_std_wind.to_csv(path + "vacf_blue_std_wind.csv")
            vacf_r_wind.to_csv(path + "vacf_red_wind.csv")
            #vacf_sigmas.to_csv(path + "vacf_sigmas.csv")
    else:
        if raw: 
            path = "../data/analysis/vacovf/raw/"
        else: 
            path = "../data/analysis/vacovf/smooth/"

        vacf_b_wind = pd.read_csv(path + "vacf_blue_wind.csv", index_col=0)
        vacf_b_std_wind = pd.read_csv(path + "vacf_blue_std_wind.csv", index_col=0)
        vacf_r_wind = pd.read_csv(path + "vacf_red_wind.csv", index_col=0)
        #vacf_sigmas = pd.read_csv(path + "vacf_sigmas.csv", index_col=0)

    return vacf_b_wind, vacf_b_std_wind, vacf_r_wind#, vacf_sigmas

print("Start of vacf analysis --  Raw Trajectories")
a, b, c = vacf_vindowed(True, True, rawTrajs, True)

print("Start of vacf analysis --  Smooth Trajectories")
a, b, c = vacf_vindowed(True, True, smoothTrajs, False)