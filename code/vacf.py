import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
from yupi import Trajectory
import yupi.graphics as yg
import yupi.stats as ys

rawTrajs = pd.read_csv("../data/csv/Processed_data2.csv")
red_particle_idx = 17
nDrops = len(rawTrajs.loc[rawTrajs.frame==0])
nFrames = max(rawTrajs.frame) + 1
smoothTrajs = rawTrajs.copy()
windLen = 30
orderofPoly = 2
for i in range(nDrops):
    smoothTrajs.loc[smoothTrajs.particle == i, "x"] = savgol_filter(rawTrajs.loc[rawTrajs.particle == i].x.values, windLen, orderofPoly)
    smoothTrajs.loc[smoothTrajs.particle == i, "y"] = savgol_filter(rawTrajs.loc[rawTrajs.particle == i].y.values,  windLen, orderofPoly)

maxLagtime = 1000 # maximum lagtime to be considered --> 100s
window = 3200
stride = 100

print(f"window of {window/10} s, stride of {stride/10} s")

startFrames = np.arange(0, nFrames-window+1, stride, dtype=int)
endFrames = startFrames + window
nSteps = len(startFrames)
print(f"number of steps: {nSteps}")

vacf_blue_windowed_smooth = []
vacf_red_windowed_smooth = []

for k in tqdm(range(nSteps)):
    trajs = smoothTrajs.loc[smoothTrajs.frame.between(startFrames[k], endFrames[k])]
    blueTrajs = []
    redTraj = []

    for i in range(0, nDrops):
        if i != red_particle_idx:
            p = trajs.loc[trajs.particle==i, ["x","y"]]
            redTraj.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
        if i == red_particle_idx:
            p = trajs.loc[trajs.particle==i, ["x","y"]]
            blueTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))

    vacf_blue_windowed_smooth.append(ys.vacf(blueTrajs, time_avg=True, lag=maxLagtime)[0])
    vacf_red_windowed_smooth.append(ys.vacf(redTraj, time_avg=True, lag=maxLagtime)[0])
    
pd.DataFrame(vacf_blue_windowed_smooth).to_csv("../data/csv/vacf_blue_windowed_smooth.csv")
pd.DataFrame(vacf_red_windowed_smooth).to_csv("../data/csv/vacf_red_windowed_smooth.csv")