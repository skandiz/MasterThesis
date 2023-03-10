import numpy as np
import trackpy as tp
from scipy.optimize import curve_fit
from yupi import Trajectory
import yupi.stats as ys
from tqdm import tqdm

# get trajectories
def get_trajs(nDrops, red_particle_idx, trajs):
    # raw trajectories
    blueTrajs = []
    redTraj = []
    for i in range(0, nDrops):
        if i == red_particle_idx:
            p = trajs.loc[trajs.particle == i, ["x","y"]]
            redTraj.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
        else:
            p = trajs.loc[trajs.particle == i, ["x","y"]]
            blueTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))

    return blueTrajs, redTraj

# get speed distributions windowed in time
def speed_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs):
    v_blue_wind = []
    v_red_wind = []
    for k in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
        blueTrajs = []
        redTraj = []

        for i in range(0, nDrops):
            if i == red_particle_idx:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                redTraj.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
            else:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                blueTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
            
        v_blue_wind.append(ys.speed_ensemble(blueTrajs, step=10))
        v_red_wind.append(ys.speed_ensemble(redTraj, step=10))
    
    return v_blue_wind, v_red_wind
    
# get turning angles distributions windowed in time
def theta_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs):
    theta_blue_wind = []
    theta_red_wind = []
    for k in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
        blueTrajs = []
        redTraj = []

        for i in range(0, nDrops):
            if i == red_particle_idx:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                redTraj.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
            else:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                blueTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))

        theta_blue_wind.append(ys.turning_angles_ensemble(blueTrajs, centered= True))
        theta_red_wind.append(ys.turning_angles_ensemble(redTraj, centered= True))
        
    return theta_blue_wind, theta_red_wind


# Power Law fit
def powerLawFit(funct, nDrops):
    fit = np.zeros(funct.shape).T
    powerlawExponents = np.zeros(nDrops)
    for i in range(nDrops):
        powerlawFit = tp.utils.fit_powerlaw(funct[i], plot = False) 
        powerlawExponents[i] = powerlawFit.n.values 
        fit[i] = powerlawFit.A.values * np.array(funct.index)**powerlawExponents[i] 
    return fit, powerlawExponents

# 2D Maxwell-Boltzmann distribution
def MB_2D(v, sigma):
    return v/(sigma**2) * np.exp(-v**2/(2*sigma**2))

# Normal distribution
def normal_distr(x, sigma, mu):   
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

# Histogram fit
def fit_hist(y, bins_, distribution, p0_):
    bins_c = bins_[:-1] + np.diff(bins_) / 2
    bin_heights, _ = np.histogram(y, bins = bins_, density = True)
    ret, pcov = curve_fit(distribution, bins_c , bin_heights, p0 = p0_)
    ret_std = np.sqrt(np.diag(pcov))
    
    return ret, ret_std


