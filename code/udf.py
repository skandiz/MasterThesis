import numpy as np
import trackpy as tp
from scipy.optimize import curve_fit
from yupi import Trajectory

# get trajectories
def get_trajectories(nDrops, red_particle_idx, rawTrajs, smoothTrajs):
    # raw trajectories
    blueTrajs = []
    redTraj = []
    for i in range(0, nDrops):
        if i == red_particle_idx:
            p = rawTrajs.loc[rawTrajs.particle == i, ["x","y"]]
            redTraj.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
        else:
            p = rawTrajs.loc[rawTrajs.particle == i, ["x","y"]]
            blueTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
        
    # smooth trajectories
    blueTrajs_smooth = []
    redTraj_smooth = []
    for i in range(0, nDrops):
        if i == red_particle_idx:
            p = smoothTrajs.loc[smoothTrajs.particle == i, ["x","y"]]
            redTraj_smooth.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
        else:
            p = smoothTrajs.loc[smoothTrajs.particle == i, ["x","y"]]
            blueTrajs_smooth.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
            
    return blueTrajs, redTraj, blueTrajs_smooth, redTraj_smooth

# Power Law fit
def powerLawFit(funct, fit, powerlawExponents, nDrops):
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


