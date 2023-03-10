import numpy as np
import trackpy as tp
from scipy.optimize import curve_fit
from yupi import Trajectory
import yupi.stats as ys
from tqdm import tqdm

# Power Law fit
def powerLaw(x, a, k):
    return a*x**k


def powerLawFit(f, x, nDrops, yerr):
    if nDrops == 1:
        ret = np.zeros((2, 2))
        ret[0], pcov = curve_fit(powerLaw, x, f, p0 = [1., 1.])
        ret[1] = np.sqrt(np.diag(pcov))
        fit = ret[0, 0] * x**ret[0, 1]
    else:
        fit = np.zeros((nDrops, f.shape[0])) # np.zeros(f.shape).T
        ret = np.zeros((nDrops, 2, 2))
        for i in range(nDrops):
            if yerr is None:
                ret[i, 0], pcov = curve_fit(powerLaw, x, f[i], p0 = [1., 1.])
            else:
                ret[i, 0], pcov = curve_fit(powerLaw, x, f[i], p0 = [1., 1.], sigma = yerr)
            ret[i, 1] = np.sqrt(np.diag(pcov))
            fit[i] = ret[i, 0, 0] * x**ret[i, 0, 1]
    return fit, ret 


def get_imsd(trajs, pxDimension, fps, maxLagtime, nDrops):
    imsd = tp.imsd(trajs, mpp = pxDimension, fps = fps, max_lagtime = maxLagtime)
    fit, pw_exp = powerLawFit(imsd[1:], imsd[1:].index, nDrops, None)
    return imsd, fit, pw_exp


def get_emsd(imsd, x, red_particle_idx, nDrops):
    MSD = np.array(imsd)
    MSD_b = [MSD[:, [x for x in range(nDrops) if x != red_particle_idx]].mean(axis = 1),
                MSD[:,[x for x in range(nDrops) if x != red_particle_idx]].std(axis = 1)]
    MSD_r = MSD[:, red_particle_idx]
    fit_b, pw_exp_b = powerLawFit(MSD_b[0][9:], x, 1, MSD_b[1][9:])
    fit_r, pw_exp_r = powerLawFit(MSD_r[9:], x, 1, None)
    res = {"fit_b":fit_b, "pw_exp_b":pw_exp_b, "fit_r":fit_r, "pw_exp_r":pw_exp_r}
    return MSD_b, MSD_r, res



def get_imsd_windowed(nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, nDrops):
    MSD_wind = []
    fit_wind = np.zeros((nSteps, nDrops, maxLagtime-9))
    pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
    for i in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
        temp, fit_wind[i], pw_exp_wind[i] = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, nDrops)
        MSD_wind.append(temp)
    return MSD_wind, fit_wind, pw_exp_wind


def get_emsd_windowed(imsds, x, nDrops, red_particle_idx, nSteps, maxLagtime):
    EMSD_wind = np.array(imsds)
    EMSD_wind_b = [EMSD_wind[:, :, [x for x in range(nDrops) if x != red_particle_idx]].mean(axis = 2), 
                    EMSD_wind[:, :, [x for x in range(nDrops) if x != red_particle_idx]].std(axis = 2)]
    EMSD_wind_r = EMSD_wind[:, :, red_particle_idx]

    fit_wind_b = np.zeros((nSteps, maxLagtime-9))
    pw_exp_wind_b = np.zeros((nSteps, 2, 2))
    fit_wind_r = np.zeros((nSteps, maxLagtime-9))
    pw_exp_wind_r = np.zeros((nSteps, 2, 2))

    for i in tqdm(range(nSteps)):
        fit_wind_b[i], pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, 9:], x, 1, EMSD_wind_b[1][i, 9:])
        fit_wind_r[i], pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[i, 9:], x, 1, None)
    
    res = {"fit_wind_b":fit_wind_b, "pw_exp_wind_b":pw_exp_wind_b, "fit_wind_r":fit_wind_r, "pw_exp_wind_r":pw_exp_wind_r}

    return EMSD_wind_b, EMSD_wind_r, res


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
    ret, pcov = curve_fit(distribution, bins_c, bin_heights, p0 = p0_)
    ret_std = np.sqrt(np.diag(pcov))
    return ret, ret_std