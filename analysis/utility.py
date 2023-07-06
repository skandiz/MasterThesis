import numpy as np
import trackpy as tp
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from yupi import Trajectory
import yupi.stats as ys
from tqdm import tqdm
import joblib

def get_smooth_trajs(trajs, nDrops, windLen, orderofPoly):
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    ret = trajs.copy()
    for i in range(nDrops):
        ret.loc[ret.particle == i, "x"] = savgol_filter(trajs.loc[trajs.particle == i].x.values, windLen, orderofPoly)
        ret.loc[ret.particle == i, "y"] = savgol_filter(trajs.loc[trajs.particle == i].y.values, windLen, orderofPoly)    
    return ret

def get_velocities(trajList):
        v = np.zeros((len(trajList), 2, len(trajList[0])), dtype=np.float64)
        for i in range(len(trajList)):
            v[i] = np.array(trajList[i].v).T
        return v

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
    # fit the diffusive region of the MSD --> 1s onwards
    fit, pw_exp = powerLawFit(imsd[1:], imsd[1:].index, nDrops, None)
    return imsd, fit, pw_exp


def get_emsd(imsd, x, fps, red_mask, nDrops):
    MSD = np.array(imsd)
    MSD_b = [MSD[:, ~red_mask].mean(axis = 1),
                MSD[:, ~red_mask].std(axis = 1)]
    MSD_r = [MSD[:, red_mask].mean(axis = 1),
                MSD[:, red_mask].std(axis = 1)]
    # fit the diffusive region of the MSD
    fit_b, pw_exp_b = powerLawFit(MSD_b[0][fps-1:], x, 1, MSD_b[1][fps-1:])
    fit_r, pw_exp_r = powerLawFit(MSD_r[0][fps-1:], x, 1, MSD_r[1][fps-1:])
    results = {"fit_b": fit_b, "pw_exp_b": pw_exp_b, "fit_r": fit_r, "pw_exp_r": pw_exp_r}
    return MSD_b, MSD_r, results


def get_imsd_windowed(nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, nDrops):
    MSD_wind = []
    # diffusive region of the MSD
    fit_wind = np.zeros((nSteps, nDrops, maxLagtime - fps+1))
    pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
    for i in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
        temp, fit_wind[i], pw_exp_wind[i], = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, nDrops)
        MSD_wind.append(temp)
    return MSD_wind, fit_wind, pw_exp_wind


def get_emsd_windowed(imsds, x, fps, red_mask, nSteps, maxLagtime):
    EMSD_wind = np.array(imsds)
    EMSD_wind_b = [EMSD_wind[:, :, ~red_mask].mean(axis = 2), 
                    EMSD_wind[:, :, ~red_mask].std(axis = 2)]
    EMSD_wind_r = [EMSD_wind[:, :, red_mask].mean(axis = 2), 
                    EMSD_wind[:, :, red_mask].std(axis = 2)]

    # diffusive region of the MSD
    fit_wind_b = np.zeros((nSteps, maxLagtime-fps+1))
    pw_exp_wind_b = np.zeros((nSteps, 2, 2))
    fit_wind_r = np.zeros((nSteps, maxLagtime-fps+1))
    pw_exp_wind_r = np.zeros((nSteps, 2, 2))
    
    for i in tqdm(range(nSteps)):
        fit_wind_b[i], pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, fps-1:], x, 1, EMSD_wind_b[1][i, fps-1:])
        fit_wind_r[i], pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[0][i, fps-1:], x, 1, EMSD_wind_r[1][i, fps-1:])
    
    results = {"fit_wind_b":fit_wind_b, "pw_exp_wind_b":pw_exp_wind_b, "fit_wind_r":fit_wind_r,\
                            "pw_exp_wind_r":pw_exp_wind_r}

    return EMSD_wind_b, EMSD_wind_r, results


# get trajectories
def get_trajs(nDrops, red_particle_idx, trajs):
    # raw trajectories
    blueTrajs = []
    redTrajs = []
    for i in range(0, nDrops):
        if i in red_particle_idx:
            p = trajs.loc[trajs.particle == i, ["x","y"]]
            redTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
        else:
            p = trajs.loc[trajs.particle == i, ["x","y"]]
            blueTrajs.append(Trajectory(p.x, p.y, dt = 1/10, traj_id=i))
    return blueTrajs, redTrajs


# get speed distributions windowed in time
def speed_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs, v_step, fps):
    v_blue_wind = []
    v_red_wind = []
    for k in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
        blueTrajs = []
        redTrajs = []
        for i in range(nDrops):
            if i in red_particle_idx:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                redTrajs.append(Trajectory(p.x, p.y, dt = 1/fps, traj_id=i))
            else:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                blueTrajs.append(Trajectory(p.x, p.y, dt = 1/fps, traj_id=i))
        v_blue_wind.append(ys.speed_ensemble(blueTrajs, step=v_step))
        v_red_wind.append(ys.speed_ensemble(redTrajs, step=v_step))
    return v_blue_wind, v_red_wind
    

# get turning angles distributions windowed in time
def theta_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs, fps):
    theta_blue_wind = []
    theta_red_wind = []
    for k in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
        blueTrajs = []
        redTrajs = []
        for i in range(nDrops):
            if i in red_particle_idx:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                redTrajs.append(Trajectory(p.x, p.y, dt = 1/fps, traj_id=i))
            else:
                p = trajs_wind.loc[trajs_wind.particle==i, ["x","y"]]
                blueTrajs.append(Trajectory(p.x, p.y, dt = 1/fps, traj_id=i))
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

# Histogram fit
def fit_hist(y, bins_, distribution, p0_):
    bins_c = bins_[:-1] + np.diff(bins_) / 2
    bin_heights, _ = np.histogram(y, bins = bins_, density = True)
    ret, pcov = curve_fit(distribution, bins_c, bin_heights, p0 = p0_)
    ret_std = np.sqrt(np.diag(pcov))
    return ret, ret_std

def vacf_vindowed(trajectories, raw):        
    vacf_b_wind = []
    vacf_b_std_wind = []
    vacf_r_wind = []
    vacf_r_std_wind = []
    
    for k in tqdm(range(nSteps)):
        trajs = trajectories.loc[trajectories.frame.between(startFrames[k], endFrames[k])]
        blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajs)
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
    return vacf_b_wind, vacf_b_std_wind, vacf_r_wind, vacf_r_std_wind

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

@joblib.delayed
def rdf_center_frame(frame, COORDS, r_c, rList, dr, rho):
    coords = COORDS[frame*nDrops:(frame+1)*nDrops,:]
    kd = KDTree(coords)
    avg_n = np.zeros(len(rList))
    for i, r in enumerate(rList):
        # find all the points within r+dr
        a = kd.query_ball_point(r_c, r + dr)
        n1 = len(a) 
        # find all the points within r+dr
        b = kd.query_ball_point(r_c, r)
        n2 = len(b)
        avg_n[i] = n1 - n2
    rdf = avg_n/(np.pi*(dr**2 + 2*rList*dr)*rho)
    return rdf

def get_rdf_center(run_analysis_verb, nFrames, trajectories, r_c, rList, dr, rho):
    
    if run_analysis_verb:
        COORDS = np.array(trajectories.loc[:,["x","y"]])
        parallel = joblib.Parallel(n_jobs = -2)
        rdf_c = parallel(
            rdf_center_frame(frame, COORDS, r_c, rList, dr, rho)
            for frame in tqdm( range(nFrames) )
        )
        rdf_c = np.array(rdf_c)
        rdf_c_df = pd.DataFrame(rdf_c)
        # string columns for parquet filetype
        rdf_c_df.columns = [str(i) for i in rList]
        pd.DataFrame(rdf_c_df).to_parquet(f"./{analysis_data_path}/rdf/rdf_center.parquet")
        
    if not run_analysis_verb :
        try: 
            rdf_c = np.array(pd.read_parquet(f"./{analysis_data_path}/rdf/rdf_center.parquet"))
        except: 
            raise ValueError("rdf data not found. Run analysis verbosely first.")
    return rdf_c