import joblib
from scipy.signal import savgol_filter
from tqdm import tqdm
import cv2
import numpy as np
from scipy.optimize import curve_fit
import trackpy as tp
import pandas as pd
from scipy.spatial import KDTree, cKDTree, Voronoi, voronoi_plot_2d, ConvexHull
import yupi.stats as ys
from yupi import Trajectory, WindowType, DiffMethod

def trim_up_to_char(s, char):
    index = s.find(char)
    if index != -1:
        return s[:index]
    return s

def get_smooth_trajs(trajs, nDrops, windLen, orderofPoly):
    # Trajectory Smoothing: using a Savgol Filter in order to drop the noise due to the tracking procedure
    ret = trajs.copy()
    for i in range(nDrops):
        ret.loc[ret.particle == i, 'x'] = savgol_filter(trajs.loc[trajs.particle == i].x.values, windLen, orderofPoly)
        ret.loc[ret.particle == i, 'y'] = savgol_filter(trajs.loc[trajs.particle == i].y.values, windLen, orderofPoly)    
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
    fit, pw_exp = powerLawFit(imsd_to_fit, imsd_to_fit.index, len(trajs.particle.unique()), None)
    return imsd, fit, pw_exp


def get_emsd(imsd, fps, red_mask, fit_range):
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
    results = {'fit_b': fit_b, 'pw_exp_b': pw_exp_b, 'fit_r': fit_r, 'pw_exp_r': pw_exp_r}
    return MSD_b, MSD_r, results


def get_imsd_windowed(nDrops, nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, fit_range):
    MSD_wind = []
    # fit region of the MSD
    fit_wind = np.zeros((nSteps, nDrops, len(fit_range)))
    pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
    for i in tqdm(range(nSteps)):
        trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
        temp, fit_wind[i], pw_exp_wind[i], = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, fit_range)
        MSD_wind.append(temp)
    return MSD_wind, fit_wind, pw_exp_wind


def get_emsd_windowed(imsd, x, fps, red_mask, nSteps, maxLagtime, fit_range):
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
    
    results = {'fit_wind_b':fit_wind_b, 'pw_exp_wind_b':pw_exp_wind_b, 'fit_wind_r':fit_wind_r,\
                            'pw_exp_wind_r':pw_exp_wind_r}

    return EMSD_wind_b, EMSD_wind_r, results


# get trajectories
def get_trajs(nDrops, red_particle_idx, trajs, subsample_factor, fps):
    blueTrajs = []
    redTrajs = []
    for i in range(0, nDrops):
        if i in red_particle_idx:
            p = trajs.loc[trajs.particle == i, ['x','y']][::subsample_factor]
            redTrajs.append(Trajectory(p.x, p.y, dt = 1/fps*subsample_factor, traj_id=i, diff_est={'method':DiffMethod.LINEAR_DIFF, 
                                                                                  'window_type': WindowType.CENTRAL}))
        else:
            p = trajs.loc[trajs.particle == i, ['x','y']][::subsample_factor]
            blueTrajs.append(Trajectory(p.x, p.y, dt = 1/fps*subsample_factor, traj_id=i))
    return blueTrajs, redTrajs


# get speed distributions windowed in time
def speed_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajs, subsample_factor, fps, progress_verb):
    v_blue_wind = []
    v_red_wind = []
    if progress_verb: 
        for k in tqdm(range(nSteps)):
            trajs_wind = trajs.loc[trajs.frame.between(startFrames[k], endFrames[k])]
            blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajs_wind, subsample_factor, fps)
            v_blue_wind.append(ys.speed_ensemble(blueTrajs, step=1))
            v_red_wind.append(ys.speed_ensemble(redTrajs, step=1))
    else:
        for k in range(nSteps):
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

def vacf_windowed(trajectories, nSteps, startFrames, endFrames, red_particle_idx, subsample_factor, fps, maxLagtime):        
    vacf_b_wind = []
    vacf_b_std_wind = []
    vacf_r_wind = []
    vacf_r_std_wind = []
    
    for k in tqdm(range(nSteps)):
        trajs_wind = trajectories.loc[trajectories.frame.between(startFrames[k], endFrames[k])]
        blueTrajs, redTrajs = get_trajs(len(trajs_wind.particle.unique()), red_particle_idx, trajs_wind, subsample_factor, fps)
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
''' 
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
        COORDS = np.array(trajectories.loc[:, ['x','y']])
        parallel = joblib.Parallel(n_jobs = -2)
        rdf = parallel(
            rdf_frame(frame, COORDS, rList, dr, rho)
            for frame in tqdm(range(nFrames))
        )
        rdf = np.array(rdf)
        rdf_df = pd.DataFrame(rdf)
        # string columns for parquet filetype
        rdf_df.columns = [f'{r}' for r in rList]
        rdf_df.to_parquet(f'./{analysis_data_path}/rdf/rdf.parquet')

    elif not run_analysis_verb :
        try:
            rdf = np.array(pd.read_parquet(f'./{analysis_data_path}/rdf/rdf.parquet'))
        except: 
            raise ValueError('rdf data not found. Run analysis verbosely first.')
    else: 
        raise ValueError('run_analysis_verb must be True or False')
    return rdf
'''

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
    COORDS_blue = np.array(trajectories.loc[~trajectories.particle.isin(red_particle_idx), ['x','y']])
    COORDS_red = np.array(trajectories.loc[trajectories.particle.isin(red_particle_idx), ['x','y']])
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
    COORDS = np.array(trajectories.loc[:,['x','y']])
    parallel = joblib.Parallel(n_jobs = -2)
    rdf_c = parallel(
        rdf_center_frame(frame, COORDS, r_c, rList, dr, rho)
        for frame in tqdm( frames )
    )
    rdf_c = np.array(rdf_c)
    return rdf_c

def get_frame(cap, frame, x1, y1, x2, y2, w, h, preprocess):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()
    if preprocess:
        npImage = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        alpha = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)
        npAlpha = np.array(alpha)
        npImage = npImage*npAlpha
        ind = np.where(npImage == 0)
        npImage[ind] = npImage[200, 200]
        npImage = npImage[y1:y2, x1:x2]
        npImage = cv2.resize(npImage, (500, 500))
        return npImage
    elif not preprocess:
        npImage = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        npImage = npImage[y1:y2, x1:x2]
        npImage = cv2.resize(npImage, (500, 500))
        return npImage
    else:
        raise ValueError("preprocess must be a boolean")