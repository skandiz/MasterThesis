
def get_smooth_trajs(trajs, nDrops, windLen, orderofPoly):
    """
    Smooth the trajectories in the trajs DataFrame, in order to drop the noise due to the tracking procedure.
    The smoothing is performed using a Savgol Filter.
    """
    # Trajectory Smoothing: using a Savgol Filter 
    ret = trajs.copy()
    for i in range(nDrops):
        ret.loc[ret.particle == i, "x"] = savgol_filter(trajs.loc[trajs.particle == i].x.values, windLen, orderofPoly)
        ret.loc[ret.particle == i, "y"] = savgol_filter(trajs.loc[trajs.particle == i].y.values, windLen, orderofPoly)    
    return ret

def get_velocities(trajList):
    """
    Compute the velocities of the particles in the trajList.
    The velocities are computed as the difference between the positions of the particles in two consecutive frames.
    """
    v = np.zeros((len(trajList), 2, len(trajList[0])), dtype=np.float64)
    for i in range(len(trajList)):
        v[i] = np.array(trajList[i].v).T
    return v

# Power Law fit
def powerLaw(x, a, k):
    """
    Power law function used to fit the MSD.
    """
    return a*x**k

def powerLawFit(f, x, nDrops, yerr):
    """
    Fit the MSD with a power law function.
    The fit is performed using the scipy.optimize.curve_fit function.
    """
    if nDrops == 1:
        ret = np.zeros((2, 2))
        ret[0], pcov = curve_fit(powerLaw, x, f, p0 = [1., 1.])
        ret[1] = np.sqrt(np.diag(pcov))
        fit = ret[0, 0] * x**ret[0, 1]
        return fit, ret
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



def get_imsd(trajs, pxDimension, fps, maxLagtime, nDrops, region):
    """
    Compute the mean square displacement of the particles in the trajs DataFrame.
    The MSD is computed using the trackpy.imsd function.
    The MSD is then fitted with a power law function in order to extract the diffusion coefficient.
    The fit is performed using the powerLawFit function.
    """
    imsd = tp.imsd(trajs, mpp = pxDimension, fps = fps, max_lagtime = maxLagtime)
    if region == "all":
        fit, pw_exp = powerLawFit(imsd, imsd.index, nDrops, None)
        return imsd, fit, pw_exp
    elif region == "diffusive":
        fit, pw_exp = powerLawFit(imsd[1:], imsd[1:].index, nDrops, None)
        return imsd, fit, pw_exp
    elif region == "ballistic":
        fit, pw_exp = powerLawFit(imsd[:1], imsd[:1].index, nDrops, None)
        return imsd, fit, pw_exp
    else:
        raise ValueError("region flag must be 'all', 'diffusive' or 'ballistic'")
    

"""    
    # fit the diffusive region of the MSD
    fit_b, pw_exp_b = powerLawFit(MSD_b[0][9:], x, 1, MSD_b[1][9:])
    fit_r, pw_exp_r = powerLawFit(MSD_r[9:], x, 1, None)
    diffusive_results = {"fit_b":fit_b, "pw_exp_b":pw_exp_b, "fit_r":fit_r, "pw_exp_r":pw_exp_r}

    # fit the 'ballistic' region of the MSD
    fit_b, pw_exp_b = powerLawFit(MSD_b[0][:10], np.arange(0.1, 1.1, 0.1), 1, MSD_b[1][:10])
    fit_r, pw_exp_r = powerLawFit(MSD_r[:10], np.arange(0.1, 1.1, 0.1), 1, None)
    ballistic_results = {"fit_b":fit_b, "pw_exp_b":pw_exp_b, "fit_r":fit_r, "pw_exp_r":pw_exp_r}
    return MSD_b, MSD_r, diffusive_results, ballistic_results
"""
def get_emsd(imsd, red_particle_idx, nDrops, region):
    """
    Compute the ensemble mean square displacement of the particles in the trajs DataFrame.
    """
    MSD = np.array(imsd)
    MSD_b = [MSD[:, [i for i in range(nDrops) if i != red_particle_idx]].mean(axis = 1),
                MSD[:, [i for i in range(nDrops) if i != red_particle_idx]].std(axis = 1)]
    MSD_r = MSD[:, red_particle_idx]

    if region == "all":
        x = imsd.index.values
        fit_b, pw_exp_b = powerLawFit(MSD_b[0], x, 1, MSD_b[1])
        fit_r, pw_exp_r = powerLawFit(MSD_r, x, 1, None)
        results = {"fit_b":fit_b, "pw_exp_b":pw_exp_b, "fit_r":fit_r, "pw_exp_r":pw_exp_r}
        return MSD_b, MSD_r, results
    
    elif region == "diffusive":
        x = imsd[1:].index.values
        fit_b, pw_exp_b = powerLawFit(MSD_b[0][9:], x, 1, MSD_b[1][9:])
        fit_r, pw_exp_r = powerLawFit(MSD_r[9:], x, 1, None)
        results = {"fit_b":fit_b, "pw_exp_b":pw_exp_b, "fit_r":fit_r, "pw_exp_r":pw_exp_r}
        return MSD_b, MSD_r, results
    
    elif region == "ballistic":
        x = imsd[:1].index
        fit_b, pw_exp_b = powerLawFit(MSD_b[0][:9], x, 1, MSD_b[1][:9])
        fit_r, pw_exp_r = powerLawFit(MSD_r[:9], x, 1, None)
        results = {"fit_b":fit_b, "pw_exp_b":pw_exp_b, "fit_r":fit_r, "pw_exp_r":pw_exp_r}
        return MSD_b, MSD_r, results
    
    else:
        raise ValueError("region flag must be 'all', 'diffusive' or 'ballistic'")


def get_imsd_windowed(nSteps, startFrames, endFrames, trajs, pxDimension, fps, maxLagtime, nDrops, region):
    IMSD_wind = []
    # diffusive region of the MSD
    if region == "all":
        fit_wind = np.zeros((nSteps, nDrops, maxLagtime))
        pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
        for i in tqdm(range(nSteps)):
            trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
            temp, fit_wind[i], pw_exp_wind[i] = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, nDrops, region)
            IMSD_wind.append(temp)
        return IMSD_wind, fit_wind, pw_exp_wind

    elif region == "diffusive":
        fit_wind = np.zeros((nSteps, nDrops, maxLagtime-9))
        pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
        for i in tqdm(range(nSteps)):
            trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
            temp, fit_wind[i], pw_exp_wind[i] = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, nDrops, region)
            IMSD_wind.append(temp)
        return IMSD_wind, fit_wind, pw_exp_wind

    elif region == "ballistic":
        fit_wind = np.zeros((nSteps, nDrops, 10))
        pw_exp_wind = np.zeros((nSteps, nDrops, 2, 2))
        for i in tqdm(range(nSteps)):
            trajs_wind = trajs.loc[trajs.frame.between(startFrames[i], endFrames[i])]
            temp, fit_wind[i], pw_exp_wind[i] = get_imsd(trajs_wind, pxDimension, fps, maxLagtime, nDrops, region)
            IMSD_wind.append(temp)
        return IMSD_wind, fit_wind, pw_exp_wind
    else:
        raise ValueError("region flag must be 'all', 'diffusive' or 'ballistic'")


def get_emsd_windowed(IMSD_wind, nDrops, red_particle_idx, nSteps, maxLagtime, region):
    EMSD_wind = np.array(IMSD_wind)
    EMSD_wind_b = [EMSD_wind[:, :, [x for x in range(nDrops) if x != red_particle_idx]].mean(axis = 2), 
                    EMSD_wind[:, :, [x for x in range(nDrops) if x != red_particle_idx]].std(axis = 2)]
    EMSD_wind_r = EMSD_wind[:, :, red_particle_idx]
    if region == "all":
        fit_wind_b = np.zeros((nSteps, maxLagtime))
        pw_exp_wind_b = np.zeros((nSteps, 2, 2))
        fit_wind_r = np.zeros((nSteps, maxLagtime))
        pw_exp_wind_r = np.zeros((nSteps, 2, 2))
        x = IMSD_wind[0].index.values

        for i in range(nSteps):
            fit_wind_b[i], pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i], x, 1, EMSD_wind_b[1][i])
            fit_wind_r[i], pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[i], x, 1, None)
        
        results = {"fit_wind_b":fit_wind_b, "pw_exp_wind_b":pw_exp_wind_b, "fit_wind_r":fit_wind_r,\
                            "pw_exp_wind_r":pw_exp_wind_r}
        return EMSD_wind_b, EMSD_wind_r, results
    elif region == "diffusive":
        fit_wind_b = np.zeros((nSteps, maxLagtime-10))
        pw_exp_wind_b = np.zeros((nSteps, 2, 2))
        fit_wind_r = np.zeros((nSteps, maxLagtime-10))
        pw_exp_wind_r = np.zeros((nSteps, 2, 2))
        x = IMSD_wind[0].index.values[10:]
        
        for i in range(nSteps):
            fit_wind_b[i], pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, 10:], x, 1, EMSD_wind_b[1][i, 10:])
            fit_wind_r[i], pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[i, 10:], x, 1, None)
        
        results = {"fit_wind_b":fit_wind_b, "pw_exp_wind_b":pw_exp_wind_b, "fit_wind_r":fit_wind_r,\
                            "pw_exp_wind_r":pw_exp_wind_r}
        return EMSD_wind_b, EMSD_wind_r, results
    elif region == "ballistic":
            # 'ballistic' region of the MSD
        fit_wind_b = np.zeros((nSteps, 10))
        pw_exp_wind_b = np.zeros((nSteps, 2, 2))
        fit_wind_r = np.zeros((nSteps, 10))
        pw_exp_wind_r = np.zeros((nSteps, 2, 2))
        x = x = IMSD_wind[0].index.values[:10]
        for i in range(nSteps):
            fit_wind_b[i], pw_exp_wind_b[i] = powerLawFit(EMSD_wind_b[0][i, :10], x, 1, EMSD_wind_b[1][i, :10])
            fit_wind_r[i], pw_exp_wind_r[i] = powerLawFit(EMSD_wind_r[i, :10], x, 1, None)
        results = {"fit_wind_b":fit_wind_b, "pw_exp_wind_b":pw_exp_wind_b, "fit_wind_r":fit_wind_r,\
                            "pw_exp_wind_r":pw_exp_wind_r}
        return EMSD_wind_b, EMSD_wind_r, results

    else:
        raise ValueError("region flag must be 'all', 'diffusive' or 'ballistic'")