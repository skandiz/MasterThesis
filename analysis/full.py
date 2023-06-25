import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation
import numpy as np
import pandas as pd

from scipy.spatial import KDTree, cKDTree
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import time
import joblib
from tqdm import tqdm
import trackpy as tp
from numba import njit, prange

from yupi import Trajectory
import yupi.graphics as yg
import yupi.stats as ys

show_verb = False
run_windowed_analysis = False
plot_verb = True
animated_plot_verb = True
save_verb = True
run_analysis_verb = False

msd_run = False
speed_run = False
turn_run = False
autocorr_run = False
rdf_run = True

#####################################################################################################################
#                                                 DEFINE FUNCTIONS                                                  #
#####################################################################################################################
if 1:
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


    def get_emsd_windowed(imsds, x, fps, red_particle_idx, nSteps, maxLagtime):
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

#############################################################################################################
#                                               IMPORT DATA
#############################################################################################################

if 1: 
    print("Import data...")
    if 0:
        data_path = "../tracking/49b_1r/49b_1r_pre_merge/df_linked.parquet"
        res_path = "./49b_1r/results"
        analysis_data_path = "./49b_1r/analysis_data"
        red_particle_idx = 8
        fps = 10
        v_step = 10
    else:
        data_path = "../tracking/25b_25r/df_linked.parquet"
        res_path = "./25b_25r/results"
        analysis_data_path = "./25b_25r/analysis_data"
        red_particle_idx = np.sort(np.array([27, 24, 8, 16, 21, 10, 49, 14, 12, 9, 7, 37, 36, 40, 45, 42, 13, 20, 26, 2, 39, 5, 11, 22, 44])).astype(int)
        fps = 30
        v_step = 30

    rawTrajs = pd.read_parquet(data_path)
    nDrops = int(len(rawTrajs.loc[rawTrajs.frame==0]))
    red_mask = np.zeros(nDrops, dtype=bool)
    red_mask[red_particle_idx] = True
    colors = np.array(['b' for i in range(nDrops)])
    colors[red_particle_idx] = 'r'

    frames = rawTrajs.frame.unique().astype(int)
    nFrames = len(frames)
    print(f"Number of Droplets: {nDrops}")
    print(f"Number of Frames: {nFrames} at {fps} fps --> {nFrames/fps:.2f} s")

    # ANALYSIS PARAMETERS
    pxDimension = 1 # has to be defined 
    maxLagtime = 3000 # maximum lagtime to be considered
    x = np.arange(1, maxLagtime/fps + 1/fps, 1/fps) # range of power law fit

    # WINDOWED ANALYSIS PARAMETERS
    window = 300*fps # 320 s
    stride = 10*fps # 10 s
    print("Windowed analysis args:")
    startFrames = np.arange(0, nFrames-window, stride, dtype=int)
    endFrames = startFrames + window
    nSteps = len(startFrames)
    print(f"window of {window/fps} s, stride of {stride/fps} s --> {nSteps} steps")
    units = "px/s"
    default_kwargs_blue = {"color": "#00FFFF", "ec": (0, 0, 0, 0.6), "density": True}
    default_kwargs_red = {"color": "#EE4B2B", "ec": (0, 0, 0, 0.6), "density": True}

#############################################################################################################
#                                              MSD ANALYSIS
#############################################################################################################
if msd_run:
    print("Global IMSD")
    imsd, fit, pw_exp = get_imsd(rawTrajs, pxDimension, fps, maxLagtime, nDrops)

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        for i in range(nDrops):
            ax.plot(imsd.index, imsd[i], color = colors[i])
        ax.set(xscale = 'log', yscale = 'log', xlabel = "Time Lag [s]", ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]')
        ax.grid()
        ax1.scatter(np.arange(nDrops), pw_exp[:, 0, 1], color = colors)
        ax1.set(xlabel = "Particle ID", ylabel = "Powerlaw Exponent")
        ax1.grid()
        plt.suptitle("Mean Squared Displacement - Raw Trajectories")
        plt.tight_layout()
        if save_verb: plt.savefig(f"./{res_path}/mean_squared_displacement/IMSD_raw.png", bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()
        
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(8, 5), tight_layout=True)
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(nDrops):
            ax1.plot(imsd.index, imsd.values[:, i], color = colors[i], linewidth = 0.5)
        ax1.set(xscale="log", yscale = "log", xlabel = "lag time [s]", ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', title = "IMSD")
        ax1.grid()

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(np.arange(nDrops), pw_exp[:, 0, 1], s = 10,  color = colors)
        ax2.set(xlabel = "Droplet ID", ylabel = r"$\alpha$", title = "power law exponents")
        ax2.grid()

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(np.arange(nDrops), pw_exp[:, 0, 0], s = 10, color = colors)
        ax3.set(xlabel="Droplet ID", ylabel = "K", title = "Diffusion coefficients")
        ax3.grid()
        if save_verb: plt.savefig(f"./{res_path}/mean_squared_displacement/IMSD_raw_v2.png", bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
            
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(pw_exp[:, 0, 0], pw_exp[:, 0, 1], s = 10,  color = colors)
        ax.set(xlabel = "Diffusion Coefficient", ylabel = r"$\alpha$")
        ax.grid(linewidth = 0.2)
        if save_verb: plt.savefig(f"./{res_path}/mean_squared_displacement/k_alpha_scatterplot.png", bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()
        


    print("Global EMSD")
    MSD_b, MSD_r, fit = get_emsd(imsd, x, fps, red_particle_idx, nDrops)

    # Trajs: temp variable to print pw_exp results
    alpha_b = [round(fit["pw_exp_b"][0, 1], 3), round(fit["pw_exp_b"][1, 1], 3)]
    k_b = [round(fit["pw_exp_b"][0, 0], 3), round(fit["pw_exp_b"][1, 0], 3)]
    alpha_r = [round(fit["pw_exp_r"][0, 1], 3), round(fit["pw_exp_r"][1, 1], 3)]
    k_r = [round(fit["pw_exp_r"][0, 0], 3), round(fit["pw_exp_r"][1, 0], 3)]

    print(f"Blue Particles: a = {alpha_b[0]} ± {alpha_b[1]}, K = {k_b[0]} ± {k_b[1]}")
    print(f"Red Particle: a = {alpha_r[0]} ± {alpha_r[1]}, K = {k_r[0]} ± {k_r[1]}")

    if plot_verb:
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        ax.plot(imsd.index, MSD_b[0], 'b-', label = "Blue particles") 
        ax.plot(imsd[1:].index, fit["fit_b"], 'b--')
        ax.fill_between(imsd.index, MSD_b[0] - MSD_b[1], MSD_b[0] + MSD_b[1], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        ax.plot(imsd.index, MSD_r[0], 'r-', label = "Red particle")
        ax.plot(imsd[1:].index, fit["fit_r"], 'r--')
        ax.fill_between(imsd.index, MSD_r[0] - MSD_r[1], MSD_r[0] + MSD_r[1], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]',   
                xlabel = 'lag time $t$ [s]', title = "EMSD - Raw Trajectories")
        ax.legend()
        ax.grid()
        if save_verb: plt.savefig(f"./{res_path}/mean_squared_displacement/EMSD_raw.png", bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
             


    print("Windowed IMSD")
    if run_windowed_analysis: MSD_wind, fit_wind, pw_exp_wind = get_imsd_windowed(nSteps, startFrames, endFrames, rawTrajs, pxDimension, fps, maxLagtime, nDrops)

    if run_windowed_analysis and plot_verb:
        if animated_plot_verb:
            
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
            anim_running = True
            def onClick(event):
                global anim_running
                if anim_running:
                    ani.event_source.stop()
                    anim_running = False
                else:
                    ani.event_source.start()
                    anim_running = True
            def update_graph(step):
                for i in range(nDrops):
                    graphic_data[i].set_ydata(np.array(MSD_wind[step].iloc[:, i]))
                    graphic_data2[i].set_data(startFrames[:step]/fps, pw_exp_wind[:step, i, 0, 1])
                title.set_text(f"Mean Squared Displacement - Trajectories - window [{startFrames[step]/fps} - {endFrames[step]/fps}] s")
                ax1.set_xlim(0, startFrames[step]/fps + 0.0001)
                return graphic_data, graphic_data2,
            title = ax.set_title(f"Mean Squared Displacement - Trajectories - window [{startFrames[0]/fps} - {endFrames[0]/fps}] s")
            graphic_data = []
            for i in range(nDrops):
                if i in red_particle_idx:
                    graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]), color=colors[i], alpha = 0.3)[0])
                else:
                    graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]), color=colors[i], alpha = 0.3)[0])
            ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim = (0.01, 10**5))
            ax.grid()
            graphic_data2 = []
            for i in range(nDrops):
                if i in red_particle_idx:
                    graphic_data2.append(ax1.plot(startFrames[0]/fps, pw_exp_wind[0, i, 0, 1], color=colors[i], alpha = 0.3)[0])
                else:
                    graphic_data2.append(ax1.plot(startFrames[0]/fps, pw_exp_wind[0, i, 0, 1], color=colors[i], alpha = 0.3)[0])
            ax1.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2))
            ax1.grid()
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/mean_squared_displacement/windowed_analysis/IMSD_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()



    print("Windowed EMSD")
    if run_windowed_analysis: EMSD_wind_b, EMSD_wind_r, fit_dict = get_emsd_windowed(MSD_wind, x, fps, red_particle_idx, nSteps, maxLagtime)
    
    if run_windowed_analysis and plot_verb:
        # Power law exponents plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(f"Power Law Exponents - Trajectories")
        ax.plot(startFrames/fps, fit_dict["pw_exp_wind_b"][:, 0, 1], 'b-', alpha = 0.5, label = 'blue particles')
        ax.fill_between(startFrames/fps, fit_dict["pw_exp_wind_b"][:, 0, 1] - fit_dict["pw_exp_wind_b"][:, 1, 1],     
                            fit_dict["pw_exp_wind_b"][:, 0, 1] + fit_dict["pw_exp_wind_b"][:, 1, 1],
                            alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
        ax.plot(startFrames/fps, fit_dict["pw_exp_wind_r"][:, 0, 1], 'r-', alpha = 0.5, label = 'red particle ')
        ax.fill_between(startFrames/fps, fit_dict["pw_exp_wind_r"][:, 0, 1] - fit_dict["pw_exp_wind_r"][:, 1, 1],
                            fit_dict["pw_exp_wind_r"][:, 0, 1] + fit_dict["pw_exp_wind_r"][:, 1, 1],
                            alpha=0.5, edgecolor='#F0FFFF', facecolor='#FF5A52')
        ax.plot(startFrames/fps, np.ones(nSteps), 'k-')
        ax.legend()
        ax.grid()
        ax.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2))
        if save_verb: plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_alpha.png', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
        

        # Generalized Diffusion Coefficients plot
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        plt.suptitle(f"Generalized Diffusion Coefficients - Trajectories")
        ax.plot(startFrames/fps, fit_dict["pw_exp_wind_b"][:, 0, 0], 'b-', alpha = 0.5, label = 'blue particles')
        ax.set(xlabel = 'Window time [s]', ylabel = 'K')
        ax.legend()
        ax.grid()
        ax1.plot(startFrames/fps, fit_dict["pw_exp_wind_r"][:, 0, 0], 'r-', alpha = 0.5, label = 'red particle ')
        ax1.legend()
        ax1.grid()
        ax1.set(xlabel = 'Window time [s]')
        plt.tight_layout()
        if save_verb: plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_D.png', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
        
        if animated_plot_verb:
            # Lower and Higher bounds for fill between 
            Y1_msd_b = EMSD_wind_b[0] - EMSD_wind_b[1]
            Y2_msd_b = EMSD_wind_b[0] + EMSD_wind_b[1]
            Y1_msd_r = EMSD_wind_r[0] - EMSD_wind_r[1]
            Y2_msd_r = EMSD_wind_r[0] + EMSD_wind_r[1]

            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
            anim_running = True
            def onClick(event):
                global anim_running
                if anim_running:
                    ani.event_source.stop()
                    anim_running = False
                else:
                    ani.event_source.start()
                    anim_running = True
            def update_graph(step):
                # update title
                title.set_text(f"Mean Squared Displacement - Trajectories - window {startFrames[step]/fps} - {endFrames[step]/fps} seconds")
                # update MSD
                graphic_data[0].set_ydata(EMSD_wind_b[0][step])
                graphic_data[1].set_ydata(EMSD_wind_r[0][step])
                # update fill between
                path = fill_graph.get_paths()[0]
                verts = path.vertices
                verts[1:maxLagtime+1, 1] = Y1_msd_b[step, :]
                verts[maxLagtime+2:-1, 1] = Y2_msd_b[step, :][::-1]

                # update fill between
                path = fill_graph2.get_paths()[0]
                verts = path.vertices
                verts[1:maxLagtime+1, 1] = Y1_msd_r[step, :]
                verts[maxLagtime+2:-1, 1] = Y2_msd_r[step, :][::-1]

                # update powerlaw exponents
                line.set_data(startFrames[:step]/fps, fit_dict["pw_exp_wind_b"][:step, 0, 1])
                line1.set_data(startFrames[:step]/fps, fit_dict["pw_exp_wind_r"][:step, 0, 1]) 
                line2.set_data(startFrames[:step]/fps, np.ones(step)) 
                ax1.set_xlim(0, (startFrames[step]+fps)/fps)
                return graphic_data, fill_graph, line, line1, 

            title = ax.set_title(f"Mean Squared Displacement - Trajectories - window {startFrames[0]/fps} - {endFrames[0]/fps} seconds")
            graphic_data = []
            graphic_data.append(ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), EMSD_wind_b[0][0], 'b-', alpha=0.5, label = "Blue particles")[0])
            graphic_data.append(ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), EMSD_wind_r[0][0], 'r-' , label = "Red particle")[0] )
            fill_graph = ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), Y1_msd_b[0], Y2_msd_b[0], alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
            fill_graph2 = ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), Y1_msd_r[0], Y2_msd_r[0], alpha=0.5, edgecolor='#FF5A52', facecolor='#FF5A52')

            ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim=(0.01, 10**5))
            ax.legend()
            ax.grid()
            line, = ax1.plot(startFrames[0]/fps, fit_dict["pw_exp_wind_b"][0, 0, 1], 'b-', alpha = 0.5, label = 'Blue particles')
            line1, = ax1.plot(startFrames[0]/fps, fit_dict["pw_exp_wind_r"][0, 0, 1], 'r-', alpha = 0.5, label = 'Red particle')
            line2, = ax1.plot(startFrames[0]/fps, 1, 'k-')
            ax1.legend()
            ax1.grid(linewidth=0.2)
            ax1.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2))
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()

#############################################################################################################
#                                              SPEED DISTRIBUTION ANALYSIS
#############################################################################################################
if speed_run:

    print(f"Speed Analysis: show_verb = {show_verb}, animated_plot_verb = {animated_plot_verb}")

    print("\n Global speed distribution analysis")
    blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, rawTrajs)
    bin_borders = np.arange(0, 100, .2)
    bin_centers = np.arange(0, 100, .2)[:-1] + .2 / 2
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

    v_blue = ys.speed_ensemble(blueTrajs, step = v_step)
    v_red = ys.speed_ensemble(redTrajs, step = v_step)
    T_blue, T_blue_std = fit_hist(v_blue, bin_borders, MB_2D, [1.])
    T_red, T_red_std = fit_hist(v_red, bin_borders, MB_2D, [1.])
    print("Trajectories")
    print(f"Blue Particles σ: {T_blue[0]} ± {T_blue_std[0]}")
    print(f"Red Particle σ: {T_red[0]} ± {T_red_std[0]}")

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 5))
        ax.hist(v_blue, bins = bin_borders, **default_kwargs_blue)
        ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, T_blue), 'k-', label = f"$T = {T_blue[0]:.2f} \pm {T_blue_std[0]:.2f}$")
        ax.set(title = "Blue particles velocity distribution", xlabel = f"speed [{units}]", ylabel = "pdf", xlim = (0, 30), ylim = (0, 0.45))
        ax.legend()

        ax1.hist(v_red, bins = bin_borders, **default_kwargs_red)
        ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, T_red), 'k-', label = f"$T = {T_red[0]:.2f} \pm {T_red_std[0]:.2f}$")
        ax1.set(title = "Red particle velocity distribution", xlabel = f"speed [{units}]", ylabel = "pdf", xlim = (0, 30), ylim = (0, 0.45))
        ax1.legend()

        plt.suptitle("trajectories")
        plt.tight_layout()
        if save_verb: plt.savefig(f"./{res_path}/speed_distribution/speed.png", )
        if show_verb: 
            plt.show()
        else:
            plt.close()



    print("\n Windowed speed distribution Analysis")
    v_blue_wind, v_red_wind = speed_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, rawTrajs, v_step, fps)
    blue_fit_wind = np.ones((nSteps, 2))
    red_fit_wind = np.ones((nSteps, 2))

    for k in range(nSteps):
        blue_fit_wind[k, 0], blue_fit_wind[k, 1] = fit_hist(v_blue_wind[k], bin_borders, MB_2D, [1.])
        red_fit_wind[k, 0], red_fit_wind[k, 1] = fit_hist(v_red_wind[k], bin_borders, MB_2D, [1.])

    if plot_verb:   

        # Effetcive Temperature plot
        fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 4), sharex=True)
        ax.errorbar(startFrames/fps, blue_fit_wind[:, 0], yerr = blue_fit_wind[:, 1], fmt = 'b', label="blue particles")
        ax.set(ylabel = "T [??]", ylim = (1, 5), title = "Effective Temperature - Trajectories")
        ax.legend()
        ax.grid()
        ax1.errorbar(startFrames/fps, red_fit_wind[:, 0], yerr = red_fit_wind[:, 1], fmt = 'r', label="red particles")
        ax1.set(xlabel = "Window Time [s]", ylabel = "T [??]")
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        if save_verb: plt.savefig(f'./{res_path}/speed_distribution/T_eff.png', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        if animated_plot_verb:
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
            anim_running = True

            def onClick(event):
                global anim_running
                if anim_running:
                    ani.event_source.stop()
                    anim_running = False
                else:
                    ani.event_source.start()
                    anim_running = True


            def prepare_animation(bar_container, bar_container2):
                def animate(frame):
                    # update titles
                    title.set_text(f"Trajectories - window {startFrames[frame]/fps} - {endFrames[frame]/fps} seconds")
                    #title2.set_text(f"Red particle velocity pdf {startFrames[frame]/fps} - {endFrames[frame]/fps} seconds")

                    # update histogram 1
                    n, _ = np.histogram(v_blue_wind[frame], bin_borders, density = True)
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    
                    line.set_ydata(MB_2D(x_interval_for_fit, blue_fit_wind[frame, 0]))

                    # update histogram 2
                    n2, _ = np.histogram(v_red_wind[frame], bin_borders, density = True)
                    for count2, rect2 in zip(n2, bar_container2.patches):
                        rect2.set_height(count2)
                        
                    line1.set_ydata(MB_2D(x_interval_for_fit, red_fit_wind[frame, 0]))

                    return bar_container.patches, bar_container2.patches
                return animate

            _, _, bar_container = ax.hist(v_blue_wind[0], bin_borders, **default_kwargs_blue, label="blue particles")
            title = ax.set_title(f"Trajectories - window {startFrames[0]/fps} - {endFrames[0]/fps} seconds")
            line, = ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, blue_fit_wind[0, 0]), label='fit')
            ax.set(xlabel = f"speed [{units}]", ylabel = "pdf", xlim = (0, 30), ylim = (0, 0.5))
            ax.legend()

            _, _, bar_container2 = ax1.hist(v_red_wind[0], bin_borders,  **default_kwargs_red, label="red particle")
            #title2 = ax1.set_title(f"Red particle velocity pdf {startFrames[0]/fps} - {endFrames[0]/fps} seconds")
            line1, = ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, red_fit_wind[0, 0]), label='fit')
            ax1.set(xlabel = f"speed [{units}]", ylabel = "pdf", xlim = (0, 30), ylim = (0, 0.5))
            ax1.legend()

            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, prepare_animation(bar_container, bar_container2), nSteps, repeat=True, blit=False)
            if save_verb: ani.save(f'./{res_path}/speed_distribution/speed_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb: 
                plt.show()
            else:
                plt.close()


#############################################################################################################
#                                              TRURNING ANGLES DISTRIBUTION ANALYSIS
#############################################################################################################
if turn_run:
    # choose the distribution to fit
    distribution_str = "lorentzian"

    if distribution_str == "gaussian":
        distribution = normal_distr
    elif distribution_str == "lorentzian":
        distribution = lorentzian_distr
        distribution_str = "lorentzian"
    else:
        raise ValueError("distribution_str must be either 'gaussian' or 'lorentzian'")

    print(f"Turning Angles Analysis: show_verb = {show_verb}, animated_plot_verb = {animated_plot_verb}")


    print("\n Global turning angles analysis")
    blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, rawTrajs)


    bin_borders_turn = np.arange(-np.pi, np.pi + 0.0001, np.pi/50)
    bin_centers_turn = bin_borders_turn[:-1] + np.diff(bin_borders_turn) / 2
    x_interval_for_fit_turn = np.linspace(bin_borders_turn[0], bin_borders_turn[-1], 10000)

    theta_blue = ys.turning_angles_ensemble(blueTrajs, centered = True)
    theta_red = ys.turning_angles_ensemble(redTrajs, centered = True)
    # normal distribution fit
    T_blue_rot, T_blue_rot_std = fit_hist(theta_blue, bin_borders_turn, distribution, [1., 0.])
    T_red_rot, T_red_rot_std = fit_hist(theta_red, bin_borders_turn, distribution, [1., 0.])
    print("Trajectories")
    print(f"Blue Particles σ: {T_blue_rot[0]} ± {T_blue_rot_std[0]}, μ: {T_blue_rot[1]} ± {T_blue_rot_std[1]}")
    print(f"Red Particle σ: {T_red_rot[0]} ± {T_red_rot_std[0]}, μ: {T_red_rot[1]} ± {T_red_rot_std[1]}")

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        ax.hist(theta_blue, bin_borders_turn, **default_kwargs_blue, label="blue particles")
        ax.plot(x_interval_for_fit_turn, distribution(x_interval_for_fit_turn, *T_blue_rot), 'k-',
                        label = f"$T = {T_blue_rot[0]:.2f} \pm {T_blue_rot_std[0]:.2f}$")
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.legend()
        ax.set_ylim(0, 2)

        ax1.hist(theta_red, bin_borders_turn, **default_kwargs_red, label="red particle")
        ax1.plot(x_interval_for_fit_turn, distribution(x_interval_for_fit_turn, *T_red_rot), 'k-',
                        label = f"$T = {T_red_rot[0]:.2f} \pm {T_red_rot_std[0]:.2f}$")
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax1.legend()
        ax1.set_ylim(0, 2)
        plt.suptitle("Turning angles pdf - trajectories")
        if save_verb: plt.savefig(f"./{res_path}/turning_angles/{distribution_str}/turn_ang.png")
        if show_verb:
            plt.show()
        else:
            plt.close()

    print("\n Windowed turning angles analysis")
    theta_blue_wind, theta_red_wind = theta_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, rawTrajs, fps)

    blue_fit_wind_turn = np.ones((nSteps, 2, 2))
    red_fit_wind_turn = np.ones((nSteps, 2, 2))
    for k in range(nSteps):
        blue_fit_wind_turn[k, 0], blue_fit_wind_turn[k, 1] = fit_hist(theta_blue_wind[k], bin_borders_turn, distribution, [1., 0.])
        red_fit_wind_turn[k, 0], red_fit_wind_turn[k, 1] = fit_hist(theta_red_wind[k], bin_borders_turn, distribution, [1., 0.])


    if plot_verb: 

        ##############################################################################################################
        #                                              Fit results plot                                              #
        ##############################################################################################################

        
        fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        ax.plot(startFrames/fps, blue_fit_wind_turn[:, 0, 0], 'b', label="blue particles")
        ax.fill_between(startFrames/fps, blue_fit_wind_turn[:, 0, 0] - blue_fit_wind_turn[:, 1, 0],
                        blue_fit_wind_turn[:, 0, 0] + blue_fit_wind_turn[:, 1, 0], color='b', alpha=0.2)
        ax.set(ylabel = "T [??]", title = "Effective Temperature - Trajectories")
        ax.legend()
        ax.grid()
        ax1.plot(startFrames/fps, red_fit_wind_turn[:, 0, 0], 'r', label="red particles")
        ax1.fill_between(startFrames/fps, red_fit_wind_turn[:, 0, 0] - red_fit_wind_turn[:, 1, 0],
                            red_fit_wind_turn[:, 0, 0] + red_fit_wind_turn[:, 1, 0], color='r', alpha=0.2)
        ax1.set(xlabel = "Window Time [s]", ylabel = "T [??]")
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        if save_verb: plt.savefig(f'./{res_path}/turning_angles/{distribution_str}/effective_T.png',  )
        if show_verb:
            plt.show()
        else:
            plt.close()

        ##############################################################################################################
        #                                              Animated Plots                                                #
        ##############################################################################################################
        if animated_plot_verb:
            
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
            anim_running = True
            def onClick(event):
                global anim_running
                if anim_running:
                    ani.event_source.stop()
                    anim_running = False
                else:
                    ani.event_source.start()
                    anim_running = True
            def prepare_animation(bar_container, bar_container2):
                def animate(frame):
                    title.set_text(f"Turning angles pdf - Trajectories - window {startFrames[frame]/fps} - {endFrames[frame]/fps} seconds")
                    n, _ = np.histogram(theta_blue_wind[frame], bin_borders_turn, density = True)
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    line.set_ydata(distribution(x_interval_for_fit_turn, *blue_fit_wind_turn[frame, 0]))
                    n2, _ = np.histogram(theta_red_wind[frame], bin_borders_turn, density = True)
                    for count2, rect2 in zip(n2, bar_container2.patches):
                        rect2.set_height(count2)
                    line1.set_ydata(distribution(x_interval_for_fit_turn, *red_fit_wind_turn[frame, 0]))
                    return bar_container.patches, bar_container2.patches
                return animate
            _, _, bar_container = ax.hist(theta_red_wind[0], bin_borders_turn, **default_kwargs_blue, label="blue particles")
            line, = ax.plot(x_interval_for_fit_turn, distribution(x_interval_for_fit_turn, *blue_fit_wind_turn[0, 0]), label='fit')
            title = ax.set_title(f"Turning angles pdf - Trajectories - window {startFrames[0]/fps} - {endFrames[0]/fps} seconds")
            ax.set(ylabel = "pdf", ylim = (0, 3))
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            _, _, bar_container2 = ax1.hist(theta_red_wind[0], bin_borders_turn,  **default_kwargs_red, label="red particle")
            line1, = ax1.plot(x_interval_for_fit_turn, distribution(x_interval_for_fit_turn, *red_fit_wind_turn[0, 0]), label='fit')
            ax1.set(ylabel = "pdf", ylim = (0, 3))
            ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, prepare_animation(bar_container, bar_container2), nSteps, repeat=True, blit=False)
            ani.save(f'./{res_path}/turning_angles/{distribution_str}/turn_ang_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()
            

#############################################################################################################
#                                           VELOCITY AUTOCORRELATION ANALYSIS
#############################################################################################################
if autocorr_run:
    print(f"Velocity Autocorrelation Analysis: show_verb = {show_verb}, run_windowed_analysis = {run_windowed_analysis}, animated_plot_verb = {animated_plot_verb}")

    print("Global Velocity Autocovariance Function")
    
    blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, rawTrajs)
    vacf_b, vacf_std_b = ys.vacf(blueTrajs, time_avg=True, lag = maxLagtime)
    vacf_r, vacf_std_r = ys.vacf(redTrajs, time_avg=True, lag = maxLagtime)
    print(vacf_b.shape)

    if plot_verb:
        #Global Velocity Autocovariance
        fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 5))
        ax.errorbar(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), vacf_b, fmt='o', markersize = 1, color = "blue", label = 'blue particles')
        ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), vacf_b + vacf_std_b, vacf_b - vacf_std_b, alpha=1, edgecolor='#F0FFFF', facecolor='#00FFFF')
        ax.grid()
        ax.legend()
        ax.set(xlim = (-1, 10), xlabel = 'Lag time [s]', ylabel = r'VACF [$(px/s)^2$]')
        ax1.errorbar(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), vacf_r, fmt='o', markersize = 1, color = "red", label = 'red particles')
        ax1.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), vacf_r + vacf_std_r, vacf_r - vacf_std_r, alpha=1, edgecolor='#FF0000', facecolor='#FF5A52')
        ax1.set(xlim = (-1, 10), xlabel = 'Lag time [s]', ylabel = r'VACF [$(px/s)^2$]')
        ax1.grid()
        ax1.legend()
        plt.suptitle("Velocity autocorrelation function - trajectories")
        plt.tight_layout()
        if save_verb: plt.savefig(f"./{res_path}/velocity_autocovariance/vacf.png", )
        if show_verb: 
            plt.show()
        else:
            plt.close()
    
    print("Windowed analysis")
    if run_windowed_analysis:
        vacf_b_wind, vacf_b_std_wind, vacf_r_wind, vacf_r_std_wind = vacf_vindowed(rawTrajs, True)
    else:
        vacf_b_wind     = pd.read_parquet(f"./{analysis_data_path}/velocity_autocovariance/vacf_b_wind.parquet")
        vacf_b_std_wind = pd.read_parquet(f"./{analysis_data_path}/velocity_autocovariance/vacf_b_std_wind.parquet")
        vacf_r_wind     = pd.read_parquet(f"./{analysis_data_path}/velocity_autocovariance/vacf_r_wind.parquet")
        vacf_r_std_wind = pd.read_parquet(f"./{analysis_data_path}/velocity_autocovariance/vacf_r_std_wind.parquet")
    
    if plot_verb:
        # Velocity Variance
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(startFrames/fps, vacf_b_wind["0"], 'b', label = "Blue particles")
        ax.plot(startFrames/fps, vacf_r_wind["0"], 'r', label = "Red particle")
        ax.set(title = "Time evolution of velocity variance - Trajectories", ylabel = "$\sigma$", xlabel = "Window Time [s]")
        ax.grid()
        ax.legend()
        if save_verb: plt.savefig(f"./{res_path}/velocity_autocovariance/vacf_wind_0.png", )
        if show_verb:
            plt.show()
        else: 
            plt.close()

        # Animated Plots
        if animated_plot_verb:
            fig = plt.figure(figsize=(8, 5))
            anim_running = True
            def onClick(event):
                global anim_running
                if anim_running:
                    ani.event_source.stop()
                    anim_running = False
                else:
                    ani.event_source.start()
                    anim_running = True
            def update_graph(step):
                line.set_ydata(vacf_b_wind.iloc[step]/vacf_b_wind.iloc[step]["0"])
                title.set_text(f"Velocity autocorrelation - Trajectories - window [{startFrames[step]/fps} - {endFrames[step]/fps}] s")
                line1.set_ydata(vacf_r_wind.iloc[step]/vacf_r_wind.iloc[step]["0"])
                return line, line1,
            ax = fig.add_subplot(211)
            title = ax.set_title(f"Velocity autocorrelation - Trajectories - window [{startFrames[0]/fps} - {endFrames[0]/fps}] s")
            line, = ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), vacf_b_wind.iloc[0]/vacf_b_wind.iloc[0]["0"], 'b-', label = 'Blue particles')
            ax.set(ylabel = r'vacf [$(px/s)^2$]', xlabel = 'lag time $t$ [s]', xlim = (-1, 20), ylim = (-0.5, 1.1))
            ax.grid()
            ax.legend()
            ax1 = fig.add_subplot(212)
            line1, = ax1.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), vacf_r_wind.iloc[0]/vacf_r_wind.iloc[0]["0"], 'r-', label = 'Red particle')
            ax1.set(ylabel = r'vacf [$(px/s)^2$]', xlabel = 'lag time $t$ [s]', xlim = (-1, 20), ylim = (-0.5, 1.1))
            ax1.grid()
            ax1.legend()
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/velocity_autocovariance/vacf_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()

#############################################################################################################
#                                      RADIAL DISTRIBUTION FUNCTION ANALYSIS
#############################################################################################################
if rdf_run:
    print(f"Radial Distribution Function Analysis: run_analysis_verb = {run_analysis_verb}, show_verb = {show_verb}, animated_plot_verb = {animated_plot_verb}")

    dr = 5
    rDisk = 822/2
    rList = np.arange(0, 2*rDisk, 1)
    rho = nDrops/(np.pi*rDisk**2) # nDrops - 1 !???

    print("RDF - Trajectories")
    rdf = get_rdf(run_analysis_verb, nFrames, rawTrajs, rList, dr, rho)

    if animated_plot_verb:
        # Animated plot for trajs results
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        anim_running = True
        def onClick(event):
            global anim_running
            if anim_running:
                ani.event_source.stop()
                anim_running = False
            else:
                ani.event_source.start()
                anim_running = True


        line, = ax.plot(rList, rdf[0])
        title = ax.set_title('RDF - Trajectories 0 s')
        def animate(frame):
            line.set_ydata(rdf[frame])  # update the data.
            title.set_text('RDF - Trajectories {} s'.format(frame/fps))
            return line, 

        ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, rdf.shape[0], 10), interval=5, blit=False)
        ax.set(ylim = (-0.5, 30), ylabel = "g(r)", xlabel = "r (px)", title = "Radial Distribution Function from center")
        fig.canvas.mpl_connect('button_press_event', onClick)
        if save_verb: ani.save(f'./{res_path}/radial_distribution_function/rdf.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
        if show_verb:
            plt.show()
        else:
            plt.close()

    g_plot = rdf.T
    timearr = np.linspace(0, rdf.shape[0], 10)/fps
    timearr = timearr.astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    img = ax.imshow(g_plot, vmin = 0, vmax = 8)
    ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
    ax.set(xticklabels = timearr, yticklabels = np.linspace(0, 2*rDisk, 10).astype(int))
    ax.set(xlabel = "Time [s]", ylabel = "r [px]", title = "rdf heatmap")
    fig.colorbar(img, ax=ax)
    ax.set_aspect(15)
    if save_verb: plt.savefig(f"./{res_path}/radial_distribution_function/rdf_heatmap.png", bbox_inches='tight' )
    if show_verb: 
        plt.show()
    else:
        plt.close()

    dr = 5
    rDisk = 822/2
    rList = np.arange(0, rDisk, 1)
    rho = nDrops/(np.pi*rDisk**2) # nDrops -1 !
    r_c = [470, 490] #center of the image --> to refine

    print("RDF from center - Trajectories")
    rdf_c = get_rdf_center(run_analysis_verb, nFrames, rawTrajs, r_c, rList, dr, rho)

    fig, ax = plt.subplots(1, 1, figsize = (10, 4))
    ax.plot(rList, rdf_c.mean(axis=0), label="mean")
    ax.fill_between(rList, rdf_c.mean(axis=0) - rdf_c.std(axis=0), \
                        rdf_c.mean(axis=0) + rdf_c.std(axis=0), alpha=0.3, label="std")
    ax.set(xlabel = "r [px]", ylabel = "g(r)", title = "RDF from center - Trajectories")
    ax.legend()
    if save_verb: plt.savefig(f"./{res_path}/radial_distribution_function/rdf_center.png", bbox_inches='tight' )
    if show_verb: 
        plt.show()
    else:
        plt.close()
   
    if animated_plot_verb:
        # Animated plot for Smooth Trajectories
        fig, ax = plt.subplots()
        anim_running = True

        def onClick(event):
            global anim_running
            if anim_running:
                ani.event_source.stop()
                anim_running = False
            else:
                ani.event_source.start()
                anim_running = True

        line, = ax.plot(rList, rdf_c[0])
        title = ax.set_title('RDF from center - Trajectories 0 s')
        ax.set_ylim(-1, 15)

        def animate(frame):
            line.set_ydata(rdf_c[frame])  # update the data.
            title.set_text('RDF from center - Trajectories {} s'.format(frame/fps))
            return line, 

        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, nFrames, fps), interval = 5, blit=False)
        if save_verb: ani.save(f'./{res_path}/radial_distribution_function/rdf_from_center.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
        if show_verb:
            plt.show()
        else:
            plt.close()


    timearr = np.linspace(0, rdf.shape[0], 10)/fps
    timearr = timearr.astype(int)
    g_plot = rdf_c.T
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    img = ax.imshow(g_plot, vmin = 0, vmax = 10)
    ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
    ax.set(xticklabels = timearr, yticklabels = np.linspace(0, rDisk, 10).astype(int))
    ax.set(xlabel = "Time [s]", ylabel = "r [px]", title = "rdf from center heatmap - Trajectories")
    fig.colorbar(img, ax=ax)
    ax.set_aspect(30)
    if save_verb: plt.savefig(f"./{res_path}/radial_distribution_function/rdf_center_heatmap.png", bbox_inches='tight')
    if show_verb: 
        plt.show()
    else:
        plt.close()