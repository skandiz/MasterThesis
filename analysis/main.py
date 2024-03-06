import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation
import matplotlib as mpl
font = {'size' : 12}
mpl.rc('font', **font)
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree, Voronoi, voronoi_plot_2d, ConvexHull
from scipy.optimize import curve_fit
from tqdm import tqdm
import yupi.stats as ys
import networkx as nx 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import graph_tool.all as gt
from analysis_utils import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import joblib
parallel = joblib.Parallel(n_jobs=-2)


show_verb = False
plot_verb = True
animated_plot_verb = True
save_verb = True
run_analysis_verb = True


ABP_verb = False
radius_verb = False
msd_verb = False
velocity_verb = False
turning_angles_verb = False
velocity_autocovariance_verb = False
rdf_verb = True
graph_verb = False
motif_verb = False

#############################################################################################################
#                                               IMPORT DATA
#############################################################################################################
if 1:
    video_selection = '49b1r_post_merge'
    if video_selection == '25b25r-1':
        nDrops             = 50
        xmin, ymin, xmax, ymax = 95, 30, 535, 470    
        pxDimension = 90/500 # 9cm is the petri dish --> 90mm
        red_particle_idx = np.sort(np.array([38, 42, 25, 4, 23, 13, 45, 33, 46, 29, 10, 3, 35, 18, 12, 0, 27, 19, 26, 47, 7, 48, 21, 20, 22], dtype=int))
        original_trajectories = pd.read_parquet('../tracking/25b25r-1/modified_2D_versatile_fluo/interpolated_tracking_25b25r-1_modified_2D_versatile_fluo_0_539999.parquet')
        original_trajectories.r = original_trajectories.r * pxDimension
        original_trajectories = original_trajectories.loc[:, ['x', 'y', 'r', 'frame', 'particle', 'color']]

    elif video_selection == '49b1r':
        nDrops             = 50
        xmin, ymin, xmax, ymax = 20, 50, 900, 930
        pxDimension = 90/500 
        original_trajectories = pd.read_parquet('../tracking/49b1r/modified_2D_versatile_fluo/interpolated_tracking_49b1r_modified_2D_versatile_fluo_pre_merge.parquet')
        original_trajectories.r = original_trajectories.r * pxDimension
        original_trajectories = original_trajectories.loc[:, ['x', 'y', 'r', 'frame', 'particle', 'color']]
        red_particle_idx = np.array([19]).astype(int)

    elif video_selection == '49b1r_post_merge':
        nDrops             = 49
        xmin, ymin, xmax, ymax = 20, 50, 900, 930
        pxDimension = 90/500 
        original_trajectories = pd.read_parquet('../tracking/49b1r/modified_2D_versatile_fluo/interpolated_tracking_49b1r_modified_2D_versatile_fluo_post_merge.parquet')
        original_trajectories.r = original_trajectories.r * pxDimension
        original_trajectories = original_trajectories.loc[:, ['x', 'y', 'r', 'frame', 'particle', 'color']]
        red_particle_idx = np.array([15]).astype(int)

    windowLenght = 600 # seconds
    path = trim_up_to_char(video_selection, '_')
    source_path        = f'../tracking/data/{path}.mp4'
    res_path           = f'./{video_selection}/results_{windowLenght}'
    pdf_res_path       = f'../../thesis_project/images/{video_selection}'
    analysis_data_path = f'./{video_selection}/analysis_data_{windowLenght}'
    system_name        = f'{video_selection} system'


    video = cv2.VideoCapture(source_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    subsample_factor = 1#int(fps/10)
    n_frames_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video has {n_frames_video} frames with a resolution of {w}x{h} and a framerate of {fps} fps')

    frames = original_trajectories.frame.unique().astype(int)
    nFrames = len(frames)
    print(f'Number of Droplets: {nDrops}')
    print(f'Number of Frames: {nFrames} at {fps} fps --> {nFrames/fps:.2f} s')


    red_mask = np.zeros(nDrops, dtype=bool)
    red_mask[red_particle_idx] = True
    colors = np.array(['b' for i in range(nDrops)])
    colors[red_particle_idx] = 'r'

    # ANALYSIS PARAMETERS
    maxLagtime = 100*fps # maximum lagtime to be considered in the analysis, 100 seconds
    x_diffusive = np.linspace(10, maxLagtime/fps, int((maxLagtime/fps + 1/fps - 10)*fps)) 
    x_ballistic = np.linspace(1/fps, 1, int((1-1/fps)*fps)+1)

    # WINDOWED ANALYSIS PARAMETERS
    window = windowLenght*fps # windowLenght seconds
    stride = 10*fps # 10 s
    print('Windowed analysis args:')
    startFrames = np.arange(frames[0], frames[-1]-window, stride, dtype=int)
    endFrames = startFrames + window
    nSteps = len(startFrames)
    print(f'window of {window/fps} s, stride of {stride/fps} s --> {nSteps} steps')

    speed_units = 'mm/s'
    dimension_units = 'mm'
    default_kwargs_blue  = {'color': '#00FFFF', 'ec': (0, 0, 0, 0.6), 'density': True}
    default_kwargs_blue2 = {'color': '#F0FFFF', 'ec': (0, 0, 0, 0.6), 'density': True}
    default_kwargs_blue3 = {'color': '#0000FF', 'ec': (0, 0, 0, 0.6), 'density': True}
    default_kwargs_red   = {'color': '#EE4B2B', 'ec': (0, 0, 0, 0.6), 'density': True}
    default_kwargs_red2  = {'color': '#880808', 'ec': (0, 0, 0, 0.6), 'density': True}
    default_kwargs_red3  = {'color': '#D2042D', 'ec': (0, 0, 0, 0.6), 'density': True}

    print('Smoothing trajectories..')
    trajectories = get_smooth_trajs(original_trajectories, nDrops, int(fps/2), 4)

    if 1:
        df = trajectories.loc[trajectories.frame == 0]
        df1 = trajectories.loc[trajectories.frame == frames[-1]]
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 5))
        ax.imshow(get_frame(video, 0, xmin, ymin, xmax, ymax, w, h, False))
        for i in range(len(df)):
            if i in red_particle_idx:
                ax.add_artist(plt.Circle((df.x.iloc[i], df.y.iloc[i]), df.r.iloc[i]/pxDimension, color = 'red', fill = False))
            else:
                ax.add_artist(plt.Circle((df.x.iloc[i], df.y.iloc[i]), df.r.iloc[i]/pxDimension, color = 'blue', fill = False))
        ax1.imshow(get_frame(video, frames[-1], xmin, ymin, xmax, ymax, w, h, False))
        for i in range(len(df)):
            if i in red_particle_idx:
                ax1.add_artist(plt.Circle((df1.x.iloc[i], df1.y.iloc[i]), df1.r.iloc[i]/pxDimension, color = 'red', fill = False))
            else:
                ax1.add_artist(plt.Circle((df1.x.iloc[i], df1.y.iloc[i]), df1.r.iloc[i]/pxDimension, color = 'blue', fill = False))
        ax.set(xticks=[], yticks=[], title = 'Initial Frame')
        ax1.set(xticks=[], yticks=[], title = 'Final Frame')
        plt.tight_layout()
        if save_verb:
            plt.savefig(f'./{res_path}/initial_final_frame.png', bbox_inches='tight')
            #plt.savefig(f'{pdf_res_path}/initial_final_frame.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

#############################################################################################################
#                                              ABP SIMULATION
#############################################################################################################
if ABP_verb:
    droplet_radius = 1
    outer_radius = 10
    num_droplets = 1000
    dt = 0.001 # time step size
    time_steps = 100000

    v0 = 10 # magnitude of the self-propulsion velocity
    D_r = 1 # rotational diffusion coefficient
    t_r = 1 # characteristic time of the rotational diffusion
    D_t = 1 # translational diffusion coefficient


    # Initialize droplet positions and orientations
    positions = np.random.uniform(-outer_radius, outer_radius, size = (num_droplets, 2))
    orientations = np.random.uniform(0, 2*np.pi, size = num_droplets)

    frames, x, y, label = [], [], [], []

    for step in tqdm(range(time_steps)):
        # update magnitude of self-propulsion velocity
        # Update positions
        positions += v0 * np.array([np.cos(orientations), np.sin(orientations)]).T * dt + \
                        np.random.normal(scale=np.sqrt(2 * D_t * dt), size=(num_droplets, 2))
        orientations += np.random.normal(scale=np.sqrt(2 * D_r * dt), size=num_droplets)
        orientations %= 2 * np.pi

        frames += [step for i in range(num_droplets)]
        x += list(positions[:, 0])
        y += list(positions[:, 1])
        label += [i for i in range(num_droplets)]
        
    trajectories = pd.DataFrame({'frame': frames, 'x': x, 'y': y, 'particle': label})
    trajectories = trajectories.sort_values(['particle', 'frame'])

    x = trajectories.x.values.reshape(num_droplets, time_steps)
    dx = x - x[:, 0][:, np.newaxis]
    y = trajectories.y.values.reshape(num_droplets, time_steps)
    dy = y - y[:, 0][:, np.newaxis]
    squared_displacement = dx**2 + dy**2
    msd = np.mean(squared_displacement, axis = 0)

    t = np.arange(time_steps)*dt
    model = (4 * D_t + v0**2 *t_r)*t + (v0**2*t_r**2)/2 * (np.exp(-2*t/t_r)-1)
    model2 = 4 * D_t * t + 2*v0**2 /D_r**2 * (D_r * t - 1 + np.exp(-D_r * t))
    model3 = (4 * D_t + 2 * v0**2 * t_r)*t + 2* v0**2 * t_r**2 * (np.exp(-t/t_r)-1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(t, model, '--k', label = 'Flowing Matter model')
    ax.plot(t, model2, '--g', label = 'Article model')
    ax.plot(t, model3, '--b', label = 'Article model 2')
    ax.axvline(t_r, color = 'r', linestyle = '--', label = r'$t_r$')
    ax.scatter(t, msd, s = 1, color = 'b', label = 'simulation')
    ax.legend()
    ax.grid()
    ax.set(xscale='log', yscale='log', xlabel = 'time', ylabel = 'MSD', title = 'Simulation of 1000 ABPs')
    plt.savefig('ABP_simulation.png', bbox_inches='tight', dpi = 500)
    plt.show()

#############################################################################################################
#                                              RADIUS ANALYSIS
#############################################################################################################
if radius_verb:
    print('Radius as seen from above and depth analysis...')
    mean_radius_b = np.mean(np.array(trajectories.r).reshape(nFrames, nDrops)[:, ~red_mask], axis=1)
    mean_radius_r = np.mean(np.array(trajectories.r).reshape(nFrames, nDrops)[:, red_mask], axis=1)

    fit_b, pcov_b = curve_fit(powerLaw, frames[1:]/fps, mean_radius_b[1:], p0 = [1., 1.], maxfev = 1000)
    fit_r, pcov_r = curve_fit(powerLaw, frames[1:]/fps, mean_radius_r[1:], p0 = [1., 1.], maxfev = 1000)
    print(f'Blue droplets: r = {fit_b[0]:.2f} t^{fit_b[1]:.2f}')
    print(f'Red droplets: r = {fit_r[0]:.2f} t^{fit_r[1]:.2f}')

    fit_b_exp, pcov_b_exp = curve_fit(exp, frames/fps, mean_radius_b, p0 = [1., 1., 0.], maxfev = 1000)
    fit_r_exp, pcov_r_exp = curve_fit(exp, frames/fps, mean_radius_r, p0 = [1., 1., 0.], maxfev = 1000)
    print(f'Blue droplets: r = {fit_b_exp[0]:.2f} exp(-t/{fit_b_exp[1]:.2f}) + {fit_b_exp[2]:.2f}')
    print(f'Red droplets: r = {fit_r_exp[0]:.2f} exp(-t/{fit_r_exp[1]:.2f}) + {fit_r_exp[2]:.2f}')

    mean_r_wind = np.zeros(nSteps)
    d_wind      = np.zeros((nSteps, nDrops))
    d_wind_std  = np.zeros((nSteps, nDrops))
    for i, start in enumerate(startFrames):
        temp = trajectories.loc[(trajectories.frame >= start) & (trajectories.frame < start+window)]
        mean_r_wind[i] = np.mean(temp.r.values)
        for j in range(nDrops):
            temp_j           = temp.loc[temp.particle == j].r.values
            d_wind[i, j]     = np.mean(temp_j)
            d_wind_std[i, j] = np.std(temp_j)

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 4), sharex=True, sharey=True)
        ax.plot(frames/fps, mean_radius_b, 'b', label = "Blue droplets")
        ax.set(xlabel  = "Time [s]", ylabel = f"r [{dimension_units}]")
        ax.grid(True,  linestyle = '-', color = '0.75')
        ax.legend()
        ax1.plot(frames/fps, mean_radius_r, 'r', label = "Red droplets")
        ax1.set(xlabel = "Time [s]", ylabel = f"r [{dimension_units}]")
        ax1.grid(True, linestyle = '-', color = '0.75')
        ax1.legend()
        plt.suptitle(f"Mean radius of the droplets - {system_name}")
        plt.tight_layout()
        if save_verb: 
            plt.savefig(res_path     + "/dimension_analysis/mean_radius.png", bbox_inches='tight')
            #plt.savefig(pdf_res_path + "/dimension_analysis/mean_radius.pdf", bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 4), sharex=True, sharey=True)
        ax.plot(frames/fps, mean_radius_b, 'b', label = "Blue droplets")
        ax.plot(frames[1:]/fps, powerLaw(frames[1:]/fps, *fit_b), 'k--')
        ax.plot(frames/fps, exp(frames/fps, *fit_b_exp), 'r--')
        ax.set(xscale='log', yscale='log')
        ax.grid()
        ax1.plot(frames/fps, mean_radius_r, 'r', label = "Red droplets")
        ax1.plot(frames[1:]/fps, powerLaw(frames[1:]/fps, *fit_r), 'k--')
        ax1.plot(frames/fps, exp(frames/fps, *fit_r_exp), 'b--')
        ax1.set(xscale='log', yscale='log')
        ax1.grid()
        plt.suptitle(f"Mean radius of the droplets - {system_name}")
        plt.tight_layout()
        if save_verb:
            plt.savefig(res_path     + "/dimension_analysis/mean_radius_fit.png", bbox_inches='tight')
            #plt.savefig(pdf_res_path + "/dimension_analysis/mean_radius_fit.pdf", bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        ax.plot(startFrames/fps + windowLenght/2, mean_r_wind, 'k--', linewidth = 2, zorder=20)
        for i in range(nDrops):
            if i in red_particle_idx:
                ax.plot(startFrames/fps + windowLenght/2, d_wind[:, i], 'r-', zorder=20, alpha = 1)
            else:
                ax.plot(startFrames/fps + windowLenght/2, d_wind[:, i], 'b-', zorder=0, alpha = 0.5)
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.set(xlabel = "Window time [s]", ylabel = f"r [{dimension_units}]", title = f"Droplet radius by window time - {system_name}")
        ax.grid(linewidth = 0.2)
        ax.legend(["Mean", "Blue droplets", "Red droplets"])
        if save_verb: 
            plt.savefig(res_path + "/dimension_analysis/radius_wind.png", bbox_inches='tight')
            #plt.savefig(pdf_res_path + "/dimension_analysis/radius_wind.pdf", bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

    # Droplets depth from radius as seen from above
    initial_r_b = np.mean(trajectories.loc[(trajectories.frame==0) & ~(trajectories.particle.isin(red_particle_idx))].r.values)
    r_b = np.mean(trajectories.loc[~(trajectories.particle.isin(red_particle_idx))].r.values.reshape(nFrames, nDrops-len(red_particle_idx)), axis=1)
    h_b = (2*initial_r_b - np.sqrt(4*initial_r_b**2 - 4*r_b**2))/2

    initial_r_r = np.mean(trajectories.loc[(trajectories.frame==0) & (trajectories.particle.isin(red_particle_idx))].r.values)
    r_r = np.mean(trajectories.loc[(trajectories.particle.isin(red_particle_idx))].r.values.reshape(nFrames, len(red_particle_idx)), axis=1)
    h_r = (2*initial_r_r - np.sqrt(4*initial_r_r**2 - 4*r_r**2))/2

    if plot_verb:
        # take solution which has h-r <0
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frames/fps, h_b-r_b, 'b', label='Blue droplet')
        ax.plot(frames/fps, h_r-r_r, 'r', label='Red droplet')
        ax.set(xlabel='Time [s]', ylabel='Z [mm]', title=f'Droplets depth over time - {system_name}')
        ax.grid(linewidth = 0.2)
        ax.legend()
        if save_verb:
            plt.savefig(f'{res_path}/dimension_analysis/depth_over_time.png')
            #plt.savefig(f'{pdf_res_path}/dimension_analysis/depth_over_time.pdf')
        if show_verb:
            plt.show()
        else:
            plt.close()

#############################################################################################################
#                                              MSD ANALYSIS
#############################################################################################################
if msd_verb:
    print('Mean Squared Displacement analysis...')
    # Global (computed on the full trajectory) IMSD and EMSD
    imsd, fit, pw_exp = get_imsd(trajectories, pxDimension, fps, maxLagtime, x_diffusive)
    MSD_b, MSD_r, fit = get_emsd(imsd, fps, red_mask, x_diffusive)

    alpha_b = [round(fit['pw_exp_b'][0, 1], 3), round(fit['pw_exp_b'][1, 1], 3)]
    k_b = [round(fit['pw_exp_b'][0, 0], 3), round(fit['pw_exp_b'][1, 0], 3)]
    alpha_r = [round(fit['pw_exp_r'][0, 1], 3), round(fit['pw_exp_r'][1, 1], 3)]
    k_r = [round(fit['pw_exp_r'][0, 0], 3), round(fit['pw_exp_r'][1, 0], 3)]
    print(f'Blue droplets: a = {alpha_b[0]} ± {alpha_b[1]}, K = {k_b[0]} ± {k_b[1]} {dimension_units}²')
    print(f'Red droplets:  a = {alpha_r[0]} ± {alpha_r[1]}, K = {k_r[0]} ± {k_r[1]} {dimension_units}²')

    if plot_verb:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i in range(nDrops):
            ax.plot(imsd.index, imsd[i], color = colors[i])
        ax.set(xscale = 'log', yscale = 'log', xlabel = 'Time Lag [s]', ylabel = r'$\overline{\delta^2(\tau)}$ [$mm^2$]')
        ax.grid(linewidth = 0.2)
        plt.suptitle(f'Mean Squared Displacement - {system_name}')
        if save_verb:
            plt.savefig(f'./{res_path}/mean_squared_displacement/IMSD.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/IMSD.pdf', bbox_inches='tight')
        plt.close()

        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        for i in range(nDrops):
            ax.plot(imsd.index, imsd[i], color = colors[i])
        ax.set(xscale = 'log', yscale = 'log', xlabel = 'Time Lag [s]', ylabel = r'$\langle \delta r^2 \rangle$ [$mm^2$]')
        ax.grid(linewidth = 0.2)
        ax1.scatter(np.arange(nDrops), pw_exp[:, 0, 1], color = colors)
        ax1.set(xlabel = 'Particle ID', ylabel = r'$\alpha$')
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Mean Squared Displacement - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/IMSD_2.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/IMSD_2.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()
        
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(8, 5), tight_layout=True)
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(nDrops):
            ax1.plot(imsd.index, imsd.values[:, i], color = colors[i], linewidth = 0.5)
        ax1.set(xscale='log', yscale = 'log', xlabel = 'lag time [s]', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]', title = 'IMSD')
        ax1.grid(linewidth = 0.2)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(np.arange(nDrops), pw_exp[:, 0, 1], s = 10,  color = colors)
        ax2.set(xlabel = 'Droplet ID', ylabel = r'$\alpha$', title = 'power law exponents')
        ax2.grid()

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(np.arange(nDrops), pw_exp[:, 0, 0], s = 10, color = colors)
        ax3.set(xlabel='Droplet ID', ylabel = r'$K_\alpha \; [mm^2/s^\alpha]$', title = 'Diffusion coefficients')
        ax3.grid()
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/IMSD_v2.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/IMSD_v2.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
            
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(pw_exp[:, 0, 0], pw_exp[:, 0, 1], s = 10,  color = colors)
        ax.set(xlabel = r'$K_\alpha \; [mm^2/s^\alpha]$', ylabel = r'$\alpha$', title = f'Diffusion coefficients vs Scaling exponent - {system_name}')
        ax.grid(linewidth = 0.2)
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/k_alpha_scatterplot.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/k_alpha_scatterplot.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        ax.plot(imsd.index, MSD_b[0], 'b-', label = 'Blue droplets') 
        ax.plot(x_diffusive, fit['fit_b'], 'b--')
        ax.fill_between(imsd.index, MSD_b[0] - MSD_b[1], MSD_b[0] + MSD_b[1], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
        ax.plot(imsd.index, MSD_r[0], 'r-', label = 'Red droplets')
        ax.plot(x_diffusive, fit['fit_r'], 'r--')
        ax.fill_between(imsd.index, MSD_r[0] - MSD_r[1], MSD_r[0] + MSD_r[1], alpha=0.5, edgecolor='#FF0000', facecolor='#FF5A52')
        ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]',   
                xlabel = 'lag time $t$ [s]', title = f'EMSD - {system_name}')
        ax.legend()
        ax.grid(linewidth = 0.2)
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/EMSD.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/EMSD.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

    MSD_wind, fit_wind, pw_exp_wind = get_imsd_windowed(nDrops, nSteps, startFrames, endFrames, trajectories, pxDimension, fps, maxLagtime, x_diffusive)
    EMSD_wind_b, EMSD_wind_r, fit_dict = get_emsd_windowed(MSD_wind, x_diffusive, fps, red_mask, nSteps, maxLagtime, x_diffusive)

    max_alpha_b = max(fit_dict['pw_exp_wind_b'][:, 0, 1])
    max_alpha_b_std = fit_dict['pw_exp_wind_b'][np.where(fit_dict['pw_exp_wind_b'][:, 0, 1] == max_alpha_b)[0][0], 1, 1]
    print('Max α blue:', round(max_alpha_b, 3), '±', round(max_alpha_b_std, 3))

    max_d_b = max(fit_dict['pw_exp_wind_b'][:, 0, 0])
    max_d_b_std = fit_dict['pw_exp_wind_b'][np.where(fit_dict['pw_exp_wind_b'][:, 0, 0] == max_d_b)[0][0], 1, 0]
    print('Max D blue:', round(max_d_b, 3), '±', round(max_d_b_std, 3))

    min_alpha_b = min(fit_dict['pw_exp_wind_b'][:, 0, 1])
    min_alpha_b_std = fit_dict['pw_exp_wind_b'][np.where(fit_dict['pw_exp_wind_b'][:, 0, 1] == min_alpha_b)[0][0], 1, 1]
    print('Min α blue:', round(min_alpha_b, 3), '±', round(max_alpha_b_std, 3))

    min_d_b = min(fit_dict['pw_exp_wind_b'][:, 0, 0])
    min_d_b_std = fit_dict['pw_exp_wind_b'][np.where(fit_dict['pw_exp_wind_b'][:, 0, 0] == min_d_b)[0][0], 1, 0]
    print('Min D blue:', round(min_d_b, 3), '±', round(min_d_b_std, 3))

    max_alpha_r = max(fit_dict['pw_exp_wind_r'][:, 0, 1])
    max_alpha_b_std = fit_dict['pw_exp_wind_r'][np.where(fit_dict['pw_exp_wind_r'][:, 0, 1] == max_alpha_r)[0][0], 1, 1]
    print('Max α red:', round(max_alpha_r, 3), '±', round(max_alpha_b_std, 3))

    max_d_r = max(fit_dict['pw_exp_wind_r'][:, 0, 0])
    max_d_b_std = fit_dict['pw_exp_wind_r'][np.where(fit_dict['pw_exp_wind_r'][:, 0, 0] == max_d_r)[0][0], 1, 0]
    print('Max D red:', round(max_d_r, 3), '±', round(max_d_b_std, 3))

    min_alpha_r = min(fit_dict['pw_exp_wind_r'][:, 0, 1])
    min_alpha_b_std = fit_dict['pw_exp_wind_r'][np.where(fit_dict['pw_exp_wind_r'][:, 0, 1] == min_alpha_r)[0][0], 1, 1]
    print('Min α red:', round(min_alpha_r, 3), '±', round(max_alpha_b_std, 3))

    min_d_r = min(fit_dict['pw_exp_wind_r'][:, 0, 0])
    min_d_b_std = fit_dict['pw_exp_wind_r'][np.where(fit_dict['pw_exp_wind_r'][:, 0, 0] == min_d_r)[0][0], 1, 0]
    print('Min D red:', round(min_d_r, 3), '±', round(min_d_b_std, 3))

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), sharex = True)
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 1], 'b-', alpha = 0.5, label = 'Blue droplets')
        ax.fill_between(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 1] - fit_dict['pw_exp_wind_b'][:, 1, 1],     
                            fit_dict['pw_exp_wind_b'][:, 0, 1] + fit_dict['pw_exp_wind_b'][:, 1, 1],
                            alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 1], 'r-', alpha = 0.5, label = 'Red droplets ')
        ax.fill_between(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 1] - fit_dict['pw_exp_wind_r'][:, 1, 1],
                            fit_dict['pw_exp_wind_r'][:, 0, 1] + fit_dict['pw_exp_wind_r'][:, 1, 1],
                            alpha=0.5, edgecolor='#F0FFFF', facecolor='#FF5A52')
        ax.plot(startFrames/fps + windowLenght/2, np.ones(nSteps), 'k-')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2), title = f'Power Law Exponents')
        ax1.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 0], 'b-', alpha = 0.5, label = 'Blue droplets')
        ax1.set(xlabel = 'Window time [s]', title = f'Generalized Diffusion Coefficients')
        ax1.set_ylabel(r'$K{_\alpha, blue} \; [mm^2/s^\alpha]$', color = 'b')
        ax1.grid(linewidth = 0.2)
        ax1.yaxis.label.set_color('blue')
        ax1.tick_params(axis='y', colors='blue')
        ax2 = ax1.twinx()
        ax2.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 0], 'r-', alpha = 0.5, label = 'Red droplets')
        ax2.set_ylabel(r'$K{_\alpha, red} \; [mm^2/s^\alpha]$', color = 'r')
        ax2.yaxis.label.set_color('red')
        ax2.tick_params(axis='y', colors='red')
        plt.suptitle(f'Results for the EMSD windowed analysis {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_windowed.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/windowed_analysis/EMSD_windowed.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
            
        # Power law exponents plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(f'Power Law Exponents - {system_name}')
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 1], 'b-', alpha = 0.5, label = 'Blue droplets')
        ax.fill_between(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 1] - fit_dict['pw_exp_wind_b'][:, 1, 1],     
                            fit_dict['pw_exp_wind_b'][:, 0, 1] + fit_dict['pw_exp_wind_b'][:, 1, 1],
                            alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 1], 'r-', alpha = 0.5, label = 'Red droplets ')
        ax.fill_between(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 1] - fit_dict['pw_exp_wind_r'][:, 1, 1],
                            fit_dict['pw_exp_wind_r'][:, 0, 1] + fit_dict['pw_exp_wind_r'][:, 1, 1],
                            alpha=0.5, edgecolor='#F0FFFF', facecolor='#FF5A52')
        ax.plot(startFrames/fps + windowLenght/2, np.ones(nSteps), 'k-')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2))
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_alpha.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/windowed_analysis/EMSD_alpha.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
        

        # Generalized Diffusion Coefficients plot
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4),sharex = True)
        plt.suptitle(f'Generalized Diffusion Coefficients - {system_name}')
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 0], 'b-', alpha = 0.5, label = 'Blue droplets')
        ax.set(xlabel = 'Window time [s]', ylabel = r'$K_\alpha$ [$mm^2/s^\alpha$]')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax1.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 0], 'r-', alpha = 0.5, label = 'Red droplets ')
        ax1.legend()
        ax1.grid(linewidth = 0.2)
        ax1.set(xlabel = 'Window time [s]')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_D.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/windowed_analysis/EMSD_D.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        # Generalized Diffusion Coefficients plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        plt.suptitle(f'Generalized Diffusion Coefficients - {system_name}')
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 0], 'b-', alpha = 0.5, label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 0], 'r-', alpha = 0.5, label = 'Red droplets')
        ax.set(xlabel = 'Window time [s]', ylabel = r'$K_\alpha$ [$mm^2/s^\alpha$]')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_D_v2.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/windowed_analysis/EMSD_D_v2.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:, 0, 0], 'b-', alpha = 0.5, label = 'Blue droplets')
        ax.set(xlabel = 'Window time [s]')
        ax.set_ylabel(r'$K{_\alpha, blue} \; [mm^2/s^\alpha]$', color = 'b')
        ax.grid(linewidth = 0.2)
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax1 = ax.twinx()
        ax1.plot(startFrames/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:, 0, 0], 'r-', alpha = 0.5, label = 'Red droplets')
        ax1.set_ylabel(r'$K{_\alpha, red} \; [mm^2/s^\alpha]$', color = 'r')
        ax1.yaxis.label.set_color('red')
        ax1.tick_params(axis='y', colors='red')
        plt.suptitle(f'Generalized Diffusion Coefficients - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_D_v3.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/mean_squared_displacement/windowed_analysis/EMSD_D_v3.pdf', bbox_inches='tight')
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
            def update_graph(step):
                for i in range(nDrops):
                    graphic_data[i].set_ydata(np.array(MSD_wind[step].iloc[:, i]))
                    graphic_data2[i].set_data(startFrames[:step]/fps + windowLenght/2, pw_exp_wind[:step, i, 0, 1])
                title.set_text(f'Mean Squared Displacement - {system_name} - window [{startFrames[step]/fps} - {endFrames[step]/fps}] s')
                ax1.set_xlim(frames[0]/fps, startFrames[step]/fps + windowLenght/2 + 100)
                return graphic_data, graphic_data2,
            title = ax.set_title(f'Mean Squared Displacement - {system_name} - window [{startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps}] s')
            graphic_data = []
            for i in range(nDrops):
                if i in red_particle_idx:
                    graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]), color=colors[i], alpha = 0.3)[0])
                else:
                    graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]), color=colors[i], alpha = 0.3)[0])
            ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]', xlabel = 'lag time $t$ [s]', ylim = (10**(-5), 10**5))
            ax.grid(linewidth = 0.2)
            graphic_data2 = []
            for i in range(nDrops):
                if i in red_particle_idx:
                    graphic_data2.append(ax1.plot(startFrames[0]/fps + windowLenght/2, pw_exp_wind[0, i, 0, 1], color=colors[i], alpha = 0.3)[0])
                else:
                    graphic_data2.append(ax1.plot(startFrames[0]/fps + windowLenght/2, pw_exp_wind[0, i, 0, 1], color=colors[i], alpha = 0.3)[0])
            ax1.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2))
            ax1.set_xlim(frames[0]/fps, startFrames[0]/fps + windowLenght/2 + 100)
            ax1.grid(linewidth = 0.2)
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/mean_squared_displacement/windowed_analysis/IMSD_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.close()


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
                title.set_text(f'Mean Squared Displacement - {system_name} - window {startFrames[step]/fps} - {endFrames[step]/fps} seconds')
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
                line.set_data(startFrames[:step]/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][:step, 0, 1])
                line1.set_data(startFrames[:step]/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][:step, 0, 1]) 
                line2.set_data(startFrames[:step]/fps + windowLenght/2, np.ones(step)) 
                ax1.set_xlim(frames[0]/fps, startFrames[step]/fps + windowLenght/2 + 100)
                return graphic_data, fill_graph, line, line1, 

            title = ax.set_title(f'Mean Squared Displacement - {system_name} - window {startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps} seconds')
            graphic_data = []
            graphic_data.append(ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), EMSD_wind_b[0][0], 'b-', alpha=0.5, label = 'Blue droplets')[0])
            graphic_data.append(ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), EMSD_wind_r[0][0], 'r-' , label = 'Red droplets')[0] )
            fill_graph = ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), Y1_msd_b[0], Y2_msd_b[0], alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
            fill_graph2 = ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), Y1_msd_r[0], Y2_msd_r[0], alpha=0.5, edgecolor='#FF5A52', facecolor='#FF5A52')

            ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]', xlabel = 'lag time $t$ [s]', ylim=(10**(-5), 10**5))
            ax.legend()
            
            ax.grid(linewidth = 0.2)
            line, = ax1.plot(startFrames[0]/fps + windowLenght/2, fit_dict['pw_exp_wind_b'][0, 0, 1], 'b-', alpha = 0.5, label = 'Blue droplets')
            line1, = ax1.plot(startFrames[0]/fps + windowLenght/2, fit_dict['pw_exp_wind_r'][0, 0, 1], 'r-', alpha = 0.5, label = 'Red droplets')
            line2, = ax1.plot(startFrames[0]/fps + windowLenght/2, 1, 'k-')
            ax1.legend()
            ax1.set_xlim(frames[0]/fps, startFrames[0]/fps + windowLenght/2 + 100)
            ax1.grid(linewidth=0.2)
            ax1.set(xlabel = 'Window time [s]', ylabel = r'$\alpha$', ylim = (0, 2))
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.close()

            print('Done')# Lower and Higher bounds for fill between 
            Y1_msd_b = EMSD_wind_b[0] - EMSD_wind_b[1]
            Y2_msd_b = EMSD_wind_b[0] + EMSD_wind_b[1]
            Y1_msd_r = EMSD_wind_r[0] - EMSD_wind_r[1]
            Y2_msd_r = EMSD_wind_r[0] + EMSD_wind_r[1]

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
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
                title.set_text(f'Mean Squared Displacement - {system_name} - window {startFrames[step]/fps} - {endFrames[step]/fps} seconds')
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
                return graphic_data, fill_graph

            title = ax.set_title(f'Mean Squared Displacement - {system_name} - window {startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps} seconds')
            graphic_data = []
            graphic_data.append(ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), EMSD_wind_b[0][0], 'b-', alpha=0.5, label = 'Blue droplets')[0])
            graphic_data.append(ax.plot(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), EMSD_wind_r[0][0], 'r-' , label = 'Red droplets')[0] )
            fill_graph  = ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), Y1_msd_b[0], Y2_msd_b[0], alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
            fill_graph2 = ax.fill_between(np.arange(1/fps, maxLagtime/fps + 1/fps, 1/fps), Y1_msd_r[0], Y2_msd_r[0], alpha=0.5, edgecolor='#FF5A52', facecolor='#FF5A52')

            ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$mm^2$]', xlabel = 'lag time $t$ [s]', ylim=(10**(-5), 10**5))
            ax.legend()
            ax.grid(linewidth = 0.2)
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/mean_squared_displacement/windowed_analysis/EMSD_wind_v2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            plt.close()


#############################################################################################################
#                                              VELOCITY ANALYSIS
#############################################################################################################
if velocity_verb:
    bin_borders = np.arange(0, 100, .2)*pxDimension
    bin_centers = (bin_borders[1:] + bin_borders[:-1]) / 2
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

    print(f'Speed distribution analysis...')

    blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajectories, subsample_factor, fps)
    v_blue = ys.speed_ensemble(blueTrajs, step = 1)*pxDimension
    v_red  = ys.speed_ensemble(redTrajs, step = 1)*pxDimension
    mean_v_blue = np.mean(v_blue)
    mean_v_red  = np.mean(v_red)

    # fit speed distributions with 2D Maxwell Boltzmann distribution
    sigma_blue, sigma_blue_std = fit_hist(v_blue, bin_borders, MB_2D, [1.])
    sigma_red , sigma_red_std  = fit_hist(v_red , bin_borders, MB_2D, [1.])
    
    # compute R²
    y = np.histogram(v_blue, bins = bin_borders, density = True)[0]
    y_fit = MB_2D(bin_centers, sigma_blue)
    r2_blue = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    y = np.histogram(v_red, bins = bin_borders, density = True)[0]
    y_fit = MB_2D(bin_centers, sigma_red)
    r2_red = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

    print(f'Blue droplets --> R² = {round(r2_blue,2)} -- μ = {round(mean_v_blue, 3)} mm/s -- σ = {round(sigma_blue[0], 3)} ± {round(sigma_blue_std[0], 3)} mm/s')
    print(f'Red droplets --> R² = {round(r2_red,2)} -- μ = {round(mean_v_red, 3)} mm/s --σ = {round(sigma_red[0], 3)} ± {round(sigma_red_std[0], 3)} mm/s')

    # fit speed distributions with a generalization of a 2D Maxwell Boltzmann distribution
    sigma_blue_g, sigma_blue_std_g = fit_hist(v_blue, bin_borders, MB_2D_generalized, [1., 2., 1.])
    sigma_red_g, sigma_red_std_g = fit_hist(v_red, bin_borders, MB_2D_generalized, [1., 2., 1.])

    # compute R²
    y = np.histogram(v_blue, bins = bin_borders, density = True)[0]
    y_fit = MB_2D_generalized(bin_centers, *sigma_blue_g)
    r2_blue_g = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    y = np.histogram(v_red, bins = bin_borders, density = True)[0]
    y_fit = MB_2D_generalized(bin_centers, *sigma_red_g)
    r2_red_g = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

    print(f'Blue droplets --> R² = {round(r2_blue_g, 3)} -- σ = {round(sigma_blue_g[0], 3)} ± {round(sigma_blue_std_g[0], 3)} mm/s -- b = {round(sigma_blue_g[1], 3)} ± {round(sigma_blue_std_g[1], 3)} -- A = {round(sigma_blue_g[2], 3)} ± {round(sigma_blue_std_g[2], 3)} ')
    print(f'Red droplets --> R² = {round(r2_red_g, 3)} -- σ = {round(sigma_red_g[0], 4)} ± {round(sigma_red_std_g[0], 4)} mm/s -- b = {round(sigma_red_g[1], 3)} ± {round(sigma_red_std_g[1], 3)} -- A = {round(sigma_red_g[2], 3)} ± {round(sigma_red_std_g[2], 3)}')

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 4), sharey=True, sharex = True)
        ax.hist(v_blue, bins = bin_borders, **default_kwargs_blue, label = 'Blue droplets')
        ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, sigma_blue), 'k-', label = f'$ σ= {sigma_blue[0]:.3f} \pm {sigma_blue_std[0]:.3f}$')
        ax.set(xlabel = f'v [{speed_units}]', ylabel = 'pdf [s/mm]')
        ax.legend()
        ax.grid(linewidth = 0.2)

        ax1.hist(v_red, bins = bin_borders, **default_kwargs_red, label = 'Red droplets')
        ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, sigma_red), 'k-', label = f'$ σ = {sigma_red[0]:.3f} \pm {sigma_red_std[0]:.3f}$')
        ax1.set(xlabel = f'v [{speed_units}]', xlim = (-.1, 5), ylim = (0, 7))
        ax1.legend()
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Velocity distribution - MB fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/speed_distribution.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/speed_distribution.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()
        
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 4), sharey=True, sharex = True)
        ax.hist(v_blue, bins = bin_borders, **default_kwargs_blue, label = 'Blue droplets')
        ax.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit, *sigma_blue_g), 'k-', label = f'$ σ= {sigma_blue_g[0]:.3f} \pm {sigma_blue_std_g[0]:.3f}$')
        ax.set(xlabel = f'v [{speed_units}]', ylabel = 'pdf [s/mm]')
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax1.hist(v_red, bins = bin_borders, **default_kwargs_red, label = 'Red droplets')
        ax1.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit, *sigma_red_g), 'k-', label = f'$ σ = {sigma_red_g[0]:.3f} \pm {sigma_red_std_g[0]:.3f}$')
        ax1.set(xlabel = f'v [{speed_units}]', xlim = (-.1, 5), ylim = (0, 7))
        ax1.legend()
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Velocity distribution - Generalized MB fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/speed_distribution_generalized.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/speed_distribution_generalized.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()

    print('Windowed speed distribution analysis...')
    v_blue_wind, v_red_wind = speed_windowed(nDrops, nSteps, startFrames, endFrames,\
                                             red_particle_idx, trajectories, subsample_factor, fps, )
    v_blue_wind = np.array(v_blue_wind)*pxDimension
    v_red_wind  = np.array(v_red_wind)*pxDimension
    v_blue_wind_mean = np.mean(v_blue_wind, axis = 1)
    v_red_wind_mean  = np.mean(v_red_wind , axis = 1)

    # fit windowed speed distributions with 2D Maxwell Boltzmann distribution
    r2_blue_wind = np.zeros(nSteps)
    r2_red_wind = np.zeros(nSteps)
    blue_fit_wind = np.ones((nSteps, 2))
    red_fit_wind = np.ones((nSteps, 2))
    for k in range(nSteps):
        blue_fit_wind[k, 0], blue_fit_wind[k, 1] = fit_hist(v_blue_wind[k], bin_borders, MB_2D, [1.])
        y = np.histogram(v_blue_wind[k], bins = bin_borders, density = True)[0]
        y_fit = MB_2D(bin_centers, blue_fit_wind[k, 0])
        r2_blue_wind[k] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

        red_fit_wind[k, 0], red_fit_wind[k, 1] = fit_hist(v_red_wind[k], bin_borders, MB_2D, [1.])
        y = np.histogram(v_red_wind[k], bins = bin_borders, density = True)[0]
        y_fit = MB_2D(bin_centers, red_fit_wind[k, 0])
        r2_red_wind[k] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    # fit windowed speed distributions with a generalization of a 2D Maxwell Boltzmann distribution
    r2_blue_g_wind = np.zeros(nSteps)
    r2_red_g_wind = np.zeros(nSteps)
    blue_fit_wind_g = np.zeros((nSteps, 3))
    blue_fit_wind_g_std = np.zeros((nSteps, 3))
    red_fit_wind_g = np.zeros((nSteps, 3))
    red_fit_wind_g_std = np.zeros((nSteps, 3))
    for step in range(nSteps):
        blue_fit_wind_g[step], blue_fit_wind_g_std[step] = fit_hist(v_blue_wind[step], bin_borders, MB_2D_generalized, [1., 2., 1.])
        red_fit_wind_g[step], red_fit_wind_g_std[step] = fit_hist(v_red_wind[step], bin_borders, MB_2D_generalized, [1., 2., 1.])
        y = np.histogram(v_blue_wind[step], bins = bin_borders, density = True)[0]
        y_fit = MB_2D_generalized(bin_centers, *blue_fit_wind_g[step])
        r2_blue_g_wind[step] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
        y = np.histogram(v_red_wind[step], bins = bin_borders, density = True)[0]
        y_fit = MB_2D_generalized(bin_centers,  *red_fit_wind_g[step])
        r2_red_g_wind[step] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

    if plot_verb:
        fig, ax = plt.subplots(1, 1, figsize = (12, 4))
        ax.plot(startFrames/fps + windowLenght/2, v_blue_wind_mean, 'b-',label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, v_red_wind_mean, 'r-', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.set(xlabel = 'Window time [s]', ylabel = f'v [{speed_units}]', title=f'Windowed mean speed - {system_name}')
        ax.legend()
        if save_verb:
            plt.savefig(f'./{res_path}/speed_distribution/speed_windowed_mean.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/speed_windowed_mean.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_wind, 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_wind, 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = r'$R^2$')
        test = ax1.errorbar(startFrames/fps + windowLenght/2, blue_fit_wind[:, 0], yerr = blue_fit_wind[:, 1], fmt = 'b', label = 'Blue droplets')
        ax1.set(ylim = (min(blue_fit_wind[:, 0]*0.8), max(blue_fit_wind[:, 0]*1.2)))
        ax1.errorbar(startFrames/fps + windowLenght/2, red_fit_wind[:, 0], yerr = red_fit_wind[:, 1], fmt = 'r', label = 'Red droplets')
        ax1.set(ylabel = r'$\sigma \; [mm/s]$', xlabel = 'Window time [s]', title = r'$\sigma$')
        ax1.legend()
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Windowed velocity distribution with 2D Maxwell Boltzmann fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/speed_distribution_windowed.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/speed_distribution_windowed.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_wind, 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_wind, 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = f'R² of the fit of the velocity distribution - {system_name}')
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/r2.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/r2.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        # Sigma of velocity distribution plot
        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        ax.errorbar(startFrames/fps + windowLenght/2, blue_fit_wind[:, 0], yerr = blue_fit_wind[:, 1], fmt = 'b', label = 'Blue droplets')
        ax.errorbar(startFrames/fps + windowLenght/2, red_fit_wind[:, 0], yerr = red_fit_wind[:, 1], fmt = 'r', label = 'Red droplets')
        ax.set(ylabel = r'$\sigma \; [mm/s]$', xlabel = 'Window time [s]', title = f'Sigma of MB distribution - {system_name}')
        ax.set_ylim(min(blue_fit_wind[:, 0]*0.8), max(blue_fit_wind[:, 0]*1.2))
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/sigma_MB.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/sigma_MB.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_g_wind, 'b', label = 'Generalized fit')
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_wind, 'b--', label = 'Maxwell-Boltzmann fit')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_g_wind, 'r', label = 'Generalized fit')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_wind, 'r--', label = 'Maxwell-Boltzmann fit')
        ax.set(ylabel = 'R2', xlabel = 'Window time [s]', title = f'R2 confront with MB Generalized fit - {system_name}')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/r2_generalized_MB.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/r2_generalized_MB.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        ax.plot(startFrames/fps + windowLenght/2, blue_fit_wind_g[:, 0], 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, red_fit_wind_g[:, 0], 'r', label = 'Red droplets')
        ax.set(xlabel = 'Window time [s]', ylabel = r'$\sigma$')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax1.plot(startFrames/fps + windowLenght/2, blue_fit_wind_g[:, 1], 'b', label = 'Blue droplets')
        ax1.plot(startFrames/fps + windowLenght/2, red_fit_wind_g[:, 1], 'r', label = 'Red droplets')
        ax1.set(xlabel = 'Window time [s]', ylabel = r'$\beta$')
        ax1.grid(linewidth = 0.2)
        ax2.plot(startFrames/fps + windowLenght/2, blue_fit_wind_g[:, 2], 'b', label = 'Blue droplets')
        ax2.plot(startFrames/fps + windowLenght/2, red_fit_wind_g[:, 2], 'r', label = 'Red droplets')
        ax2.grid(linewidth = 0.2)
        ax2.set(xlabel = 'Window time [s]', ylabel = r'$A$')
        plt.suptitle(f'Generalized MB fit parameters evolution - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/speed_distribution/fit_results_generalizedMB.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/speed_distribution/fit_results_generalizedMB.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        if animated_plot_verb:
            fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 5), sharex=True, sharey=True)
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
                    title.set_text(f'Velocity Distribution at window {startFrames[frame]/fps} - {endFrames[frame]/fps} seconds - {system_name}')

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

            _, _, bar_container = ax.hist(v_blue_wind[0], bin_borders, **default_kwargs_blue, label='Blue droplets')
            title = ax.set_title(f'{system_name} - window {startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps} seconds  - {system_name}')
            line, = ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, blue_fit_wind[0, 0]), label = 'MB fit')
            ax.set(xlabel = f'v [{speed_units}]', ylabel = 'pdf [s/mm]')
            ax.grid(linewidth = 0.2)
            ax.legend()

            _, _, bar_container2 = ax1.hist(v_red_wind[0], bin_borders,  **default_kwargs_red, label='Red droplets')
            #title2 = ax1.set_title(f'Red droplets velocity pdf {startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps} seconds')
            line1, = ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, red_fit_wind[0, 0]), label = 'MB fit')
            ax1.set(xlabel = f'v [{speed_units}]', ylabel = 'pdf [s/mm]', xlim = (-.1, 5), ylim = (0, 8))
            ax1.legend()
            ax1.grid(linewidth = 0.2)
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, prepare_animation(bar_container, bar_container2), nSteps, repeat=True, blit=False)
            if save_verb: ani.save(f'./{res_path}/speed_distribution/speed_wind.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()


            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)
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
                    title.set_text(f'Windowed speed distribution - {system_name} - [{startFrames[frame]/fps} - {endFrames[frame]/fps}] s')
                    # update histogram 1
                    n, _ = np.histogram(v_blue_wind[frame], bin_borders, density = True)
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    line.set_ydata(MB_2D(x_interval_for_fit, blue_fit_wind[frame, 0]))
                    line1.set_ydata(MB_2D_generalized(x_interval_for_fit, *blue_fit_wind_g[frame]))
                    # update histogram 2
                    n2, _ = np.histogram(v_red_wind[frame], bin_borders, density = True)
                    for count2, rect2 in zip(n2, bar_container2.patches):
                        rect2.set_height(count2)
                    line2.set_ydata(MB_2D(x_interval_for_fit, red_fit_wind[frame, 0]))
                    line3.set_ydata(MB_2D_generalized(x_interval_for_fit,  *red_fit_wind_g[frame]))
                    return bar_container.patches, bar_container2.patches
                return animate

            _, _, bar_container = ax.hist(v_blue_wind[0], bin_borders, **default_kwargs_blue, label='Blue droplets')
            title = ax.set_title(f'Windowed speed distribution - {system_name} - [{startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps}] s')
            line, = ax.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, blue_fit_wind[0, 0]), label = '2D MB fit')
            line1, = ax.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit,  *blue_fit_wind_g[0]), label='Generalized 2D MB fit')
            ax.set(xlabel = f'v [{speed_units}]', ylabel = 'pdf [s/mm]')
            ax.grid(linewidth = 0.2)
            ax.legend()

            _, _, bar_container2 = ax1.hist(v_red_wind[0], bin_borders,  **default_kwargs_red, label='Red droplets')
            line2, = ax1.plot(x_interval_for_fit, MB_2D(x_interval_for_fit, red_fit_wind[0, 0]), label='2D MB fit')
            line3, = ax1.plot(x_interval_for_fit, MB_2D_generalized(x_interval_for_fit,  *red_fit_wind_g[0]), label = 'Generalized 2D MB fit')
            ax1.set(xlabel = f'v [{speed_units}]', ylabel = 'pdf [s/mm]', xlim = (-.1, 5), ylim = (0, 8))
            ax1.grid(linewidth = 0.2)
            ax1.legend()

            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, prepare_animation(bar_container, bar_container2), nSteps, repeat=True, blit=False)
            if save_verb: ani.save(f'./{res_path}/speed_distribution/speed_wind_confront_generalized_MB.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb: 
                plt.show()
            else:
                plt.close()

#############################################################################################################
#                                              TURNING ANGLES ANALYSIS
#############################################################################################################
if turning_angles_verb:
    bin_borders_turn = np.arange(-np.pi, np.pi + 0.0001, np.pi/50)
    bin_centers_turn = bin_borders_turn[:-1] + np.diff(bin_borders_turn) / 2
    x_interval_for_fit_turn = np.linspace(bin_borders_turn[0], bin_borders_turn[-1], 10000)

    print(f'Turning angles analysis...')
    blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajectories, subsample_factor, fps)
    theta_blue = ys.turning_angles_ensemble(blueTrajs, centered = True)
    theta_red  = ys.turning_angles_ensemble(redTrajs, centered = True)

    # fit turning angles distributions with normal distribution
    sigma_blue, sigma_blue_std = fit_hist(theta_blue, bin_borders_turn, normal_distr, [1., 0.])
    sigma_red, sigma_red_std = fit_hist(theta_red, bin_borders_turn, normal_distr, [1., 0.])
    # compute R²
    y = np.histogram(theta_blue, bins = bin_borders_turn, density = True)[0]
    y_fit = normal_distr(bin_centers_turn, *sigma_blue)
    r2_blue = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    y = np.histogram(theta_red, bins = bin_borders_turn, density = True)[0]
    y_fit = normal_distr(bin_centers_turn, *sigma_red)
    r2_red = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    print(f'Blue droplets σ = {round(sigma_blue[0],3)} ± {round(sigma_blue_std[0],3)}, μ = {round(sigma_blue[1],4)} ± {round(sigma_blue_std[1],4)}, r2 = {round(r2_blue, 3)}')
    print(f'Red droplets σ = {round(sigma_red[0],3)} ± {round(sigma_red_std[0],3)}, μ = {round(sigma_red[1],4)} ± {round(sigma_red_std[1],4)}, r2 = {round(r2_red, 3)}')


    # fit turning angles distributions with lorentzian distribution
    gamma_blue, gamma_blue_std = fit_hist(theta_blue, bin_borders_turn, lorentzian_distr, [1., 0.])
    gamma_red, gamma_red_std = fit_hist(theta_red, bin_borders_turn, lorentzian_distr, [1., 0.])
    # compute R²
    y = np.histogram(theta_blue, bins = bin_borders_turn, density = True)[0]
    y_fit = lorentzian_distr(bin_centers_turn, *gamma_blue)
    r2_blue = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    y = np.histogram(theta_red, bins = bin_borders_turn, density = True)[0]
    y_fit = lorentzian_distr(bin_centers_turn, *gamma_red)
    r2_red = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

    print(f'Blue droplets γ = {round(gamma_blue[0],3)} ± {round(gamma_blue_std[0],3)}, μ = {round(gamma_blue[1],4)} ± {round(gamma_blue_std[1],4)}, r2 = {round(r2_blue, 3)}')
    print(f'Red droplets γ = {round(gamma_red[0],3)} ± {round(gamma_red_std[0],3)}, μ = {round(gamma_red[1],4)} ± {round(gamma_red_std[1],4)}, r2 = {round(r2_red, 3)}')

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        ax.hist(theta_blue, bin_borders_turn, **default_kwargs_blue, label='Blue droplets')
        ax.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *sigma_blue), 'k-',
                        label = f'$σ = {sigma_blue[0]:.2f} \pm {sigma_blue_std[0]:.2f}$')
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set(ylabel='pdf', xlabel= r'$\theta$ [rad]')
        ax.legend()
        ax.set_ylim(0, 3)
        ax.grid(linewidth = 0.2)
        ax1.hist(theta_red, bin_borders_turn, **default_kwargs_red, label='Red droplets')
        ax1.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *sigma_red), 'k-',
                        label = f'$σ = {sigma_red[0]:.2f} \pm {sigma_red_std[0]:.2f}$')
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax1.set(xlabel= r'$\theta$ [rad]')
        ax1.legend()
        ax1.set_ylim(0, 3)
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Turning angles pdf  - Gaussian fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/turn_ang_gaussian.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/turn_ang_gaussian.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
        
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        ax.hist(theta_blue, bin_borders_turn, **default_kwargs_blue, label='Blue droplets')
        ax.plot(x_interval_for_fit_turn, lorentzian_distr(x_interval_for_fit_turn, *gamma_blue), 'k-',
                        label = f'$γ = {gamma_blue[0]:.2f} \pm {gamma_blue_std[0]:.2f}$')
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set(ylabel='pdf', xlabel= r'$\theta$ [rad]')
        ax.legend()
        ax.set_ylim(0, 3)
        ax.grid(linewidth = 0.2)

        ax1.hist(theta_red, bin_borders_turn, **default_kwargs_red, label='Red droplets')
        ax1.plot(x_interval_for_fit_turn, lorentzian_distr(x_interval_for_fit_turn, *gamma_red), 'k-',
                        label = f'$γ = {gamma_red[0]:.2f} \pm {gamma_red_std[0]:.2f}$')
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax1.set(xlabel= r'$\theta$ [rad]')
        ax1.legend()
        ax1.set_ylim(0, 3)
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Turning angles pdf - Lorentzian fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/turn_ang_lorentzian.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/turn_ang_lorentzian.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

    print('Windowed turning angles analysis...')
    theta_blue_wind, theta_red_wind = turning_angles_windowed(nDrops, nSteps, startFrames, endFrames, red_particle_idx, trajectories, subsample_factor, fps)

    # fit windowed turning angles distributions with normal distribution
    blue_fit_wind_turn_gaussian = np.ones((nSteps, 2, 2))
    red_fit_wind_turn_gaussian = np.ones((nSteps, 2, 2))
    r2_blue_gaussian = np.zeros(nSteps)
    r2_red_gaussian = np.zeros(nSteps)
    for k in range(nSteps):
        blue_fit_wind_turn_gaussian[k, 0], blue_fit_wind_turn_gaussian[k, 1] = fit_hist(theta_blue_wind[k], bin_borders_turn, normal_distr, [2., 0.])
        red_fit_wind_turn_gaussian[k, 0], red_fit_wind_turn_gaussian[k, 1] = fit_hist(theta_red_wind[k], bin_borders_turn, normal_distr, [2., 0.])
        y = np.histogram(theta_blue_wind[k], bins = bin_borders_turn, density = True)[0]
        y_fit = normal_distr(bin_centers_turn, *blue_fit_wind_turn_gaussian[k, 0])
        r2_blue_gaussian[k] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
        y = np.histogram(theta_red_wind[k], bins = bin_borders_turn, density = True)[0]
        y_fit = normal_distr(bin_centers_turn, *red_fit_wind_turn_gaussian[k, 0])
        r2_red_gaussian[k] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # fit windowed turning angles distributions with lorentzian distribution
    blue_fit_wind_turn_lorentzian = np.ones((nSteps, 2, 2))
    red_fit_wind_turn_lorentzian = np.ones((nSteps, 2, 2))
    r2_blue_lorentzian = np.zeros(nSteps)
    r2_red_lorentzian = np.zeros(nSteps)
    for k in range(nSteps):
        blue_fit_wind_turn_lorentzian[k, 0], blue_fit_wind_turn_lorentzian[k, 1] = fit_hist(theta_blue_wind[k], bin_borders_turn, lorentzian_distr, [1., 0.])
        red_fit_wind_turn_lorentzian[k, 0], red_fit_wind_turn_lorentzian[k, 1] = fit_hist(theta_red_wind[k], bin_borders_turn, lorentzian_distr, [1., 0.])
        y = np.histogram(theta_blue_wind[k], bins = bin_borders_turn, density = True)[0]
        y_fit = lorentzian_distr(bin_centers_turn, *blue_fit_wind_turn_lorentzian[k, 0])

        r2_blue_lorentzian[k] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
        y = np.histogram(theta_red_wind[k], bins = bin_borders_turn, density = True)[0]
        y_fit = lorentzian_distr(bin_centers_turn, *red_fit_wind_turn_lorentzian[k, 0])
        r2_red_lorentzian[k] = 1 - (np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))

    if plot_verb: 
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 4), sharex = True)
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_gaussian, 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_gaussian, 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = r'$R^2$')
        ax1.plot(startFrames/fps + windowLenght/2, blue_fit_wind_turn_gaussian[:, 0, 0], 'b', label='Blue droplets')
        ax1.fill_between(startFrames/fps + windowLenght/2, blue_fit_wind_turn_gaussian[:, 0, 0] - blue_fit_wind_turn_gaussian[:, 1, 0],
                        blue_fit_wind_turn_gaussian[:, 0, 0] + blue_fit_wind_turn_gaussian[:, 1, 0], color = 'b', alpha = 0.2)
        ax1.set(ylabel = r'$\sigma \; [mm/s]$', xlabel = 'Window time [s]', title = r'$\sigma$')
        ax1.plot(startFrames/fps + windowLenght/2, red_fit_wind_turn_gaussian[:, 0, 0], 'r', label = 'Red droplets')
        ax1.fill_between(startFrames/fps + windowLenght/2, red_fit_wind_turn_gaussian[:, 0, 0] - red_fit_wind_turn_gaussian[:, 1, 0],
                            red_fit_wind_turn_gaussian[:, 0, 0] + red_fit_wind_turn_gaussian[:, 1, 0], color='r', alpha=0.2)
        ax1.legend()
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Windowed turning angles distribution - Gaussian fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/turn_ang_gaussian_windowed.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/turn_ang_gaussian_windowed.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_gaussian, 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_gaussian, 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = f'R² of the Gaussian fit of the turning angles distribution - {system_name}')
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/r2_gaussian.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/r2_gaussian.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
        ax.plot(startFrames/fps + windowLenght/2, blue_fit_wind_turn_gaussian[:, 0, 0], 'b', label='Blue droplets')
        ax.fill_between(startFrames/fps + windowLenght/2, blue_fit_wind_turn_gaussian[:, 0, 0] - blue_fit_wind_turn_gaussian[:, 1, 0],
                        blue_fit_wind_turn_gaussian[:, 0, 0] + blue_fit_wind_turn_gaussian[:, 1, 0], color = 'b', alpha = 0.2)
        
        ax.plot(startFrames/fps + windowLenght/2, red_fit_wind_turn_gaussian[:, 0, 0], 'r', label = 'Red droplets')
        ax.fill_between(startFrames/fps + windowLenght/2, red_fit_wind_turn_gaussian[:, 0, 0] - red_fit_wind_turn_gaussian[:, 1, 0],
                            red_fit_wind_turn_gaussian[:, 0, 0] + red_fit_wind_turn_gaussian[:, 1, 0], color='r', alpha=0.2)
        ax.set(ylabel = r'$\sigma \; [mm/s]$', xlabel = 'Window time [s]', title = f'Sigma of gaussian distribution - {system_name}')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/fit_results_gaussian.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/fit_results_gaussian.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_lorentzian, 'b', label = 'Lorentzian Fit')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_lorentzian, 'r', label = 'Lorentzian fit') 
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_gaussian, 'b--', label = 'Gaussian Fit')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_gaussian, 'r--', label = 'Gaussian Fit')
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = f'R² confront fit of the turning angles distribution - {system_name}')
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/r2_confront.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/r2_confront.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
            
        fig, (ax, ax1) = plt.subplots(1, 2, figsize = (12, 4), sharex=True)
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_lorentzian, 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_lorentzian, 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.grid(linewidth = 0.2)
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = r'$R^2$')
        ax1.plot(startFrames/fps + windowLenght/2, blue_fit_wind_turn_lorentzian[:, 0, 0], 'b', label='Blue droplets')
        ax1.fill_between(startFrames/fps + windowLenght/2, blue_fit_wind_turn_lorentzian[:, 0, 0] - blue_fit_wind_turn_lorentzian[:, 1, 0],
                        blue_fit_wind_turn_lorentzian[:, 0, 0] + blue_fit_wind_turn_lorentzian[:, 1, 0], color = 'b', alpha = 0.2)
        ax1.set(ylabel = r'$\sigma \; [mm/s]$', xlabel = 'Window time [s]', title = r'$\gamma$')

        ax1.plot(startFrames/fps + windowLenght/2, red_fit_wind_turn_lorentzian[:, 0, 0], 'r', label = 'Red droplets')
        ax1.fill_between(startFrames/fps + windowLenght/2, red_fit_wind_turn_lorentzian[:, 0, 0] - red_fit_wind_turn_lorentzian[:, 1, 0],
                            red_fit_wind_turn_lorentzian[:, 0, 0] + red_fit_wind_turn_lorentzian[:, 1, 0], color='r', alpha=0.2)
        ax1.legend()
        ax1.grid(linewidth = 0.2)
        plt.suptitle(f'Windowed turning angles distribution - Lorentzian fit - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/turn_ang_lorentzian_windowed.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/turn_ang_lorentzian_windowed.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()
        
        fig, ax = plt.subplots(1 ,1, figsize = (8, 4))
        ax.plot(startFrames/fps + windowLenght/2, r2_blue_lorentzian, 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, r2_red_lorentzian, 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.set(xlabel = 'Window time [s]', ylabel = r'$R^2$', title = f'R² of the Lorentzian fit of the turning angles distribution - {system_name}')
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/r2_lorentzian.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/r2_lorentzian.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(startFrames/fps + windowLenght/2, blue_fit_wind_turn_lorentzian[:, 0, 0], 'b', label='Blue droplets')
        ax.fill_between(startFrames/fps + windowLenght/2, blue_fit_wind_turn_lorentzian[:, 0, 0] - blue_fit_wind_turn_lorentzian[:, 1, 0],
                        blue_fit_wind_turn_lorentzian[:, 0, 0] + blue_fit_wind_turn_lorentzian[:, 1, 0], color = 'b', alpha = 0.2)
        
        ax.plot(startFrames/fps + windowLenght/2, red_fit_wind_turn_lorentzian[:, 0, 0], 'r', label = 'Red droplets')
        ax.fill_between(startFrames/fps + windowLenght/2, red_fit_wind_turn_lorentzian[:, 0, 0] - red_fit_wind_turn_lorentzian[:, 1, 0],
                            red_fit_wind_turn_lorentzian[:, 0, 0] + red_fit_wind_turn_lorentzian[:, 1, 0], color='r', alpha=0.2)
        ax.set(ylabel = r'$\gamma \; [mm/s]$', xlabel = 'Window time [s]', title = f'Gamma of Lorentzian distribution - {system_name}')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.legend()
        ax.grid(linewidth = 0.2)
        if save_verb: 
            plt.savefig(f'./{res_path}/turning_angles/fit_results_lorentzian.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/turning_angles/fit_results_lorentzian.pdf', bbox_inches='tight')
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
                    title.set_text(f'Turning angles pdf - {system_name} - window {startFrames[frame]/fps} - {endFrames[frame]/fps} seconds - {system_name}')
                    n, _ = np.histogram(theta_blue_wind[frame], bin_borders_turn, density = True)
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    line.set_ydata(normal_distr(x_interval_for_fit_turn, *blue_fit_wind_turn_gaussian[frame, 0]))
                    n2, _ = np.histogram(theta_red_wind[frame], bin_borders_turn, density = True)
                    for count2, rect2 in zip(n2, bar_container2.patches):
                        rect2.set_height(count2)
                    line1.set_ydata(normal_distr(x_interval_for_fit_turn, *red_fit_wind_turn_gaussian[frame, 0]))
                    return bar_container.patches, bar_container2.patches
                return animate
            _, _, bar_container = ax.hist(theta_red_wind[0], bin_borders_turn, **default_kwargs_blue, label='Blue droplets')
            line, = ax.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *blue_fit_wind_turn_gaussian[0, 0]), label='fit')
            title = ax.set_title(f'Turning angles pdf - {system_name} - window {startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps} seconds - {system_name}')
            ax.set(ylabel = 'pdf', ylim = (0, 3))
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            ax.grid(linewidth = 0.2)

            _, _, bar_container2 = ax1.hist(theta_red_wind[0], bin_borders_turn,  **default_kwargs_red, label='Red droplets')
            line1, = ax1.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *red_fit_wind_turn_gaussian[0, 0]), label='fit')
            ax1.set(ylabel = 'pdf', ylim = (0, 3))
            ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            ax1.grid(linewidth = 0.2)

            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, prepare_animation(bar_container, bar_container2), nSteps, repeat=True, blit=False)
            if save_verb:
                ani.save(f'./{res_path}/turning_angles/turn_ang_wind_gaussian.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()

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
                    title.set_text(f'Turning angles distribution - {system_name} - [{startFrames[frame]/fps} - {endFrames[frame]/fps}] s')
                    n, _ = np.histogram(theta_blue_wind[frame], bin_borders_turn, density = True)
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    line.set_ydata(lorentzian_distr(x_interval_for_fit_turn, *blue_fit_wind_turn_lorentzian[frame, 0]))
                    line1.set_ydata(normal_distr(x_interval_for_fit_turn, *blue_fit_wind_turn_gaussian[frame, 0]))
                    n2, _ = np.histogram(theta_red_wind[frame], bin_borders_turn, density = True)
                    for count2, rect2 in zip(n2, bar_container2.patches):
                        rect2.set_height(count2)
                    line2.set_ydata(lorentzian_distr(x_interval_for_fit_turn, *red_fit_wind_turn_lorentzian[frame, 0]))
                    line3.set_ydata(normal_distr(x_interval_for_fit_turn, *red_fit_wind_turn_gaussian[frame, 0]))
                    return bar_container.patches, bar_container2.patches
                return animate
            _, _, bar_container = ax.hist(theta_red_wind[0], bin_borders_turn, **default_kwargs_blue, label='Blue droplets')
            line, = ax.plot(x_interval_for_fit_turn, lorentzian_distr(x_interval_for_fit_turn, *blue_fit_wind_turn_lorentzian[0, 0]), label='Lorentzian fit')
            line1, = ax.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *blue_fit_wind_turn_gaussian[0, 0]), label='Gaussian fit')

            title = ax.set_title(f'Turning angles distribution - {system_name} - [{startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps}] s')
            ax.set(ylabel = 'pdf', ylim = (0, 3))
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            ax.grid(linewidth = 0.2)
            ax.legend()
            _, _, bar_container2 = ax1.hist(theta_red_wind[0], bin_borders_turn,  **default_kwargs_red, label='Red droplets')
            line2, = ax1.plot(x_interval_for_fit_turn, lorentzian_distr(x_interval_for_fit_turn, *red_fit_wind_turn_lorentzian[0, 0]), label='Lorentzian fit')
            line3, = ax1.plot(x_interval_for_fit_turn, normal_distr(x_interval_for_fit_turn, *red_fit_wind_turn_gaussian[0, 0]), label='Gaussian fit')

            ax1.set(ylabel = 'pdf', ylim = (0, 3))
            ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'-$\pi$', r'$-\frac{\pi}{2}$', '$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
            ax1.grid(linewidth = 0.2)
            ax1.legend()
            plt.tight_layout()
            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, prepare_animation(bar_container, bar_container2), nSteps, repeat=True, blit=False)
            ani.save(f'./{res_path}/turning_angles/turn_ang_wind_lorentzian.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()


#############################################################################################################
#                                              VACF ANALYSIS
#############################################################################################################
if velocity_autocovariance_verb:
    print('Velocity autocovariance analysis...')

    blueTrajs, redTrajs = get_trajs(nDrops, red_particle_idx, trajectories, subsample_factor, fps)
    #Global Velocity Autocovariance
    vacf_b, vacf_std_b = ys.vacf(blueTrajs, time_avg=True, lag = maxLagtime)
    vacf_r, vacf_std_r = ys.vacf(redTrajs, time_avg=True, lag = maxLagtime)

    if plot_verb:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        ax.errorbar(np.arange(0, maxLagtime/fps, 1/fps), vacf_b, fmt='o', markersize = 1, color = 'blue', label = 'Blue droplets')
        ax.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b + vacf_std_b, vacf_b - vacf_std_b, alpha=1, edgecolor='#F0FFFF', facecolor='#00FFFF')
        ax.grid(linewidth = 0.2)
        ax.legend()
        ax.set(xlim = (-1, 10), xlabel = 'Lag time [s]', ylabel = r'VACF [$(mm/s)^2$]')
        ax1.errorbar(np.arange(0, maxLagtime/fps, 1/fps), vacf_r, fmt='o', markersize = 1, color = 'red', label = 'Red droplets')
        ax1.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r + vacf_std_r, vacf_r - vacf_std_r, alpha=1, edgecolor='#FF0000', facecolor='#FFCCCB')
        ax1.set(xlim = (-1, 10), xlabel = 'Lag time [s]')
        ax1.grid(linewidth = 0.2)
        ax1.legend()
        plt.suptitle(f'Velocity autocorrelation function - {system_name}')
        plt.tight_layout()
        if save_verb: 
            plt.savefig(f'./{res_path}/velocity_autocovariance/vacf.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/velocity_autocovariance/vacf.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()

    print('Windowed velocity autocovariance analysis...')
    if run_analysis_verb:
        vacf_b_wind, vacf_b_std_wind, vacf_r_wind, vacf_r_std_wind = vacf_windowed(trajectories, nSteps, startFrames, endFrames, red_particle_idx, subsample_factor, fps, maxLagtime)
        vacf_b_wind.to_parquet(f'./{analysis_data_path}/vacf/vacf_b_wind.parquet')
        vacf_b_std_wind.to_parquet(f'./{analysis_data_path}/vacf/vacf_b_std_wind.parquet')
        vacf_r_wind.to_parquet(f'./{analysis_data_path}/vacf/vacf_r_wind.parquet')
        vacf_r_std_wind.to_parquet(f'./{analysis_data_path}/vacf/vacf_r_std_wind.parquet')
        vacf_b_wind = np.array(vacf_b_wind)
        vacf_b_std_wind = np.array(vacf_b_std_wind)
        vacf_r_wind = np.array(vacf_r_wind)
        vacf_r_std_wind = np.array(vacf_r_std_wind)
    else:
        vacf_b_wind     = pd.read_parquet(f'./{analysis_data_path}/vacf/vacf_b_wind.parquet')
        vacf_b_std_wind = pd.read_parquet(f'./{analysis_data_path}/vacf/vacf_b_std_wind.parquet')
        vacf_r_wind     = pd.read_parquet(f'./{analysis_data_path}/vacf/vacf_r_wind.parquet')
        vacf_r_std_wind = pd.read_parquet(f'./{analysis_data_path}/vacf/vacf_r_std_wind.parquet')
        vacf_b_wind = np.array(vacf_b_wind)
        vacf_b_std_wind = np.array(vacf_b_std_wind)
        vacf_r_wind = np.array(vacf_r_wind)
        vacf_r_std_wind = np.array(vacf_r_std_wind)


    if plot_verb:
        # Velocity Variance
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.plot(startFrames/fps + windowLenght/2, vacf_b_wind[:, 0], 'b', label = 'Blue droplets')
        ax.plot(startFrames/fps + windowLenght/2, vacf_r_wind[:, 0], 'r', label = 'Red droplets')
        ax.set_xlim(int(frames[0]/fps), int(frames[-1]/fps))
        ax.set(title = f'Time evolution of velocity variance - {system_name}', ylabel = '$\sigma$', xlabel = 'Window Time [s]')
        ax.grid(linewidth = 0.2)
        ax.legend()
        if save_verb: 
            plt.savefig(f'./{res_path}/velocity_autocovariance/vacf_wind_0.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/velocity_autocovariance/vacf_wind_0.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else: 
            plt.close()
        
        fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))
        ax.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :]/vacf_b_wind[0, 0], 'b', label = 'Blue droplets')
        ax.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :]/vacf_b_wind[0, 0] - vacf_b_std_wind[0, :]/vacf_b_wind[0, 0], 
                        vacf_b_wind[0, :]/vacf_b_wind[0, 0] + vacf_b_std_wind[0, :]/vacf_b_wind[0, 0], color = 'b', alpha = 0.2)
        ax.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :]/vacf_r_wind[0, 0], 'r', label = 'Red droplets')
        ax.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :]/vacf_r_wind[0, 0] - vacf_r_std_wind[0, :]/vacf_r_wind[0, 0],
                        vacf_r_wind[0, :]/vacf_r_wind[0, 0] + vacf_r_std_wind[0, :]/vacf_r_wind[0, 0], color = 'r', alpha = 0.2)
        ax.set(xlabel = 'Lag time [s]', ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[1]/fps} - {endFrames[1]/fps}] s')
        ax1.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0], 'b', label = 'Blue droplets')
        ax1.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0] - vacf_b_std_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0],
                        vacf_b_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0] + vacf_b_std_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0], color = 'b', alpha = 0.2)
        ax1.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0], 'r', label = 'Red droplets')
        ax1.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0] - vacf_r_std_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0],
                        vacf_r_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0] + vacf_r_std_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0], color = 'r', alpha = 0.2)
        ax1.set(xlabel = 'Lag time [s]', title = f'Window [{startFrames[int(nSteps/2)]/fps} - {endFrames[int(nSteps/2)]/fps}] s')
        ax2.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[-1, :]/vacf_b_wind[-1, 0], 'b', label = 'Blue droplets')
        ax2.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[-1, :]/vacf_b_wind[-1, 0] - vacf_b_std_wind[-1, :]/vacf_b_wind[-1, 0],
                        vacf_b_wind[-1, :]/vacf_b_wind[-1, 0] + vacf_b_std_wind[-1, :]/vacf_b_wind[-1, 0], color = 'b', alpha = 0.2)
        ax2.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[-1, :]/vacf_r_wind[-1, 0], 'r', label = 'Red droplets')
        ax2.fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[-1, :]/vacf_r_wind[-1, 0] - vacf_r_std_wind[-1, :]/vacf_r_wind[-1, 0],
                        vacf_r_wind[-1, :]/vacf_r_wind[-1, 0] + vacf_r_std_wind[-1, :]/vacf_r_wind[-1, 0], color = 'r', alpha = 0.2)
        ax2.set(xlabel = 'Lag time [s]', title = f'Window [{startFrames[-1]/fps} - {endFrames[-1]/fps}] s')
        ax.legend()
        ax1.legend()
        ax2.legend()
        ax.grid(linewidth = 0.2)
        ax1.grid(linewidth = 0.2)
        ax2.grid(linewidth = 0.2)
        ax.set(xlim=(-1, 20))
        ax1.set(xlim=(-1, 20))
        ax2.set(xlim=(-1, 20))
        plt.suptitle(f'Velocity autocovariance windowed - {system_name}')
        plt.tight_layout()
        if save_verb:
            plt.savefig(f'./{res_path}/velocity_autocovariance/evolution.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/velocity_autocovariance/evolution.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        axs[0,0].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :], 'b', label = 'Blue droplets')
        axs[0,0].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :] - vacf_b_std_wind[0, :], 
                        vacf_b_wind[0, :] + vacf_b_std_wind[0, :], color = 'b', alpha = 0.2)
        axs[1,0].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :], 'r', label = 'Red droplets')
        axs[1,0].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :] - vacf_r_std_wind[0, :],
                        vacf_r_wind[0, :] + vacf_r_std_wind[0, :], color = 'r', alpha = 0.2)
        axs[0,0].set(ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[1]/fps} - {endFrames[1]/fps}] s')
        axs[1,0].set(xlabel = 'Lag time [s]', ylabel = r'VACF [$(mm/s)^2$]')
        axs[0,1].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[int(nSteps/2), :], 'b', label = 'Blue droplets')
        axs[0,1].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[int(nSteps/2), :] - vacf_b_std_wind[int(nSteps/2), :],
                        vacf_b_wind[int(nSteps/2), :] + vacf_b_std_wind[int(nSteps/2), :], color = 'b', alpha = 0.2)
        axs[1,1].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[int(nSteps/2), :], 'r', label = 'Red droplets')
        axs[1,1].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[int(nSteps/2), :] - vacf_r_std_wind[int(nSteps/2), :],
                        vacf_r_wind[int(nSteps/2), :] + vacf_r_std_wind[int(nSteps/2), :], color = 'r', alpha = 0.2)
        axs[0,1].set(ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[int(nSteps/2)]/fps} - {endFrames[int(nSteps/2)]/fps}] s')
        axs[1,1].set(xlabel = 'Lag time [s]')
        axs[0,2].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[-1, :], 'b', label = 'Blue droplets')
        axs[0,2].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[-1, :] - vacf_b_std_wind[-1, :],
                        vacf_b_wind[-1, :] + vacf_b_std_wind[-1, :], color = 'b', alpha = 0.2)
        axs[1,2].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[-1, :], 'r', label = 'Red droplets')
        axs[1,2].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[-1, :] - vacf_r_std_wind[-1, :],
                        vacf_r_wind[-1, :] + vacf_r_std_wind[-1, :], color = 'r', alpha = 0.2)
        axs[0,2].set(ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[-1]/fps} - {endFrames[-1]/fps}] s')
        axs[1,2].set(xlabel = 'Lag time [s]')
        axs[0,0].legend()
        axs[0,1].legend()
        axs[0,2].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[1,2].legend()
        axs[0,0].grid(linewidth = 0.2)
        axs[0,1].grid(linewidth = 0.2)
        axs[0,2].grid(linewidth = 0.2)
        axs[1,0].grid(linewidth = 0.2)
        axs[1,1].grid(linewidth = 0.2)
        axs[1,2].grid(linewidth = 0.2)
        axs[0,0].set(xlim = (-1, 20))
        axs[0,1].set(xlim = (-1, 20))
        axs[0,2].set(xlim = (-1, 20))
        axs[1,0].set(xlim = (-1, 20))
        axs[1,1].set(xlim = (-1, 20))
        axs[1,2].set(xlim = (-1, 20))
        plt.suptitle(f'Velocity autocovariance windowed - {system_name}')

        plt.tight_layout()
        if save_verb:
            plt.savefig(f'./{res_path}/velocity_autocovariance/evolution_v2.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/velocity_autocovariance/evolution_v2.pdf', bbox_inches='tight')
        if show_verb:
            plt.show()
        else:
            plt.close()

        fig, axs = plt.subplots(2, 3, figsize=(15, 6), sharex=True, sharey=True)
        axs[0,0].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :]/vacf_b_wind[0, 0], 'b', label = 'Blue droplets')
        axs[0,0].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :]/vacf_b_wind[0, 0] - vacf_b_std_wind[0, :]/vacf_b_wind[0, 0], 
                        vacf_b_wind[0, :]/vacf_b_wind[0, 0] + vacf_b_std_wind[0, :]/vacf_b_wind[0, 0], color = 'b', alpha = 0.2)
        axs[1,0].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :]/vacf_r_wind[0, 0], 'r', label = 'Red droplets')
        axs[1,0].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :]/vacf_r_wind[0, 0] - vacf_r_std_wind[0, :]/vacf_r_wind[0, 0],
                        vacf_r_wind[0, :]/vacf_r_wind[0, 0] + vacf_r_std_wind[0, :]/vacf_r_wind[0, 0], color = 'r', alpha = 0.2)
        axs[0,0].set(ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[1]/fps} - {endFrames[1]/fps}] s')
        axs[1,0].set(xlabel = 'Lag time [s]', ylabel = r'VACF [$(mm/s)^2$]')
        axs[0,1].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0], 'b', label = 'Blue droplets')
        axs[0,1].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0] - vacf_b_std_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0],
                        vacf_b_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0] + vacf_b_std_wind[int(nSteps/2), :]/vacf_b_wind[int(nSteps/2), 0], color = 'b', alpha = 0.2)
        axs[1,1].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0], 'r', label = 'Red droplets')
        axs[1,1].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0] - vacf_r_std_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0],
                        vacf_r_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0] + vacf_r_std_wind[int(nSteps/2), :]/vacf_r_wind[int(nSteps/2), 0], color = 'r', alpha = 0.2)
        axs[0,1].set(ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[int(nSteps/2)]/fps} - {endFrames[int(nSteps/2)]/fps}] s')
        axs[1,1].set(xlabel = 'Lag time [s]')
        axs[0,2].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[-1, :]/vacf_b_wind[-1, 0], 'b', label = 'Blue droplets')
        axs[0,2].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[-1, :]/vacf_b_wind[-1, 0] - vacf_b_std_wind[-1, :]/vacf_b_wind[-1, 0],
                        vacf_b_wind[-1, :]/vacf_b_wind[-1, 0] + vacf_b_std_wind[-1, :]/vacf_b_wind[-1, 0], color = 'b', alpha = 0.2)
        axs[1,2].plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[-1, :]/vacf_r_wind[-1, 0], 'r', label = 'Red droplets')
        axs[1,2].fill_between(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[-1, :]/vacf_r_wind[-1, 0] - vacf_r_std_wind[-1, :]/vacf_r_wind[-1, 0],
                        vacf_r_wind[-1, :]/vacf_r_wind[-1, 0] + vacf_r_std_wind[-1, :]/vacf_r_wind[-1, 0], color = 'r', alpha = 0.2)
        axs[0,2].set(ylabel = r'VACF [$(mm/s)^2$]', title = f'Window [{startFrames[-1]/fps} - {endFrames[-1]/fps}] s')
        axs[1,2].set(xlabel = 'Lag time [s]')
        axs[0,0].legend()
        axs[0,1].legend()
        axs[0,2].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[1,2].legend()
        axs[0,0].grid(linewidth = 0.2)
        axs[0,1].grid(linewidth = 0.2)
        axs[0,2].grid(linewidth = 0.2)
        axs[1,0].grid(linewidth = 0.2)
        axs[1,1].grid(linewidth = 0.2)
        axs[1,2].grid(linewidth = 0.2)
        axs[0,0].set(xlim = (-1, 20))
        axs[0,1].set(xlim = (-1, 20))
        axs[0,2].set(xlim = (-1, 20))
        axs[1,0].set(xlim = (-1, 20))
        axs[1,1].set(xlim = (-1, 20))
        axs[1,2].set(xlim = (-1, 20))
        plt.suptitle(f'Velocity autocovariance windowed - {system_name}')
        plt.tight_layout()
        if save_verb:
            plt.savefig(f'./{res_path}/velocity_autocovariance/evolution_v2_n.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/velocity_autocovariance/evolution_v2_n.pdf', bbox_inches='tight')
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
                line.set_ydata(vacf_b_wind[step, :]/vacf_b_wind[step, 0])
                title.set_text(f'Velocity autocorrelation - {system_name} - window [{startFrames[step]/fps + windowLenght/2} - {endFrames[step]/fps}] s')
                line1.set_ydata(vacf_r_wind[step, :]/vacf_r_wind[step, 0])
                return line, line1,
            ax = fig.add_subplot(211)
            title = ax.set_title(f'Velocity autocorrelation - {system_name} - window [{startFrames[0]/fps + windowLenght/2} - {endFrames[0]/fps}] s')
            line, = ax.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_b_wind[0, :]/vacf_b_wind[0, 0], 'b-', label = 'Blue droplets')
            ax.set(ylabel = r'vacf [$(mm/s)^2$]', xlabel = 'lag time $t$ [s]', xlim = (-1, 20), ylim = (-0.5, 1.1))
            ax.grid(linewidth = 0.2)
            ax.legend()
            ax1 = fig.add_subplot(212)
            line1, = ax1.plot(np.arange(0, maxLagtime/fps, 1/fps), vacf_r_wind[0, :]/vacf_r_wind[0, 0], 'r-', label = 'Red droplets')
            ax1.set(ylabel = r'vacf [$(mm/s)^2$]', xlabel = 'lag time $t$ [s]', xlim = (-1, 20), ylim = (-0.5, 1.1))
            ax1.grid(linewidth = 0.2)
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
#                                              RDF ANALYSIS
#############################################################################################################
if rdf_verb:
    print(f'Radial distribution function analysis...')
    n_red = len(red_particle_idx)
    n_blue = nDrops - n_red
    dr = trajectories.r.mean()
    # center of petri dish --> to refine
    r_c   = [(xmax-xmin)+ xmin, (ymax-ymin)+ ymin]
    rDisk = (xmax-xmin)/2
    rList = np.arange(0, 2*rDisk, 1)
    rho_b = n_blue/(np.pi*rDisk**2)
    rho_r = n_red/(np.pi*rDisk**2) 
    rho   = nDrops/(np.pi*rDisk**2)

    if run_analysis_verb:
        rdf = get_rdf(frames, trajectories, red_particle_idx, rList, dr, rho_b, rho_r, n_blue, n_red)
        rdf_b  = rdf[:, 0, :]
        rdf_b_df = pd.DataFrame(rdf_b)
        rdf_b_df.columns = [f'{r}' for r in rList]
        rdf_b_df['frame'] = frames
        rdf_b_df.to_parquet(f'./{analysis_data_path}/rdf/rdf_b.parquet')

        rdf_r  = rdf[:, 1, :]
        rdf_r_df = pd.DataFrame(rdf_r)
        rdf_r_df.columns = [f'{r}' for r in rList]
        rdf_r_df['frame'] = frames
        rdf_r_df.to_parquet(f'./{analysis_data_path}/rdf/rdf_r.parquet')

        rdf_br = rdf[:, 2, :]
        rdf_br_df = pd.DataFrame(rdf_br)
        rdf_br_df.columns = [f'{r}' for r in rList]
        rdf_br_df['frame'] = frames
        rdf_br_df.to_parquet(f'./{analysis_data_path}/rdf/rdf_br.parquet')
    elif not run_analysis_verb:
        try:
            rdf_b = np.array(pd.read_parquet(f'./{analysis_data_path}/rdf/rdf_b.parquet'))
            rdf_r = np.array(pd.read_parquet(f'./{analysis_data_path}/rdf/rdf_r.parquet'))
            rdf_br = np.array(pd.read_parquet(f'./{analysis_data_path}/rdf/rdf_br.parquet'))
        except: 
            raise ValueError('rdf data not found. Run analysis first.')
    else: 
        raise ValueError('run_analysis_verb must be True or False')

    if plot_verb:
        v_min = np.min([rdf_b, rdf_r, rdf_br])
        v_max = np.max([rdf_b, rdf_r, rdf_br])

        timearr = np.linspace(0, rdf_b.shape[0], 5)/fps
        timearr = timearr.astype(int)

        g_plot = rdf_b.T
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        img = ax.imshow(g_plot, vmin = v_min, vmax = v_max)
        ax.set(xticks = np.linspace(0, g_plot.shape[1], 5), yticks = np.linspace(0, g_plot.shape[0], 5))
        ax.set(xticklabels = timearr, yticklabels = (np.linspace(0, 2*rDisk, 5)*pxDimension).astype(int))
        ax.set(xlabel = 'Time [s]', ylabel = 'r [mm]', title = f'RDF blue-blue heatmap - {system_name}')
        fig.colorbar(img, ax=ax)
        ax.set_aspect('auto')
        if save_verb: 
            plt.savefig(f'./{res_path}/radial_distribution_function/rdf_heatmap_b.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/radial_distribution_function/rdf_heatmap_b.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()

        g_plot = rdf_r.T
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        img = ax.imshow(g_plot, vmin = v_min, vmax = v_max)
        ax.set(xticks = np.linspace(0, g_plot.shape[1], 5), yticks = np.linspace(0, g_plot.shape[0], 5))
        ax.set(xticklabels = timearr, yticklabels = (np.linspace(0, 2*rDisk, 5)*pxDimension).astype(int))
        ax.set(xlabel = 'Time [s]', ylabel = 'r [mm]', title = f'RDF r-r heatmap - {system_name}')
        fig.colorbar(img, ax=ax)
        ax.set_aspect('auto')
        if save_verb: 
            plt.savefig(f'./{res_path}/radial_distribution_function/rdf_heatmap_r.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/radial_distribution_function/rdf_heatmap_r.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()


        g_plot = rdf_br.T
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        img = ax.imshow(g_plot, vmin = v_min, vmax = v_max)
        ax.set(xticks = np.linspace(0, g_plot.shape[1], 5), yticks = np.linspace(0, g_plot.shape[0], 5))
        ax.set(xticklabels = timearr, yticklabels = (np.linspace(0, 2*rDisk, 5)*pxDimension).astype(int))
        ax.set(xlabel = 'Time [s]', ylabel = 'r [mm]', title = f'RDF b-r heatmap - {system_name}')
        fig.colorbar(img, ax=ax)
        ax.set_aspect('auto')
        if save_verb: 
            plt.savefig(f'./{res_path}/radial_distribution_function/rdf_heatmap_br.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/radial_distribution_function/rdf_heatmap_br.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()

        
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
            def animate(frame):
                line.set_ydata(rdf_b[frame])  # update the data.
                title.set_text(f'Blue droplets RDF - {system_name} at {int(frame/fps)} s')
                return line, 

            line, = ax.plot(rList*pxDimension, rdf_b[0])
            title = ax.set_title(f'Blue droplets RDF - {system_name} at 0 s')
            ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, rdf_b.shape[0], 10), interval=5, blit=False)
            ax.set(ylim = (-0.5, v_max), ylabel = 'g(r)', xlabel = 'r (mm)')
            ax.grid(linewidth = 0.2)
            fig.canvas.mpl_connect('button_press_event', onClick)
            if save_verb: ani.save(f'./{res_path}/radial_distribution_function/rdf_b.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()

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
            def animate(frame):
                line.set_ydata(rdf_r[frame])  # update the data.
                title.set_text(f'Red droplets RDF - {system_name} at {int(frame/fps)} s') 
                return line, 

            line, = ax.plot(rList*pxDimension, rdf_r[0])
            title = ax.set_title(f'Red droplets RDF - {system_name} at 0 s')
            ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, rdf_r.shape[0], 10), interval=5, blit=False)
            ax.set(ylim = (-0.5, v_max), ylabel = 'g(r)', xlabel = 'r (mm)')
            ax.grid(linewidth = 0.2)
            fig.canvas.mpl_connect('button_press_event', onClick)
            if save_verb: ani.save(f'./{res_path}/radial_distribution_function/rdf_r.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()
            
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
            def animate(frame):
                line.set_ydata(rdf_br[frame])  # update the data.
                title.set_text(f'Blue-red droplets RDF - {system_name} at s {int(frame/fps)}')
                return line, 

            line, = ax.plot(rList*pxDimension, rdf_br[0])
            title = ax.set_title(f'Blue-red droplets RDF - {system_name} at 0 s')
            ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, rdf_br.shape[0], 10), interval=5, blit=False)
            ax.set(ylim = (-0.5, v_max), ylabel = 'g(r)', xlabel = 'r (mm)')
            ax.grid(linewidth = 0.2)
            fig.canvas.mpl_connect('button_press_event', onClick)
            if save_verb: ani.save(f'./{res_path}/radial_distribution_function/rdf_br.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()

    if run_analysis_verb:
        print(print(f'Radial distribution function from center analysis...'))
        rdf_c = get_rdf_center(frames, trajectories, r_c, rList, dr, rho, nDrops)
        rdf_c_df = pd.DataFrame(rdf_c)
        # string columns for parquet filetype
        rdf_c_df.columns = [str(i) for i in rList]
        pd.DataFrame(rdf_c_df).to_parquet(f'./{analysis_data_path}/rdf/rdf_center.parquet')
    if not run_analysis_verb :
        try:
            rdf_c = np.array(pd.read_parquet(f'./{analysis_data_path}/rdf/rdf_center.parquet'))
        except: 
            raise ValueError('rdf data not found. Run analysis verbosely first.')

    if plot_verb:
        timearr = np.linspace(0, rdf_c.shape[0], 10)/fps
        timearr = timearr.astype(int)

        fig, ax = plt.subplots(1, 1, figsize = (10, 4))
        ax.plot(rList, rdf_c.mean(axis=0), label='mean')
        ax.fill_between(rList, rdf_c.mean(axis=0) - rdf_c.std(axis=0), \
                            rdf_c.mean(axis=0) + rdf_c.std(axis=0), alpha=0.3, label='std')
        ax.set(xlabel = 'r [mm]', ylabel = 'g(r)', title = f'RDF from center - {system_name}')
        ax.legend()
        if save_verb: 
            plt.savefig(f'./{res_path}/radial_distribution_function/rdf_center.png', bbox_inches='tight' )
            #plt.savefig(f'./{pdf_res_path}/radial_distribution_function/rdf_center.pdf', bbox_inches='tight' )
        if show_verb: 
            plt.show()
        else:
            plt.close()

        g_plot = rdf_c.T
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        img = ax.imshow(g_plot, vmin = 0, vmax = 10)
        ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
        ax.set(xticklabels = timearr, yticklabels = np.linspace(0, rDisk, 10).astype(int))
        ax.set(xlabel = 'Time [s]', ylabel = 'r [mm]', title = f'rdf from center heatmap - {system_name}')
        fig.colorbar(img, ax=ax)
        ax.set_aspect('auto')
        if save_verb: 
            plt.savefig(f'./{res_path}/radial_distribution_function/rdf_center_heatmap.png', bbox_inches='tight')
            #plt.savefig(f'./{pdf_res_path}/radial_distribution_function/rdf_center_heatmap.pdf', bbox_inches='tight')
        if show_verb: 
            plt.show()
        else:
            plt.close()

        if animated_plot_verb:
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
            title = ax.set_title(f'RDF from center - {system_name} 0 s')
            ax.set_ylim(-1, 15)

            def animate(frame):
                line.set_ydata(rdf_c[frame])  # update the data.
                title.set_text(f'RDF from center - {system_name} {int(frame/fps)} s')
                return line, 

            fig.canvas.mpl_connect('button_press_event', onClick)
            ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, nFrames, fps), interval = 5, blit=False)
            if save_verb: ani.save(f'./{res_path}/radial_distribution_function/rdf_from_center.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
            if show_verb:
                plt.show()
            else:
                plt.close()

#############################################################################################################
#                                              GRAPH ANALYSIS
#############################################################################################################
if graph_verb:
    factors = np.linspace(.5, 3, 20)
    n_conn_components = np.zeros(len(factors))
    for id, factor in enumerate(factors):
        cutoff_distance = factor * 2 * np.mean(trajectories.r.values.reshape(nFrames, nDrops), axis = 1)/pxDimension
        frame = 1500 * fps
        X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
        # create dictionary with positions of the agents
        dicts = {}
        for i in range(len(X)):
            dicts[i] = (X[i, 0], X[i, 1])
        # generate random geometric graph with cutoff distance 2.2 times the mean diameter the droplets have at that frame
        G = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos = dicts, dim = 2)
        n_conn_components[id] = nx.number_connected_components(G)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(factors, n_conn_components)
    ax.set(xlabel = 'Factor', ylabel = 'Number of connected components', title = f'Number of connected components vs cutoff factor')
    ax.grid(linewidth = 0.2)
    ax.axvline(x = 1.5, color = 'r', linestyle = '--')
    #plt.savefig(f'./{pdf_res_path}/connected_components.pdf', bbox_inches='tight')
    plt.show()

    factor = 1.5
    cutoff_distance = factor * 2 * np.mean(trajectories.r.values.reshape(nFrames, nDrops), axis = 1)/pxDimension

    frame = 1500 * fps
    X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
    # create dictionary with positions of the agents
    dicts = {}
    for i in range(len(X)):
        dicts[i] = (X[i, 0], X[i, 1])
    # generate random geometric graph with cutoff distance 2.2 times the mean diameter the droplets have at that frame
    G = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos = dicts, dim = 2)
    node_pos = nx.get_node_attributes(G, 'pos')
    vor = Voronoi(np.asarray(list(node_pos.values())))

    fig, ax = plt.subplots(figsize=(6, 6))
    voronoi_plot_2d(vor, ax = ax, show_vertices = False, line_colors = 'orange', line_width = 2, line_alpha = 0.6, point_size = 2)
    nx.draw(G, pos = node_pos, node_size = 25, node_color = colors, with_labels = False, ax=ax)
    ax.set(xlim = (xmin, xmax), ylim = (ymax, ymin), title = f'Random Geometric Graph at {int(frame/fps)} s -- {system_name}', xlabel='x [px]', ylabel='y [px]')
    if save_verb:
        plt.savefig(f'./{res_path}/graph/random_geometric_graph_{factor}.png', bbox_inches='tight')
        #plt.savefig(f'./{pdf_res_path}/graph/random_geometric_graph_{factor}.pdf', bbox_inches='tight')
    if show_verb:
        plt.show()
    else:
        plt.close()

    if 0:
        clust_id_list = []
        for frame in tqdm(frames):
            X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
            dicts = {}
            for j in range(len(X)):
                dicts[j] = (X[j, 0], X[j, 1])
            temp = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos=dicts, dim=2)
            clust_id = np.ones(len(X), dtype=int)*-1
            for id, c in enumerate(list(nx.connected_components(temp))):
                clust_id[list(c)] = id
            clust_id_list += list(clust_id)
        clustered_trajs = trajectories.copy()
        clustered_trajs['cluster_id'] = clust_id_list
        clustered_trajs['cluster_color'] = [colors[i] for i in clust_id_list]
        clustered_trajs.to_parquet(f'{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet')
    else:
        clustered_trajs = pd.read_parquet(f'{analysis_data_path}/clustering/trajs_simple_connections_factor{factor}.parquet')

    if 0:
        result = []
        for frame in tqdm(frames):
            df = clustered_trajs.loc[clustered_trajs.frame == frame]
            labels = df.cluster_id.values
            unique_labels, counts = np.unique(labels, return_counts=True)

            for j, cluster_id in enumerate(unique_labels[counts>2]):
                # create subgrah with positions of the agents in the cluster
                df_cluster = df.loc[df.cluster_id == cluster_id]
                X = np.array(df_cluster[['x', 'y']])
                dicts = {}
                for i in range(len(X)):
                    dicts[i] = (X[i, 0], X[i, 1])
                temp = nx.random_geometric_graph(len(dicts), cutoff_distance[frame], pos=dicts, dim=2)

                # compute area and eccentricity of the subgraph
                hull = ConvexHull(X)
                area = hull.area
                eccentricity = compute_eccentricity(X)

                # compute degree, degree centrality, betweenness centrality, clustering coefficient, number of cycles,
                # dimension of cycles and first and eigenvalue
                degree = np.mean([val for (_, val) in temp.degree()])
                degree_centrality = np.mean(list(nx.degree_centrality(temp).values()))
                betweenness_centrality = np.mean(list(nx.betweenness_centrality(temp).values()))
                clustering = np.mean(list(nx.clustering(temp).values()))
                cycles = nx.cycle_basis(temp)
                n_cycles = int(len(cycles))
                dim_cycles = [len(cycles[i]) for i in range(int(len(cycles)))]
                if n_cycles == 0:
                    mean_dim_cycles = 0
                else:
                    mean_dim_cycles = np.mean(dim_cycles)

                # laplacian eigenvalues
                lap_eigenvalues = nx.laplacian_spectrum(temp)

                # append to list
                result.append(np.concatenate(([frame], [len(unique_labels[counts>2])], [n_cycles], [degree], [degree_centrality],\
                                        [betweenness_centrality], [clustering], [mean_dim_cycles], [area], [eccentricity],\
                                        [lap_eigenvalues[1]], [lap_eigenvalues[2]])))

        label = np.array(['frame', 'n_clusters', 'n_cycles', 'degree', 'degree_centrality', 'betweenness', 'clustering',\
                'd_cycles', 'area', 'eccentricity', 'first_lapl_eigv', 'second_lapl_eigv'])
        df_graph = pd.DataFrame(result, columns=label)
        display(df_graph)
        df_graph.to_parquet(f'{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet')
    else:
        print(f'Import data with factor {factor}')
        df_graph = pd.read_parquet(f'{analysis_data_path}/graph/graph_analysis_mean_factor{factor}.parquet')
        label = np.array(df_graph.columns)


    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(df_graph.frame.unique()/fps, df_graph.groupby('frame').n_clusters.mean())
    ax.set(xlabel='Time [s]', ylabel='n_clusters', title=f'Number of clusters - {system_name}')
    ax.grid(linewidth=0.2)
    if save_verb:
        plt.savefig(f'./{res_path}/graph/n_clusters.png', bbox_inches='tight')
        #plt.savefig(f'./{pdf_res_path}/graph/n_clusters.pdf', bbox_inches='tight')
    if show_verb:
        plt.show()
    else:
        plt.close()

    x = df_graph.iloc[:, 1:].values # to skip the frame column
    x = StandardScaler().fit_transform(x) # normalizing the features
    x = x/np.std(x)
    print(x.shape, np.mean(x), np.std(x))
    normalized_df = pd.DataFrame(x, columns=label[1:])

    # explained variance ratio with PCA
    pca = PCA(n_components=label.shape[0]-2)
    p_components = pca.fit_transform(x)
    fig, ax = plt.subplots(1, 1, figsize = (6, 3))
    ax.plot(np.arange(1, p_components.shape[1]+1, 1), pca.explained_variance_ratio_.cumsum())
    ax.set(xlabel = 'N of components', ylabel = f'Cumulative explained variance ratio', title = f'PCA scree plot - {system_name}') 
    ax.grid(linewidth = 0.2)
    if save_verb:
        plt.savefig(f'{res_path}/graph/scree_plot.png', bbox_inches='tight')
        #plt.savefig(f'{pdf_res_path}/graph/scree_plot.pdf', bbox_inches='tight')
    if show_verb:
        plt.show()
    else:
        plt.close()

    pca = PCA(n_components=3)
    p_components = pca.fit_transform(x)
    expl_variance_ratio = np.round(pca.explained_variance_ratio_, 2)
    print('Explained variation per principal component: {}'.format(expl_variance_ratio))
    principal_df = pd.DataFrame(data = p_components, columns = ['pc1', 'pc2', 'pc3'])
    principal_df['frame'] = df_graph.frame.values.astype(int)

    fig, axs = plt.subplots(1, 3, figsize = (12, 4))
    scatt = axs[0].scatter(principal_df.pc1, principal_df.pc2, c = principal_df.frame, cmap='viridis', s=10)
    axs[0].set(xlabel = f'PC1 ({expl_variance_ratio[0]})', ylabel = f'PC2 ({expl_variance_ratio[1]})')
    axs[1].scatter(principal_df.pc2, principal_df.pc3, c=principal_df.frame, cmap='viridis', s=10)
    axs[1].set(xlabel = f'PC2 ({expl_variance_ratio[1]})', ylabel = f'PC3 ({expl_variance_ratio[2]})')
    axs[2].scatter(principal_df.pc1, principal_df.pc3, c=principal_df.frame, cmap='viridis', s=10)
    axs[2].set(xlabel = f'PC1 ({expl_variance_ratio[0]})', ylabel = f'PC3 ({expl_variance_ratio[2]})')
    plt.suptitle(f'PCA projections - {system_name}')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    #fig.colorbar(scatt, ax=axs, label='frame', orientation='horizontal', shrink=0.5)
    plt.tight_layout()
    if save_verb:
        plt.savefig(f'{res_path}/graph/pca.png', bbox_inches='tight')
        #plt.savefig(f'{pdf_res_path}/graph/pca.png', bbox_inches='tight', dpi = 500)
    if show_verb:
        plt.show()
    else:
        plt.close()

    loadings = pca.components_

    fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize = (10, 6), sharex=True, sharey=True)
    ax.plot(loadings[0], 'r', label = 'PC1')
    ax1.plot(loadings[1], 'b', label = 'PC2')
    ax2.plot(loadings[2], 'y', label = 'PC3')
    ax.set(ylabel = 'PC1', title = f'PCA components - {system_name}', ylim=(-1, 1))
    ax1.set(ylabel = 'PC2', ylim=(-1,1))
    ax2.set(ylabel = 'PC3', xlabel = 'features', ylim=(-1,1), xticks = np.arange(loadings.shape[1]))
    ax2.set_xticklabels(df_graph.columns[1:], rotation = 45)
    ax.grid(linewidth = 0.5)
    ax1.grid(linewidth = 0.5)
    ax2.grid(linewidth = 0.5)
    fig.tight_layout()
    if save_verb:
        plt.savefig(f'{res_path}/graph/pca_components.png', bbox_inches='tight')
        #plt.savefig(f'{pdf_res_path}/graph/pca_components.pdf', bbox_inches='tight')
    if show_verb:
        plt.show()
    else:
        plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    scatterplot = ax.scatter(principal_df.pc1, principal_df.pc2, principal_df.pc3, c = principal_df.frame, cmap = 'viridis', s = 1, rasterized=True)
    ax.grid(linewidth = 0.2)
    fig.colorbar(scatterplot, ax=ax, shrink=0.6, label = 'frame', orientation = 'horizontal')
    ax.set(xlabel = f'PC1 ({expl_variance_ratio[0]})', ylabel = f'PC2 ({expl_variance_ratio[1]})', zlabel = f'PC3 ({expl_variance_ratio[2]})')
    if 0:
        # save to gif
        import imageio
        images = []
        for n in range(0, 250):
            if n >= 5:
                ax.azim = ax.azim+1.1
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            images.append(image.reshape(1000, 1000, 3)) ## 
        imageio.mimsave(f'{res_path}/graph/pca.gif', images)
    if save_verb:
        plt.savefig(f'{res_path}/graph/pca_3d.png', bbox_inches='tight')
        #plt.savefig(f'{pdf_res_path}/graph/pca_3d.png', bbox_inches='tight', dpi = 500)
    if 1:
        plt.show()
    else:
        plt.close()

#############################################################################################################
#                                              MOTIF ANALYSIS
#############################################################################################################
if motif_verb:
    seed = np.random.seed(0)
    factor = 2.2
    cutoff_distance = factor * 2*np.mean(trajectories.r.values.reshape(nFrames, nDrops), axis = 1)/pxDimension

    frame = 20000
    X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
    g, pos = gt.geometric_graph(X, cutoff_distance[frame])
    motifs, _ = gt.motifs(g, 20)

    motif_results = {}
    for n_vertices in range(3, 10):
        different_motifs = {}
        for frame in tqdm(np.random.choice(frames, 10)):
            X = np.array(trajectories.loc[trajectories.frame == frame, ['x', 'y']])
            g, pos = gt.geometric_graph(X, cutoff_distance[frame])
            motifs, _ = gt.motifs(g, n_vertices)
            
            c = 0
            for graph_found in motifs:
                check = False
                for distinct_graph in list(different_motifs.values()):
                    checkTMP = gt.isomorphism(graph_found, distinct_graph)
                    if checkTMP:
                        check = checkTMP
                        break
                if check:
                    continue
                else:
                    different_motifs[c] = graph_found
                    gt.graph_draw(graph_found, output=f'{res_path}/graph/motif/distinct_motifs/motif_{n_vertices}vertices_{c}.png', vertex_size=10, bg_color=[1, 1, 1, 1])
                    graph_found.save(f'{analysis_data_path}/graph/motif/distinct_motifs/motif_{n_vertices}vertices_{c}.graphml')
                    
                    c += 1
        motif_results[n_vertices] = different_motifs