import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree
from scipy.signal import savgol_filter
import joblib
from tqdm import tqdm
from utility import get_smooth_trajs
run_analysis_verb = True
show_verb = False
animated_plot_verb = True


rawTrajs = pd.read_parquet("../tracking/results/parquet/pre_merge_tracking.parquet")
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
smoothTrajs = get_smooth_trajs(rawTrajs, nDrops, 30, 2)

## REGULAR RDF
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


def get_rdf(run_analysis_verb, nFrames, trajs, rList, dr, rho):
    print(trajs)
    if trajs == "raw":
        trajectories = rawTrajs
    elif trajs == "smooth":
        trajectories = smoothTrajs
    else:
        raise ValueError("trajs must be 'raw' or 'smooth'")
     
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
        rdf_df.to_parquet(f"./analysis_data/rdf_{trajs}.parquet")

    if not run_analysis_verb :
        try:
            rdf = np.array(pd.read_parquet(f"./analysis_data/rdf/rdf_{trajs}.parquet"))
        except: 
            raise ValueError("rdf data not found. Run analysis verbosely first.")
    return rdf

dr = 5
rDisk = 822/2
rList = np.arange(0, 2*rDisk, 1)
rho = nDrops/(np.pi*rDisk**2) # nDrops - 1 !???

print("RDF - Raw Trajectories")
rdf_raw = get_rdf(run_analysis_verb, nFrames, "raw", rList, dr, rho)
print("RDF - Smooth Trajectories")
rdf_smooth = get_rdf(run_analysis_verb, nFrames, "smooth", rList, dr, rho)

if animated_plot_verb:
    # Animated plot for raw trajs results
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


    line, = ax.plot(rList, rdf_raw[0])
    title = ax.set_title('RDF - Raw Trajectories 0 s')
    def animate(frame):
        line.set_ydata(rdf_raw[frame])  # update the data.
        title.set_text('RDF - Raw Trajectories {} s'.format(frame))
        return line, 

    ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, rdf_raw.shape[0], 10), interval=5, blit=False)
    ax.set(ylim = (-0.5, 30), ylabel = "g(r)", xlabel = "r (px)", title = "Radial Distribution Function from center")
    fig.canvas.mpl_connect('button_press_event', onClick)
    if save_verb: ani.save('./results/radial_distribution_function/rdf_raw.mp4', fps=120, extra_args=['-vcodec', 'libx264'])
    if show_verb:
        plt.show()
    else:
        plt.close()

    # Animated plot for smooth trajs results
    fig, ax = plt.subplots(1,1, figsize=(10, 4))
    anim_running = True
    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True


    line, = ax.plot(rList, rdf_smooth[0])
    title = ax.set_title('RDF - Smooth Trajectories - 0 s')
    def animate(frame):
        line.set_ydata(rdf_smooth[frame])  # update the data.
        title.set_text('RDF - Smooth Trajectories - {} s'.format(frame))
        return line, 

    ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, rdf_smooth.shape[0], 10), interval=5, blit=False)
    ax.set(ylim = (-0.5, 30), ylabel = "g(r)", xlabel = "r (px)", title = "Radial Distribution Function from center")
    fig.canvas.mpl_connect('button_press_event', onClick)
    if save_verb: ani.save('./results/radial_distribution_function/rdf_smooth.mp4', fps=120, extra_args=['-vcodec', 'libx264'])
    if show_verb:
        plt.show()
    else:
        plt.close()


g_plot = rdf_raw.T
timearr = np.linspace(0, rdf_raw.shape[0], 10)/10
timearr = timearr.astype(int)

fig, ax = plt.subplots(1, 1, figsize=(8,6))
img = ax.imshow(np.log(1 + g_plot))
ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
ax.set(xticklabels = timearr, yticklabels = np.linspace(0, 2*rDisk, 10).astype(int))
ax.set(xlabel = "Time [s]", ylabel = "r [px]", title="$Log(1 + g_2)$ heatmap ")
fig.colorbar(img, ax=ax)
ax.set_aspect(30)
if save_verb: plt.savefig("./results/radial_distribution_function/rdf_heatmap_raw.png", )
if show_verb: 
    plt.show()
else:
    plt.close()


g_plot = rdf_smooth.T
timearr = np.linspace(0, rdf_smooth.shape[0], 10)/10
timearr = timearr.astype(int)

fig, ax = plt.subplots(1, 1, figsize=(8,6))
img = ax.imshow(np.log(1 + g_plot))
ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
ax.set(xticklabels = timearr, yticklabels = np.linspace(0, 2*rDisk, 10).astype(int))
ax.set(xlabel = "Time [s]", ylabel = "r [px]", title="$Log(1 + g_2)$ heatmap ")
fig.colorbar(img, ax=ax)
ax.set_aspect(30)
if save_verb: plt.savefig("./results/radial_distribution_function/rdf_heatmap_smooth.png", )
if show_verb: 
    plt.show()
else:
    plt.close()


## RDF FROM CENTER
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

def get_rdf_center(run_analysis_verb, nFrames, trajs, r_c, rList, dr, rho):
    if trajs == "raw":
        trajectories = rawTrajs
    elif trajs == "smooth":
        trajectories = smoothTrajs
    else:
        raise ValueError("trajs must be 'raw' or 'smooth'")
    
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
        pd.DataFrame(rdf_c_df).to_parquet(f"./analysis_data/rdf/rdf_center_{trajs}.parquet")
        
    if not run_analysis_verb :
        try: 
            rdf_c = np.array(pd.read_parquet(f"./analysis_data/rdf/rdf_center_{trajs}.parquet"))
        except: 
            raise ValueError("rdf data not found. Run analysis verbosely first.")
    return rdf_c


dr = 5
rDisk = 822/2
rList = np.arange(0, rDisk, 1)
rho = nDrops/(np.pi*rDisk**2) # nDrops -1 !
r_c = [470, 490] #center of the image --> to refine

print("RDF from center - Raw Trajectories")
rdf_c_raw = get_rdf_center(run_analysis_verb, nFrames, "raw", r_c, rList, dr, rho)
print("RDF from center - Smooth Trajectories")
rdf_c_smooth = get_rdf_center(run_analysis_verb, nFrames, "smooth", r_c, rList, dr, rho)

fig, ax = plt.subplots(1, 1, figsize = (10, 4))
ax.plot(rList, rdf_c_raw.mean(axis=0), label="mean")
ax.fill_between(rList, rdf_c_raw.mean(axis=0) - rdf_c_raw.std(axis=0), \
                       rdf_c_raw.mean(axis=0) + rdf_c_raw.std(axis=0), alpha=0.3, label="std")
ax.set(xlabel = "r [px]", ylabel = "g(r)", title = "RDF from center - Raw Trajectories")
ax.legend()
if save_verb: plt.savefig("./results/radial_distribution_function/rdf_center_raw.png", )
if show_verb: 
    plt.show()
else:
    plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 4))
ax.plot(rList, rdf_c_smooth.mean(axis=0), label="mean")
ax.fill_between(rList, rdf_c_smooth.mean(axis=0) - rdf_c_smooth.std(axis=0), \
                       rdf_c_smooth.mean(axis=0) + rdf_c_smooth.std(axis=0), alpha=0.3, label="std")
ax.set(xlabel = "r [px]", ylabel = "g(r)", title = "RDF from center - Smooth Trajectories")
ax.legend()
if save_verb: plt.savefig("./results/radial_distribution_function/rdf_center_smooth.png", )
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

    line, = ax.plot(rList, rdf_c_raw[0])
    title = ax.set_title('RDF from center - Raw Trajectories 0 s')
    ax.set_ylim(-1, 10)

    def animate(frame):
        line.set_ydata(rdf_c_raw[frame])  # update the data.
        title.set_text('RDF from center - Raw Trajectories {} s'.format(frame))
        return line, 

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, nFrames, 10), interval=5, blit=False)
    if save_verb: ani.save('./results/radial_distribution_function/rdf_from_center_raw.mp4', fps=120, extra_args=['-vcodec', 'libx264'])
    if show_verb:
        plt.show()
    else:
        plt.close()


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

    line, = ax.plot(rList, rdf_c_smooth[0])
    title = ax.set_title('RDF from center - Smooth Trajectories 0 s')
    ax.set_ylim(-1, 10)

    def animate(frame):
        line.set_ydata(rdf_c_smooth[frame])  # update the data.
        title.set_text('RDF from center - Smooth Trajectories {} s'.format(frame))
        return line, 

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, nFrames, 10), interval=5, blit=False)
    if save_verb: ani.save('./results/radial_distribution_function/rdf_from_center_smooth.mp4', fps=120, extra_args=['-vcodec', 'libx264'])
    if show_verb:
        plt.show()
    else:
        plt.close()
g_plot = rdf_c_raw.T
timearr = np.linspace(0, rdf_raw.shape[0], 10)/10
timearr = timearr.astype(int)

fig, ax = plt.subplots(1, 1, figsize=(8,6))
img = ax.imshow(np.log(1 + g_plot))
ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
ax.set(xticklabels = timearr, yticklabels = np.linspace(0, 2*rDisk, 10).astype(int))
ax.set(xlabel = "Time [s]", ylabel = "r [px]", title="$Log(1 + g_2)$ heatmap - Raw Trajectories")
fig.colorbar(img, ax=ax)
ax.set_aspect(30)
if save_verb: plt.savefig("./results/radial_distribution_function/rdf_center_heatmap_raw.png", )
if show_verb: 
    plt.show()
else:
    plt.close()


g_plot = rdf_c_smooth.T
timearr = np.linspace(0, rdf_smooth.shape[0], 10)/10
timearr = timearr.astype(int)

fig, ax = plt.subplots(1, 1, figsize=(8,6))
img = ax.imshow(np.log(1 + g_plot))
ax.set(xticks = np.linspace(0, g_plot.shape[1], 10), yticks = np.linspace(0, g_plot.shape[0], 10))
ax.set(xticklabels = timearr, yticklabels = np.linspace(0, 2*rDisk, 10).astype(int))
ax.set(xlabel = "Time [s]", ylabel = "r [px]", title="$Log(1 + g_2)$ heatmap - Smooth Trajectories")
fig.colorbar(img, ax=ax)
ax.set_aspect(30)
if save_verb: plt.savefig("./results/radial_distribution_function/rdf_center_heatmap_smooth.png", )
if show_verb: 
    plt.show()
else:
    plt.close()