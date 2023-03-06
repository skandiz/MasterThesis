import matplotlib.pyplot as plt
import matplotlib.animation

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree
import joblib
from tqdm import tqdm

run = False
plot = True

###############################################################
#                          import data                        #
###############################################################

rawTrajs = pd.read_csv("../data/csv/Processed_data2.csv")
nDrops = len(rawTrajs.loc[rawTrajs.frame==0])
nFrames = max(rawTrajs.frame) + 1
print(f"nDrops:{nDrops}")
print(f"nFrames:{nFrames} --> {nFrames/10:.2f} s")

###############################################################
#                  rdf parameters & function                  #
###############################################################
frames = nFrames

dr = 30
rDisk = 822/2
rList = np.arange(0, rDisk, 1/4)
rho = nDrops/(np.pi*rDisk**2) # nDrops -1 !

COORDS = np.array(rawTrajs.loc[:,["x","y"]])

#centre of the image --> to refine
r_c = [470, 490]

@joblib.delayed
def rdf_from_centre(frame, COORDS, r_c, rList, dr, rho):
    coords = COORDS[frame*nDrops:(frame+1)*nDrops,:]
    kd = KDTree(coords)

    avg_n = np.zeros(len(rList))
    for i, r in enumerate(rList):
        # find all the points within r+dr
        a = kd.query_ball_point(r_c, r + 20)
        n1 = len(a) 
        # find all the points within r+dr
        b = kd.query_ball_point(r_c, r)
        n2 = len(b)
        
        avg_n[i] = n1 - n2

    g2 = avg_n/(np.pi*(dr**2 + 2*rList*dr)*rho)
    return g2



if run:
    parallel = joblib.Parallel(n_jobs = -2)
    trial = parallel(
        rdf_from_centre(frame, COORDS, r_c, rList, dr, rho)
        for frame in tqdm( range(frames) )
    )
    trial = np.array(trial)
    
    trial_df = pd.DataFrame(trial)
    trial_df.columns = [f'{i}' for i in rList]
    trial_df.round(3)
    trial_df.to_parquet("../data/csv/ref_centre.parquet", engine = 'pyarrow')
else:
    try:
        trial = np.array(pd.read_parquet("../data/csv/ref_centre.parquet"))
    except:
        print('Run analysis first')


if plot:
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

    line, = ax.plot(rList, trial[0])
    title = ax.set_title('Test, time=0')
    ax.set_ylim(-1, 10)

    def animate(frame):
        line.set_ydata(trial[frame])  # update the data.
        title.set_text('Test, time={}'.format(frame))
        return line, 

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, frames), interval=5, blit=False)
    #ani.save('../results/video/rdf_from_centre.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
    plt.show()

