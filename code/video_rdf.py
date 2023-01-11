
import matplotlib.pyplot as plt
import matplotlib.animation

plt.rcParams['figure.figsize'] = [10, 4]
writervideo = matplotlib.animation.FFMpegWriter(fps=30)

import numpy as np
import pandas as pd

dr = 5
rDisk = 822/2
rList = np.arange(0, 2*rDisk, 1)

g2 = np.array(pd.read_csv("/Users/matteoscandola/thesis/data/g2.csv"))

fig, ax = plt.subplots()
ax.set_ylim(-1, 30)
anim_running = True

def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

line, = ax.plot(rList, g2[0])
title = ax.set_title('Test, time=0')

def animate(frame):
    line.set_ydata(g2[frame])  # update the data.
    title.set_text('Time={} s'.format(np.round(frame/10)))
    #ax.set_ylim(-1, max(g2[frame]) + 4)
    return line, 

fig.canvas.mpl_connect('button_press_event', onClick)
ani = matplotlib.animation.FuncAnimation(fig, animate, range(0, 30000, 30), interval=20, blit=False)
ani.save(f'/Users/matteoscandola/thesis/Results/first_analysis/g2.mp4', writer=writervideo)
#plt.show()
plt.close()
