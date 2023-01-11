import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.animation
plt.rcParams['figure.figsize'] = (9, 6)
writervideo = matplotlib.animation.FFMpegWriter(fps=60)
import pims

# Import data
data = pims.open('/Volumes/ExtremeSSD/UNI/ThesisData/data_video/movie.mp4')
rawTrajs = pd.read_csv("/Volumes/ExtremeSSD/UNI/ThesisData/data/Processed_data2.csv")
nDrops = len(rawTrajs.loc[rawTrajs.frame==0])
nFrames = max(rawTrajs.frame)
print(f"nDrops:{nDrops}")
print(f"nFrames:{nFrames}")

fig = plt.figure()
anim_running = True

def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

def update_graph(frame):
    df = rawTrajs.loc[(rawTrajs.frame == frame) , ["x","y","color"]]
    graph.set_offsets(df)
    graph.set_edgecolor(df.color)
    graph2.set_data(data[frame])
    time = frame / 10
    title.set_text('Time = {} s'.format(time))
    return graph

ax = fig.add_subplot(111)
title = ax.set_title('Time=0')
df = rawTrajs.loc[(rawTrajs.frame == 0), ["x","y","color"]]

graph = ax.scatter(df.x, df.y, facecolors = 'none', edgecolors = df.color, s = 300)

graph2 = ax.imshow(data[0])

fig.canvas.mpl_connect('button_press_event', onClick)
ani = matplotlib.animation.FuncAnimation(fig, update_graph, 30000, interval = 20, blit=False)
ani.save(f'/Volumes/ExtremeSSD/UNI/thesis/Results/result_videos/try.mp4', writer=writervideo)
plt.close()

'''

from PIL import Image, ImageDraw

@pims.pipeline
def crop2(image, x1, y1, x2, y2):
    npImage = np.array(image)
    
    # Create same size alpha layer with circle
    alpha = Image.new('L', (920, 960), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)

    # Convert alpha Image to numpy arrayf
    npAlpha = np.array(alpha)
    npImage = npImage[:,:,1] * npAlpha
    
    ind = np.where(npImage == 0)
    # npImage[150, 150] color of the border to swap with the black
    npImage[ind] = npImage[200, 200]
    return npImage

fig = plt.figure()
anim_running = True

def onClick(event):
    global anim_running
    if anim_running:
        ani.event_source.stop()
        anim_running = False
    else:
        ani.event_source.start()
        anim_running = True

def update_graph(frame):
    df = t.loc[t.frame == frame, ["x","y","color"]]
    graph.set_offsets(df)
    graph.set_facecolor(df.color)
    graph2.set_data(frames[frame])
    title.set_text('frame={}'.format(frame))
    return graph

ax = fig.add_subplot(111)
title = ax.set_title('Ao')
df = t.loc[t['frame'] == 0, ["x","y","color"]]

graph = ax.scatter(df.x, df.y, s=50, ec = "w", facecolor = df.color)

graph2 = ax.imshow(frames[0])

fig.canvas.mpl_connect('button_press_event', onClick)
ani = matplotlib.animation.FuncAnimation(fig, update_graph, 80600, interval = 50, blit=False)
ani.save(f'/Users/matteoscandola/thesisData/videos/full_video/try.mp4', writer=writervideo)
plt.close()
'''
