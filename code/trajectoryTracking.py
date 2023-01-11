import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
writervideo = matplotlib.animation.FFMpegWriter(fps=10)
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd

import pims
import trackpy as tp
from PIL import Image, ImageDraw

import numba
import timeit
import random

@pims.pipeline
def crop(image, x1, y1, x2, y2):
    npImage = np.array(image)
    
    # Create same size alpha layer with circle
    alpha = Image.new('L', (720, 720), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)

    # Convert alpha Image to numpy arrayf
    npAlpha = np.array(alpha)
    npImage = npImage[:,:,1] * npAlpha
    
    ind = np.where(npImage == 0)
    # npImage[150, 150] color of the border to swap with the black
    npImage[ind] = npImage[150, 150]
    return npImage

frames = crop(pims.open('/Users/matteoscandola/thesisData/Movie32.mp4'), 30, 30, 650, 650)
dropSize = 29 
sep = 16

startFrame = 32270
nFrames = startFrame + 100
minMass = 2000

print("daje")
f = tp.batch(frames[startFrame:nFrames], dropSize, minmass = minMass, separation = sep, topn = 59, engine = 'numba')
print("daje2")
t = tp.link_df(f, 40, memory = 5)

a = t.loc[t.frame == 47].sort_values('particle').particle.values
b = t.loc[t.frame == 48].sort_values('particle').particle.values
print(a)
print(b)
ind = np.where(np.in1d(a, b)==False)
lost_particles = a[ind]
print(lost_particles)

n = 100
random.seed(5)
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
for i in range(max(t.particle)+1-n):
    colors.append("#00FFFF")


c = []
for p in t.particle:
    c.append(colors[p])
t["color"] = c


fig, (ax, ax1) = plt.subplots(1, 2)

df = t.loc[t['frame'] == 47, ["x","y","color"]]
ax.scatter(df.x, df.y, s=50, ec = "w", facecolor = df.color)
ax.imshow(frames[47])


df1 = t.loc[t['frame'] == 48, ["x","y","color"]]
ax1.scatter(df1.x, df1.y, s=50, ec = "w", facecolor = df1.color)
ax1.imshow(frames[48])

plt.show()







