import numpy as np
import pandas as pd
import pims
import trackpy as tp
from PIL import Image, ImageDraw
import cv2
import timeit
import random
import trackpy.diag
tp.quiet()



@pims.pipeline
def crop(image, x1, y1, x2, y2):    
    npImage = np.array(image)
    # Create same size alpha layer with circle
    #alpha = Image.new('L', (920, 960), 0)
    alpha = Image.new('L', (920, 960), 0)

    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)

    # Convert alpha Image to numpy arrayf
    npAlpha = np.array(alpha)
    npImage = npImage[:, :, 1] * npAlpha
    
    ind = np.where(npImage == 0)
    # npImage[200, 200] color of the border to swap with the black
    npImage[ind] = npImage[200, 200]
    return npImage


frames = crop(pims.open('/Volumes/ExtremeSSD/UNI/thesis/ThesisData/data_video/movie.mp4'), 55, 55, 880, 880)



dropSize = 31 #yranges of a drop 335-307 #xranges of a drop 118-90 --> 28 but needs to be odd so 29
minMass = 2000
sep = 16
nDrops = 49


start = timeit.default_timer()
startFrame = 32270
nFrames = 100 #80700

f = tp.batch(frames[startFrame:startFrame + nFrames], dropSize, 
				minmass = minMass, separation = sep, topn = nDrops,
				 engine = 'numba')
end = timeit.default_timer()
print(end-start)

#documentation --> tp.link?
t = tp.link_df(f, 150, memory = 2, link_strategy = 'hybrid', neighbor_strategy = 'KDTree', adaptive_stop = 1)

num = np.zeros(nFrames)
for i in range(nFrames):
    num[i] = len(t.loc[t['frame'] == i])
    
idx = np.where(num != nDrops)[0]
if len(idx) != 0:
    delta = np.zeros(len(idx)-1)
    for i in range(len(idx)-1):
        delta[i] = idx[i+1]-idx[i]

t.to_csv(f'/Users/matteoscandola/thesisData/Processed_data_second_part.csv', index=False)




