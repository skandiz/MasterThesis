import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 4]
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
import pims
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import joblib


@pims.pipeline
def crop2(image, x1, y1, x2, y2):    
	#image = cv2.GaussianBlur(image, ksize = [7,7], sigmaX = 1.5, sigmaY = 1.5)
	npImage = np.array(image)
	# Create same size alpha layer with circle
	alpha = Image.new('L', (920, 960), 0)

	draw = ImageDraw.Draw(alpha)
	draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)

	# Convert alpha Image to numpy arrayf
	npAlpha = np.array(alpha)
	npImage = npImage[:, :, 1] * npAlpha
	
	ind = np.where(npImage == 0)
	# npImage[200, 200] color of the border to swap with the black
	npImage[ind] = npImage[200, 200]
	npImage = cv2.medianBlur(npImage, 5)
	return npImage

@joblib.delayed
def loc_frame(correct_n, frame, img, parameters):
	temp = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, **parameters)
	if temp.shape[1] == correct_n:
		return np.hstack((temp[0], (np.ones((correct_n, 1), dtype=int)*frame), np.ones((correct_n, 1), dtype=int)*temp.shape[1]))
	else: 
		return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame), np.ones((correct_n, 1), dtype=int)*temp.shape[1]))

# load data
data = crop2(pims.open('./data/movie.mp4'), 40, 55, 895, 910)
merge_frame = 32269

# pre merge test
if 1:
	parameters_pre_merge = {"dp": 1.5, "minDist": 15, "param1": 100, "param2": 0.8, "minRadius": 15, "maxRadius": 25}
	startFrame = 0
	endFrame = 5000 #merge_frame
	frames = np.arange(startFrame, endFrame, 1)
	n = len(frames)
	parallel = joblib.Parallel(n_jobs = -1)
	temp = parallel(
	    loc_frame(50, frame, data[frame], parameters_pre_merge)
	    for frame in tqdm( frames )
	) 
	temp = pd.DataFrame(np.array(temp).reshape(n*50, 5), columns = ["x", "y", "d", "frame", "nDroplets"])
	temp = temp.replace(0, np.nan)
	temp.loc[:50, "frame"] = 0
	print(temp)

	err_frames = np.where(temp.groupby("frame").mean().x.isna())[0]
	print(f"Errors after merging:  {len(err_frames)} --> {np.round(100*len(err_frames)/n, 2)} %")

	fig, (ax, ax1) = plt.subplots(2, 1, figsize = (10, 4))
	ax.plot(temp.groupby("frame").mean().nDroplets, "o", ms = 1)
	ax1.plot(temp.d, "o", ms = 1)
	ax.set_xlabel("Frame")
	ax.set_ylabel("Number of droplets")
	ax.set_title("Number of droplets detected per frame")
	plt.show()

# Post merge test
if 0:
	parameters_post_merge = {"dp": 1.5, "minDist": 10, "param1": 100, "param2": 0.6, "minRadius": 10, "maxRadius": 35}
	startFrame = merge_frame
	endFrame = 5000 #len(data)
	frames = np.arange(startFrame, endFrame, 1)
	n = len(frames)

	parallel = joblib.Parallel(n_jobs = -1)
	temp = parallel(
	    loc_frame(49, frame, data[frame], parameters_post_merge)
	    for frame in tqdm( frames )
	) 
	temp = pd.DataFrame(np.array(temp).reshape(n*49, 5), columns = ["x", "y", "d", "frame", "nDroplets"])
	temp = temp.replace(0, np.nan)
	temp.loc[:50, "frame"] = 0
	print(temp)

	err_frames = np.where(temp.groupby("frame").mean().x.isna())[0]
	print(f"Errors after merging:  {len(err_frames)} --> {np.round(100*len(err_frames)/n, 2)} %")

	fig, (ax, ax1) = plt.subplots(2, 1, figsize = (10, 4))
	ax.plot(temp.nDroplets, "o", ms = 1)
	ax1.plot(temp.d, "o", ms = 1)
	ax.set_xlabel("Frame")
	ax.set_ylabel("Number of droplets")
	ax.set_title("Number of droplets detected per frame")
	plt.show()


