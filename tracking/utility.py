#%% IMPORTS
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['font.size'] = 8
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
import csv, json
import pims
from PIL import Image, ImageDraw
import cv2

from scipy.optimize import dual_annealing, linear_sum_assignment
from scipy.spatial import distance_matrix
from tqdm import tqdm

@pims.pipeline
def hough_preprocessing(image, x1, y1, x2, y2):    
	"""
	Pims pipeline preprocessing of the image for the HoughCircles function.
	Crops the image to remove the petri dish, converts the image to grayscale and applies a median filter.

	Parameters
	----------
	image: image
		image to preprocess.
	x1, y1, x2, y2: int
		coordinates of the circle to crop.
	Returns
	-------
	npImage: array
		image to be analyzed.
	"""
	
	npImage = np.array(image)
	# Create same size alpha layer with circle
	alpha = Image.new('L', (920, 960), 0)

	draw = ImageDraw.Draw(alpha)
	draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)

	# Convert alpha Image to numpy array
	npAlpha = np.array(alpha)
	npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)*npAlpha #npImage[:, :, 1] * npAlpha
	
	ind = np.where(npImage == 0)
	# npImage[200, 200] color of the border to swap with the black
	npImage[ind] = npImage[200, 200]
	npImage = cv2.medianBlur(npImage, 5)
	return npImage

def hough_loc_frame(correct_n, frame, img, parameters):
	"""
	Hough transform to locate the droplets in the frame.

	Parameters
	----------
	correct_n: int
		number of droplets in the frame.
	frame: int
		frame to be analyzed.
	img: image
		image for the HoughCircles function.
	parameters: dict
		parameters for the HoughCircles function.
	
	Returns
	-------
	if found_circles is not None and the number of droplets found is equal to correct_n:
		x, y position of the droplets, diameter of the droplets, frame, correct_n.
	if found_circles is not None and the number of droplets found is not equal to correct_n:
		x, y position (0,0), diameter of the droplets (0), frame, number of droplets found.
	else:
		x, y position (0,0), diameter of the droplets (0), frame, 0.
	"""
	found_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, **parameters)

	if (found_circles is not None) and (found_circles.shape[1] == correct_n):
		return np.hstack((found_circles[0], (np.ones((correct_n, 1), dtype=int)*frame),\
						  np.ones((correct_n, 1), dtype=int)*correct_n))
	elif (found_circles is not None) and (found_circles.shape[1] != correct_n):
		return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame),\
						  np.ones((correct_n, 1), dtype=int)*found_circles.shape[1]))
	else:
		return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame),\
						  np.zeros((correct_n, 1), dtype=int)))

def hough_feature_location(data_preload, frames, correct_n, params, progress_verb):
	"""
	Locates the droplets in the frames using the HoughCircles function.
	N.B. Uses cv2.HOUGH_GRADIENT_ALT method, check for the parameters in the OpenCV documentation.

	Parameters
	----------
	data_preload: list
		list of images to be analyzed.
	frames: array
		frames to be analyzed.
	correct_n: int
		number of droplets in the frame.
	params: dict
		parameters for the HoughCircles function.
	progress_verb: bool
		if True, shows the progress bar.

	Returns
	-------
	temp: dataframe
		dataframe with the x, y, d, frame, nDroplets of the found circles in the frames.
	err_frames: array
		frames where the number of droplets found is different from the correct_n.
	loss: float
		percentage of frames where the number of droplets found is different from the correct_n.
	"""
	if progress_verb:
		temp = []
		for i in tqdm(range(len(frames))):
			temp.append(hough_loc_frame(correct_n, frames[i], data_preload[i], params))
	else:
		temp = []
		for i in range(len(frames)): 
			temp.append(hough_loc_frame(correct_n, frames[i], data_preload[i], params))

	temp = pd.DataFrame(np.array(temp).reshape(len(frames)*correct_n, 5), columns = ["x", "y", "d", "frame", "nDroplets"])
	err_frames = temp.loc[temp.nDroplets != correct_n].frame.unique().astype(int)
	loss = err_frames.shape[0]/frames.shape[0]
	return temp, err_frames, loss

def optimize_params(x, *args):
	"""
	Optimizes the parameters for the HoughCircles function and saves the set of parameters with the best score.
	N.B. Uses cv2.HOUGH_GRADIENT_ALT method, check for the parameters in the OpenCV documentation.

	Parameters
	----------
	x: array
		parameters for the HoughCircles function.
	args: tuple
		data_preload, frames, correct_n, traj_part.
	
	Returns
	-------
	loss: float
		percentage of frames where the number of droplets found is different from the correct_n.
	"""
	data_preload, frames, correct_n, traj_part = args
	params = {"dp":x[0], "minDist":int(x[1]), "param1":x[2], "param2":x[3], "minRadius":int(x[4]), "maxRadius":int(x[5])}
	_, _, loss = hough_feature_location(data_preload, frames, correct_n, params, False)

	# Save the current best score and set of parameters to a CSV file
	a = [loss, x[0], int(x[1]), x[2], x[3], int(x[4]), int(x[5])]
	with open(f"./results/tracking_data/hough/{traj_part}_optimization.csv", mode = 'a', newline='') as file:
	   writer = csv.writer(file)
	   writer.writerow(a)
	print(a)
	return loss

def plot_opt_results(opt_result_df, slot2):
	"""
	Plots the loss and the parameter slot2.
	
	Parameters
	----------
	opt_result_df: dataframe
		dataframe with the loss and the parameters.
	slot2: str
		parameter to be plotted.
	
	Returns
	-------
	fig: figure
		figure with the loss and the parameter slot2.
	"""
	
	fig, ax = plt.subplots(1, 1)
	ax.plot(np.arange(0, len(opt_result_df.loss), 1), opt_result_df.loss, 'b-')
	ax.set_ylabel("loss", color = 'b') 
	ax1 = ax.twinx() 
	ax1.plot(np.arange(0, len(opt_result_df[slot2]), 1), opt_result_df[slot2], 'r.')
	ax1.set_ylabel(slot2, color='r')
	ax.grid()
	plt.show()
	return fig