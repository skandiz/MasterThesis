import numpy as np
import pandas as pd
import pims
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import joblib


@pims.pipeline
def crop(image, x1, y1, x2, y2):    
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
def loc_frame(correct_n, frame, img):
	temp = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, **parameters)
	if temp.shape[1] == correct_n:
		return np.hstack((temp[0], np.ones((correct_n, 1), dtype=int)*frame)) #temp[0]
	else: 
		return np.hstack((np.zeros((correct_n, 3)), np.ones((correct_n, 1), dtype=int)*frame))#np.zeros((correct_n, 3))


data = crop(pims.open('./data/movie.mp4'), 40, 55, 895, 910)
parameters = {"dp": 1.5, "minDist": 15, "param1": 100, "param2": 0.8, "minRadius": 15, "maxRadius": 25}

nFrames = len(data)
merge_frame = 32269

if 0:
	print("Pre merge detection")
	parallel = joblib.Parallel(n_jobs = -2)
	temp = parallel(
		loc_frame(50, frame, data[frame])
		for frame in tqdm( range(merge_frame) )
	)
	pre_merge_droplets = np.array(temp).reshape(merge_frame*50, 4)
	#pd.DataFrame(pre_merge_droplets, columns = ["x", "y", "d", "frame"]).to_parquet("pre_merge_droplets.parquet")
	print(pd.read_parquet("./pre_merge_droplets.parquet"))
else:
	print("Post merge detection")
	parallel = joblib.Parallel(n_jobs = -2)
	temp = parallel(
		loc_frame(49, frame, data[frame])
		for frame in tqdm( range(merge_frame, nFrames) )
	)
	print(np.array(temp).shape)
	post_merge_droplets = np.array(temp).reshape((nFrames-merge_frame)*49, 4)
	pd.DataFrame(post_merge_droplets, columns = ["x", "y", "d", "frame"]).to_parquet("post_merge_droplets.parquet")
	print(pd.read_parquet("./post_merge_droplets.parquet"))