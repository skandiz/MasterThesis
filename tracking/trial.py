import numpy as np
import pandas as pd
import csv, json

import pims
import trackpy as tp
tp.quiet()
from PIL import Image, ImageDraw
import cv2

from scipy.optimize import dual_annealing, minimize
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
import joblib

from scipy.spatial import distance_matrix
from scipy.ndimage import uniform_filter1d

import random

run_analysis_verb = False
show_verb = True
save_verb = True

@pims.pipeline
def hough_preprocessing(image, x1, y1, x2, y2):    
    #image = cv2.GaussianBlur(image, ksize = [7,7], sigmaX = 1.5, sigmaY = 1.5)
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

# SETUP
preload_load_data = True # takes 20 min
merge_frame = 32269
data = hough_preprocessing(pims.open('./data/movie.mp4'), 40, 55, 895, 910)
if preload_load_data: 
    data_preload = list(data[:merge_frame])

startFrame = 0
endFrame = merge_frame
frames = np.arange(startFrame, endFrame, 1)
frames_opt = np.sort(random.sample(list(frames), 5000))
correct_n = 50
default_parameters = {"dp": 1.5, "minDist": 15, "param1": 100, "param2": 0.8, "minRadius": 15, "maxRadius": 25}


def loc_frame(correct_n, frame, img, parameters):
    found_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, **parameters)
    if (found_circles is not None) and (found_circles.shape[1] == correct_n):
        return np.hstack((found_circles[0], (np.ones((correct_n, 1), dtype=int)*frame), np.ones((correct_n, 1), dtype=int)*correct_n))
    elif (found_circles is not None) and (found_circles.shape[1] != correct_n):
        return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame), np.ones((correct_n, 1), dtype=int)*found_circles.shape[1]))
    else:
        return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame), np.zeros((correct_n, 1), dtype=int)))

@joblib.delayed
def loc_frame_parallel(correct_n, frame, img, parameters):
    found_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, **parameters)
    if (found_circles is not None) and (found_circles.shape[1] == correct_n):
        return np.hstack((found_circles[0], (np.ones((correct_n, 1), dtype=int)*frame), np.ones((correct_n, 1), dtype=int)*correct_n))
    elif (found_circles is not None) and (found_circles.shape[1] != correct_n):
        return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame), np.ones((correct_n, 1), dtype=int)*found_circles.shape[1]))
    else:
        return np.hstack((np.zeros((correct_n, 3)), (np.ones((correct_n, 1), dtype=int)*frame), np.zeros((correct_n, 1), dtype=int)))

def hough_feature_location(data_preload, frames, correct_n, params, parallel_verb):
    if parallel_verb:
        parallel = joblib.Parallel(n_jobs = -1)
        temp = parallel(
            loc_frame_parallel(correct_n, frames[i], data_preload[i], params)
            for i in range(len(frames)) #tqdm(range(len(frames)) )
        )
        #print(np.array(temp).shape)
    else:
        temp = []
        for i in range(len(frames)):#tqdm(range(len(frames))):
            temp.append(loc_frame(correct_n, frames[i], data_preload[i], params))
        #print(np.array(temp).shape)
    temp = pd.DataFrame(np.array(temp).reshape(len(frames)*correct_n, 5), columns = ["x", "y", "d", "frame", "nDroplets"])
    err_frames = temp.loc[temp.nDroplets != correct_n].frame.unique().astype(int)
    loss = err_frames.shape[0]/frames.shape[0]
    return temp, err_frames, loss

def optimize_params(x, *args):
    data_preload, frames, correct_n = args
    params = {"dp":x[0], "minDist":int(x[1]), "param1":x[2], "param2":x[3], "minRadius":int(x[4]), "maxRadius":int(x[5])}
    _, _, loss = hough_feature_location(data_preload, frames, correct_n, params, False)
    # Save the current best score and set of parameters to a CSV file
    a = [loss, x[0], int(x[1]), x[2], x[3], int(x[4]), int(x[5])]
    with open('optimization_results.csv', mode = 'a', newline='') as file:
       writer = csv.writer(file)
       writer.writerow(a)
    print(a)
    return loss

def plot_optimization_results(opt_result_df, slot2):
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(0, len(opt_result_df.loss), 1), opt_result_df.loss, 'b-')
    ax.set_ylabel("loss", color = 'b') 
    ax1 = ax.twinx() 
    ax1.plot(np.arange(0, len(opt_result_df[slot2]), 1), opt_result_df[slot2], 'r.')
    ax1.set_ylabel(slot2, color='r')
    ax.grid()
    plt.show()
    return fig

# Clear the contents of the CSV file before starting the optimization
with open('optimization_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['loss', 'dp', 'minDist', 'param1', 'param1', 'minRadius', 'maxRadius'])

opt_result = dual_annealing(optimize_params, x0 = [2, 8, 20, 0.8, 10, 35], args = (data_preload, frames, correct_n),\
                            bounds = [(1, 3), (5, 20), (20, 200), (0.3, 1), (5, 20), (20, 40)], \
                            maxiter=1000, seed=1234)
opt_result