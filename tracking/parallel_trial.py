

import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['font.size'] = 8
mpl.rc('image', cmap='gray')
import trackpy as tp
tp.quiet()

import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd
import csv, json
import pims
from PIL import Image, ImageDraw
import cv2

from scipy.optimize import dual_annealing, linear_sum_assignment
from scipy.spatial import distance_matrix

import random

import skimage
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois


np.random.seed(6)
lbl_cmap = random_label_cmap()
# initialize model with versatile fluorescence pretrained weights
model = StarDist2D.from_pretrained('2D_versatile_fluo')
print(model)

def process_frame(frame):
    segm, dict_test = model.predict_instances(normalize(preprocessed_data[frame]), predict_kwargs={'verbose': False})
    test = skimage.measure.regionprops_table(segm, properties=('centroid', 'area'))
    return test

@pims.pipeline
def preprocessing(image, w, h, x1, y1, x2, y2):
    """
    Preprocessing function for the data.

    Parameters
    ----------
    image : pims.Frame
        Frame of the video.
    x1 : int
        x coordinate of the top left corner of the ROI. (region of interest)
    y1 : int
        y coordinate of the top left corner of the ROI.
    x2 : int    
        x coordinate of the bottom right corner of the ROI.
    y2 : int    
        y coordinate of the bottom right corner of the ROI.

    Returns
    -------
    npImage : np.array
        Preprocessed image.
    """
    npImage = np.array(image)
    alpha = Image.new('L', (h, w), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)
    npAlpha = np.array(alpha)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)*npAlpha
    ind = np.where(npImage == 0)
    npImage[ind] = npImage[200, 200]
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    # sharpen image https://en.wikipedia.org/wiki/Kernel_(image_processing)
    image_sharp = cv2.filter2D(src=npImage, ddepth=-1, kernel=kernel)
    #npImage = cv2.medianBlur(npImage, 5)
    #npImage = normalize(npImage)
    return npImage

if __name__ == '__main__':
    ref = pims.open('./data/25r25b-1.mp4')
    print(ref)
    data = preprocessing(ref, 480, 640, 100, 35, 530, 465) #895, 910)
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax.imshow(ref[0], cmap='gray')
    ax1.imshow(data[0], cmap='gray')
    plt.close()

    path = './stardist_res/results_25/'
    if 1:
        nFrames = 1000
        framesList = np.arange(0, nFrames, 1)
        preprocessed_data = np.zeros((nFrames, data[0].shape[0], data[0].shape[1]), dtype=data[0].dtype)
        for i in tqdm(range(nFrames)):
            preprocessed_data[i] = data[i]
        np.savez_compressed(path + 'preprocessed_data.npz', data=preprocessed_data) # --> 15 min
    else:
        preprocessed_data = np.load(path + 'preprocessed.npz')['data'] # --> 3 min

    correct_n = 53

    segm_preload = [None] * nFrames
    area = []
    y = []
    x = []
    prob = []
    frames = []

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_frame, framesList), total=nFrames))

    for ind, test in enumerate(results):
        segm_preload[ind] = test
        area += list(test['area'])
        y += list(test['centroid-0'])
        x += list(test['centroid-1'])
        prob += list(dict_test['prob'])
        frames += list(np.ones(len(list(test['centroid-0'])))*frame)



