import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
mpl.rc('image', cmap='gray')

import h5py 
import trackpy as tp
tp.quiet()

import joblib 

import numpy as np
import pandas as pd
import csv, json
import pims
from PIL import Image, ImageDraw
import cv2

from scipy.optimize import dual_annealing, linear_sum_assignment
from scipy.spatial import distance_matrix
from tqdm import tqdm
import random

import skimage
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois

np.random.seed(6)
lbl_cmap = random_label_cmap()
# initialize model with versatile fluorescence pretrained weights
model = StarDist2D.from_pretrained('2D_versatile_fluo')
print(model)

run_preprocessing_verb = False


def get_data_preload(startFrame, endFrame, data_preload_path, dataset_name):
    with h5py.File(data_preload_path, 'r') as f:
        dataset = f[dataset_name]
        frameImg = dataset[startFrame:endFrame]
    return frameImg

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
    return image_sharp

def test_setup(n_of_samples, frames):
    droplets_found = []
    area = []
    frames_sample = np.sort(np.random.choice(np.arange(0, len(frames), 1, dtype=int), n_of_samples, replace=False))
    img = get_data_preload(frames_sample[0], frames_sample[0] + 1, data_preload_path, 'dataset_name')[0]      

    labels_test, dict_test = model.predict_instances(normalize(img), predict_kwargs = {'verbose':False}) 
    test = skimage.measure.regionprops_table(labels_test, properties=('centroid', 'area'))

    fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 6), sharex=True, sharey=True)
    coord, points, prob = dict_test['coord'], dict_test['points'], dict_test['prob']
    ax.imshow(img, cmap='gray'); 
    ax.set(title = 'Preprocessed Image', xlabel='X [px]', ylabel='Y [px]')
    ax1.imshow(img, cmap='gray'); 
    _draw_polygons(coord, points, prob, show_dist=True)
    ax1.set(title = f"Stardist result", xlabel='X [px]', ylabel='Y [px]')
    plt.suptitle(f"Stardist result on frame {frames_sample[0] + frames[0]}")
    plt.tight_layout()
    #ax.set(xlim=(462, 510), ylim = (280, 330))
    plt.savefig(save_path + f'stardist_example_part{part}.pdf', format='pdf')
    plt.close()

    print(f"testing setup on {n_of_samples} samples...")
    for frame in tqdm(frames_sample):
        img = get_data_preload(frame, frame + 1, data_preload_path, 'dataset_name')[0]
        labels_test, dict_test = model.predict_instances(normalize(img), predict_kwargs = {'verbose':False}) 
        droplets_found.append(dict_test['coord'].shape[0])
        test = skimage.measure.regionprops_table(labels_test, properties=('centroid', 'area'))
        area += list(test['area'])

    print(f"Average number of droplets found: {np.mean(droplets_found)}")
    print(f"N of times droplets > {nDrops}: {np.sum(np.array(droplets_found) > nDrops)} / {n_of_samples} ")
    print(f"N of times droplets < {nDrops}: {np.sum(np.array(droplets_found) < nDrops)} / {n_of_samples}")
    print(f"Average radius of droplets found: {np.mean(np.sqrt(area)/np.sqrt(np.pi))}")

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax.plot(np.sqrt(area)/np.sqrt(np.pi))
    ax.set(ylabel='Radius [px]', title='Radius of droplets found')
    ax1.plot(frames_sample, droplets_found)
    ax.set(ylabel='Radius [px]', title='Radius of droplets found')
    plt.savefig(save_path + f'test_setup_part{part}.pdf', format='pdf')
    plt.close()

parameters = pd.read_csv('./setup_file.csv')
print(parameters)

video_selection   = parameters["video_selection"][0]
system_name       = f"{video_selection} system"
source_path       = parameters["source_path"][0]
part              = parameters["part"][0]
range_frames       = np.arange(10**5*(part-1), 10**5*part, 1, dtype = int)
start_frame = parameters["start_frame"][0]
end_frame   = parameters["end_frame"][0]
if start_frame not in range_frames or end_frame-1 not in range_frames:
    raise ValueError(f"start_frame and end_frame must be in range {range_frames[0]}-{range_frames[-1]+1} since part = {part}")
save_path         = parameters["save_path"][0] # 
data_preload_path = parameters["data_preload_path"][0] + "part2.h5"

xmin = parameters["xmin"][0]
ymin = parameters["ymin"][0]
xmax = parameters["xmax"][0]
ymax = parameters["ymax"][0]
rmax = parameters["rmax"][0]
rmin = parameters["rmin"][0]

start_frame = parameters["start_frame"][0]
end_frame   = parameters["end_frame"][0]
nDrops      = parameters["nDrops"][0]

original_video = pims.open(source_path)
fps = int(original_video.frame_rate)
data_type = original_video.pixel_type
w = original_video.frame_shape[1]
h = original_video.frame_shape[0]

preprocessed_video = preprocessing(original_video, h, w, xmin, ymin, xmax, ymax) 
frames = np.arange(start_frame, end_frame, 1, dtype=int)

# optimize by saving data from xmin to xmax and from ymin to ymax !!!!!
try: 
    test = get_data_preload(0, 1, data_preload_path, "dataset_name")
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(test[0])
    plt.close()
    print("Data already ready as h5 file")
except:
    print("Data needs to be written as h5 file first:")
    hdf = h5py.File(data_preload_path, "a")
    dset = hdf.create_dataset(name = "dataset_name", shape = (0, h, w),
                              maxshape = (None, h, w), dtype = data_type)
    for frame in tqdm(range_frames):
        dset.resize(dset.shape[0] + 1, axis=0)
        new_data = preprocessed_video[frame]
        dset[-1:] = new_data
    hdf.close()
    pd.DataFrame(frames).to_csv(f"/Volumes/ExtremeSSD/UNI/THESIS/h5_data_thesis/25b-25r/frames_part{part}.csv", index=False)

test_setup(100, frames)


if 1:
    print("Initialize model with versatile fluorescence pretrained weights...")
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    print(model)
    print("Starting location process...") 
    area, x, y, prob, framesList = [], [], [], [], []
    for frame in tqdm(frames[:10000] - frames[0]):
        img = get_data_preload(frame, frame + 1, data_preload_path, 'dataset_name')[0]
        segmented_image, dict_test = model.predict_instances(normalize(img), \
                                                             predict_kwargs = {'verbose':False})
        test = skimage.measure.regionprops_table(segmented_image, properties=('centroid', 'area'))

        area   += list(test['area'])
        y      += list(test['centroid-0'])
        x      += list(test['centroid-1'])
        prob   += list(dict_test['prob'])
        framesList += list(np.ones(len(list(test['centroid-0'])))*(frame+frames[0]))
    # save data
    print("Saving data...")
    df = pd.DataFrame({'x':x, 'y':y, 'area':area, 'prob':prob, 'frame':framesList})
    df['frame'] = df.frame.astype('int')
    df['r'] = np.sqrt(df.area/np.pi)
    df.sort_values(by=['frame', 'prob'], ascending=[True, False], inplace=True)
    df.to_parquet(save_path + f'df.parquet')
else:
    df = pd.read_parquet(save_path + 'raw_trajectories.parquet')
    display(df.head())