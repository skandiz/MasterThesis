import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import matplotlib as mpl
import joblib
parallel = joblib.Parallel(n_jobs = -1)

import random
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd

from tifffile import imsave
from tqdm import tqdm

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from csbdeep.utils import Path, normalize

from tracking_utils_old import *

np.random.seed(42)
lbl_cmap = random_label_cmap()

# SETUP
np.random.seed(0)
num_droplets = 50
time_steps = 100000
fps = 100
dt = 1 / fps
droplet_radius = 10 # radius of the droplets
outer_radius = 250
v0_init = 10 # magnitude of the self-propulsion velocity
time_constant = time_steps # time constant for the exponential decay of the self-propulsion velocity

D_r = .1 # rotational diffusion coefficient
D_t = .1 # translational diffusion coefficient

T0 = 10 # magnitude of the short-range alignment force
align_radius = 7*droplet_radius  # radius for short-range alignment

repulsion_radius = outer_radius #50
repulsion_strength = 5*10**3
# parameters for the Lennard-Jones potential
epsilon = 2
sigma = 2*droplet_radius


if 0:
    # Initialize droplet positions and orientations
    positions = initial_droplet_positions(nFeatures = num_droplets, rFeature = droplet_radius, rMax = outer_radius)
    orientations = np.random.uniform(0, 2*np.pi, size = num_droplets)

    frames, x, y, r, label = [], [], [], [], []

    for step in tqdm(range(time_steps)):
        # update magnitude of self-propulsion velocity
        v0 = v0_init * np.exp(-step / time_constant)
        # Update positions
        positions += v0 * np.array([np.cos(orientations), np.sin(orientations)]).T * dt + \
                        np.random.normal(scale=np.sqrt(2 * D_t * dt), size=(num_droplets, 2))

        # Handle collisions with the boundary and between droplets
        handle_droplet_collisions(pos = positions, droplet_radius = droplet_radius)
        #handle_boundary_collisions(pos = positions, outer_radius = outer_radius, droplet_radius = droplet_radius) 
        handle_boundary_repulsion(pos = positions, outer_radius=outer_radius, repulsion_radius = repulsion_radius, repulsion_strength = repulsion_strength, dt = dt)
        lj_interaction(pos = positions, epsilon = epsilon, sigma = sigma, dt = dt)
        
        # Update orientations with rotational diffusion and short-range alignment
        short_range_align(T0, positions, orientations, align_radius)
        orientations += np.random.normal(scale=np.sqrt(2 * D_r * dt), size=num_droplets)

        # Ensure orientations stay within [0, 2*pi)
        orientations %= 2 * np.pi

        frames += [step for i in range(num_droplets)]
        x += list(positions[:, 0])
        y += list(positions[:, 1])
        r += [droplet_radius for i in range(num_droplets)]
        label += [i for i in range(num_droplets)]
        
    trajectories = pd.DataFrame({'frame': frames, 'x': x, 'y': y, 'r': r, 'label': label})
    trajectories.to_parquet('./simulation/simulated_trajectories_100_fps.parquet')
else:
    trajectories = pd.read_parquet('./simulation/simulated_trajectories_100_fps.parquet')

frames = np.random.choice(trajectories.frame.unique(), size=10000, replace=False)
resolution = 1000
sc = int(resolution/500)


def create_gaussian(center, img_width, img_height, sigma, ampl):
    center_x, center_y = center
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2.0 * sigma**2))
    return np.round(ampl*(gaussian / np.max(gaussian))).astype(np.uint8)

@joblib.delayed
def generate_synthetic_image_from_simulation_data(trajectories, frame, height, width, gaussian_sigma, gaussian_amplitude, color, scale, save_path, sharp_verb=False):
    trajs = trajectories.loc[(trajectories.frame == frame), ["x", "y", "r"]]*scale
    # create background image
    image = np.random.randint(70, 75, (height, width), dtype=np.uint8)
    # Draw the outer circle mimicking the petri dish
    cv2.circle(image, (int(height/2), int(width/2)), int(width/2), 150)
    cv2.circle(image, (int(height/2), int(width/2)), int(width/2)-4, 150)
    image = cv2.GaussianBlur(image, (5, 5), 4)
    
    # initialize mask
    mask = np.zeros((height, width), dtype=np.uint8)
    list_of_centers = []
    circles_array = np.zeros((height, width), dtype=np.uint8)
    list_of_distances = []
    for i in range(len(trajs)):
        index = i + 1 
        center = (int(width/2 + trajs.x.values[i]), int(height/2 + trajs.y.values[i]))
        instance_radius = int(trajs.r.values[i])
        cv2.circle(image, center, instance_radius, color, -1, lineType=8) 
        circles_array += create_gaussian(center, width, height, gaussian_sigma, gaussian_amplitude)
        cv2.circle(mask, center, instance_radius, (index), -1)
    
    if sharp_verb:
        image = cv2.GaussianBlur(image, (5, 5), 2)
        kernel = np.array([[0, -1, 0],
                          [-1, 5,-1],
                          [0, -1, 0]])
        image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    
    # add gaussian profile to droplets
    image += circles_array 
    if save_path is not None: 
        imwrite(save_path + f'image/frame_{frame}_{height}_resolution.tif', image, compression='zlib')
        imwrite(save_path + f'mask/frame_{frame}_{height}_resolution.tif', mask, compression='zlib')
    return image, mask

if 1:
    parallel(generate_synthetic_image_from_simulation_data(trajectories=trajectories,\
                    frame=frame, height=500*sc, width=500*sc, gaussian_sigma=5*sc,\
                    gaussian_amplitude=20, color=100, scale = sc, save_path = f'./simulation/synthetic_dataset_100_fps_v2/',\
                    sharp_verb=True) for frame in tqdm(frames))