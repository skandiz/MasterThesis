import cv2
import numpy as np
from matplotlib import pyplot as plt
%matplotlib ipympl
import matplotlib.animation
import matplotlib as mpl
import joblib
parallel = joblib.Parallel(n_jobs = -1)

import random
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd

from tifffile import imsave, imwrite
from tqdm import tqdm

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from csbdeep.utils import Path, normalize

from tracking_utils import *

np.random.seed(42)
lbl_cmap = random_label_cmap()

if 1:
    def detect_features_from_imgs(imgs, frames, model, model_name):
        feature_properties_dict = {'frame':[], 'centroid-1':[], 'centroid-0':[], 'area':[], 'r':[], 'eccentricity':[],\
                                    'prob':[], 'area_bbox':[], 'area_convex':[], 'area_filled':[], 'axis_major_length':[],\
                                    'axis_minor_length':[], 'bbox-0':[], 'bbox-1':[], 'bbox-2':[], 'bbox-3':[],\
                                    'equivalent_diameter_area':[], 'euler_number':[], 'extent':[], 'feret_diameter_max':[],\
                                    'inertia_tensor-0-0':[], 'inertia_tensor-0-1':[], 'inertia_tensor-1-0':[],\
                                    'inertia_tensor-1-1':[], 'inertia_tensor_eigvals-0':[], 'inertia_tensor_eigvals-1':[],\
                                    'label':[]}

        for frame in tqdm(frames):
            img = imgs[frame]
            feature_properties_dict = detect_features_frame(feature_properties_dict, frame, img, model)
                    
        feature_properties_dict['r'] = np.sqrt(np.array(feature_properties_dict['area'])/np.pi)
        raw_detection_df = pd.DataFrame(feature_properties_dict)
        raw_detection_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
        raw_detection_df['frame'] = raw_detection_df.frame.astype('int')
        raw_detection_df.sort_values(by=['frame', 'prob'], ascending=[True, False], inplace=True)
        return raw_detection_df
        
    def overlap_between_circles(existing_circles, center, radius):
        for existing_center in existing_circles:
            distance = np.linalg.norm(np.array(existing_center) - np.array(center))
            if distance < 2*radius:
                return True
        return False

    def initial_droplet_positions(nFeatures, rFeature, rMax):
        list_of_centers = []
        for i in range(nFeatures):
            while True:
                # Generate a random position inside the outer circle 
                theta = random.uniform(0, 2 * np.pi)
                r = random.uniform(0, rMax - rFeature)
                center = (r * np.cos(theta), r * np.sin(theta))
                if not overlap_between_circles(list_of_centers, center, rFeature):
                    list_of_centers.append(center)
                    break
        return np.array(list_of_centers)

    # Function to check for collisions between droplets
    def handle_droplet_collisions(pos, droplet_radius):
        r_ij_m = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2)
        mask = np.tril(r_ij_m < 2 * droplet_radius, k=-1)
        r_ij = (pos[:, np.newaxis] - pos) * mask[:, :, np.newaxis]
        # Normalize displacements
        norms = np.linalg.norm(r_ij, axis=2)
        norms[norms == 0] = 1 # Avoid division by zero
        r_ij_v = r_ij / norms[:, :, np.newaxis]
        # Calculate adjustment factor
        adjustment = (2 * droplet_radius - r_ij_m) * mask
        # Apply adjustments to positions
        pos += np.sum(r_ij_v * (adjustment / 2)[:, :, np.newaxis], axis=1)
        pos -= np.sum(r_ij_v * (adjustment / 2)[:, :, np.newaxis], axis=0)

    def handle_boundary_collisions(pos, outer_radius, droplet_radius):
        distances = np.linalg.norm(pos, axis=1)
        # Find indices where distances exceed the boundary
        out_of_boundary_mask = distances > outer_radius - droplet_radius
        # Calculate adjustment factor for positions exceeding the boundary
        adjustment = (outer_radius - droplet_radius) / distances[out_of_boundary_mask]
        # Apply adjustments to positions
        pos[out_of_boundary_mask] *= adjustment[:, np.newaxis]

    def short_range_align(T0, pos, orientations, align_radius):
        T = np.zeros(pos.shape[0])
        for n in range(pos.shape[0]):
            v_n = np.array([np.cos(orientations[n]), np.sin(orientations[n])])
            r_ni = pos[n] - pos[np.arange(pos.shape[0])!=n]
            r_i = np.linalg.norm(r_ni, axis=1)
            S = np.where(r_i < align_radius)[0]
            T[n] = T0 * np.sum(np.divide(np.sum(v_n*r_ni[S], axis = 1), r_i[S]**2) *\
                                np.cross(np.array([np.cos(orientations[n]), np.sin(orientations[n])]), r_ni[S]))
        return T

    def handle_boundary_repulsion(pos, repulsion_radius, repulsion_strength, dt):    
        distances = np.linalg.norm(pos, axis = 1) 
        boundary_indices = distances > outer_radius - repulsion_radius
        if np.any(boundary_indices):
            # Calculate repulsion force direction
            directions = - pos / distances[:, np.newaxis]
            forces = repulsion_strength / ((outer_radius - distances) ** 2)[:, np.newaxis]
            pos[boundary_indices] += forces[boundary_indices] * directions[boundary_indices] * dt

    def lj_interaction(pos, epsilon, sigma, dt):
        r_ij = pos[:, np.newaxis] - pos
        r_ij_m = np.linalg.norm(r_ij, axis=2)
        directions = r_ij / r_ij_m[:, :, np.newaxis]
        directions[np.isnan(directions)] = 0
        lj_force = 4 * epsilon * (12 * sigma**12 / r_ij_m**13 - 6 * sigma**6 / r_ij_m**7)
        lj_force[np.isnan(lj_force)] = 0
        forces = np.sum(lj_force[:, :, np.newaxis] * directions, axis=1)
        pos += forces * dt


run_simulation_verb = False
generate_synthetic_images = True
run_detection_verb = False
run_linking_verb = False

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

if run_simulation_verb:
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
        handle_boundary_repulsion(pos = positions, repulsion_radius = repulsion_radius, repulsion_strength = repulsion_strength, dt = dt)
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
        
    simulated_trajectories = pd.DataFrame({'frame': frames, 'x': x, 'y': y, 'r': r, 'label': label})
    simulated_trajectories.to_parquet(f'./simulation/simulated_trajectories_{fps}_fps.parquet')
else:
    simulated_trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_{fps}_fps.parquet')


resolution = 500

test_frame = 5000
test_img, test_mask, circles_array = zip(*parallel(generate_synthetic_image_from_simulation_data_parallel(trajectories=simulated_trajectories,\
                                                                        frame=test_frame, height=resolution, width=resolution,\
                                                                        gaussian_sigma=5*resolution/500,\
                                                                        gaussian_amplitude=20, color=100, scale = resolutions/500\
                                                                        sharp_verb=True) for frame in range(1)))
test_img = test_img[0]
test_mask = test_mask[0]
circles_array = circles_array[0]

temp = simulated_trajectories.loc[(simulated_trajectories.frame == test_frame), ["x", "y", "r"]]
fig, ax = plt.subplots(1, 3, figsize = (10, 5))
for i in range(len(temp)):
    ax[0].add_artist(plt.Circle((temp.x.values[i], temp.y.values[i]), temp.r.values[i], fill = True, alpha = 0.5, color = 'b'))
ax[0].add_artist(plt.Circle((0, 0), 250, color='r', fill=False))
ax[0].set_xlim(-260, 260)
ax[0].set_ylim(260, -260)
ax[1].imshow(test_img, cmap="gray", vmin=0, vmax=255)
ax[2].imshow(test_mask, cmap=lbl_cmap)
ax[0].set_aspect('equal')
ax[0].set(xticks=[], yticks=[], title='Simulated droplet positions')
ax[1].set(xticks=[], yticks=[], title='Synthetic image')
ax[2].set(xticks=[], yticks=[], title='Synthetic mask')
plt.tight_layout()
plt.savefig('./simulation/synthetic_image_from_simulation.png', dpi = 300)
plt.close()

if generate_synthetic_images:
    sample_frames = simulated_trajectories.frame.unique()
    test = parallel(
                    generate_synthetic_image_from_simulation_data_parallel(trajectories=simulated_trajectories,\
                                                                frame=frame, height=resolution, width=resolution,\
                                                                gaussian_sigma=5*resolution/500, scale = resolutions/500, \
                                                                gaussian_amplitude=20, color=100, sharp_verb=True)
                    for frame in tqdm(sample_frames)
    )

    test_img = np.array([i[0] for i in test])

    # save images and masks for training
    for i in tqdm(range(test_img.shape[0])):
        imwrite(f'./simulation/synthetic_dataset_{fps}_fps/image/synthetic_{i}.tif', test_img[i])


fps = 100
simulated_trajectories['particle'] = simulated_trajectories['label']
simulated_trajectories.loc[:, ['x','y']] = simulated_trajectories.loc[:, ['x','y']] + 250
frames = simulated_trajectories.frame.unique()
imgs = []
for frame in tqdm(frames):
    imgs.append(imread(f'./simulation/synthetic_dataset_{fps}_fps/image/synthetic_{frame}.tif'))


if run_detection_verb:
    #model_name = 'stardist_trained'          # stardist model trained for 50 epochs on simulated synthetic dataset
    model_name = 'modified_2D_versatile_fluo' # stardist model trained for 150 epochs on simulated dataset starting from the pretrained 2D versatile fluo model
    model = StarDist2D(None, name = model_name, basedir = './models/')


    raw_detection_df = detect_features_from_imgs(imgs, frames, model, model_name)
    raw_detection_df.to_parquet(f'./simulation/simulated_video_raw_detection_{fps}_fps_{resolution}.parquet')