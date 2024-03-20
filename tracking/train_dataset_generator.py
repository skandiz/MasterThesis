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

from tifffile import imsave, imwrite
from tqdm import tqdm

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from csbdeep.utils import Path, normalize

from tracking_utils import *

lbl_cmap = random_label_cmap()


run_simulation_verb = False


if run_simulation_verb:
    # SETUP
    np.random.seed(0)

    num_droplets = 50
    time_steps = 100000
    fps = 100
    dt = 1 / fps
    droplet_radius_init = 10 # radius of the droplets
    time_constant_r = 3*time_steps # time constant for the exponential decay of the droplet radius
    outer_radius = 250
    v0_init = 10 # magnitude of the self-propulsion velocity
    time_constant = time_steps # time constant for the exponential decay of the self-propulsion velocity

    D_r = .1 # rotational diffusion coefficient
    D_t = .1 # translational diffusion coefficient

    T0 = 10 # magnitude of the short-range alignment force
    align_radius = 7*droplet_radius_init  # radius for short-range alignment

    repulsion_radius = outer_radius #50
    repulsion_strength = 5*10**3
    # parameters for the Lennard-Jones potential
    epsilon = 2
    sigma = 2*droplet_radius_init
    # Initialize droplet positions and orientations
    positions = initial_droplet_positions(nFeatures = num_droplets, rFeature = droplet_radius_init, rMax = outer_radius)
    orientations = np.random.uniform(0, 2*np.pi, size = num_droplets)

    frames, x, y, r, label = [], [], [], [], []

    for step in tqdm(range(time_steps)):
        # update magnitude of self-propulsion velocity
        v0 = v0_init * np.exp(-step / time_constant)
        droplet_radius = droplet_radius_init * np.exp(-step / time_constant_r)
        # Update positions
        positions += v0 * np.array([np.cos(orientations), np.sin(orientations)]).T * dt + \
                        np.random.normal(scale=np.sqrt(2 * D_t * dt), size=(num_droplets, 2))

        # Handle collisions with the boundary and between droplets
        handle_droplet_collisions(pos = positions, droplet_radius = droplet_radius)
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
        
    trajectories = pd.DataFrame({'frame': frames, 'x': x, 'y': y, 'r': r, 'label': label})
    trajectories.to_parquet(f'./simulation/simulated_trajectories_{fps}_fps_r_decay.parquet')
else:
    fps = 100
    trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_{fps}_fps_r_decay.parquet')



generate_synthetic_images = True
test_verb = False
fps = 100
trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_{fps}_fps_r_decay.parquet')
frames = np.random.choice(trajectories.frame.unique(), size=1000, replace=False)
print(frames[:10])
resolution = 1500
sc = int(resolution/500)


if test_verb:
    test_frame = 5000
    test_img, test_mask = zip(*parallel(generate_synthetic_image_from_simulation_data(trajectories=trajectories,\
                                                        frame=test_frame, height=500*sc, width=500*sc, gaussian_sigma=5*sc,\
                                                        gaussian_amplitude=20, color=100, scale = sc, save_path = None,\
                                                        sharp_verb=True) for frame in range(1)))
    test_img = test_img[0]
    test_mask = test_mask[0]

    temp = trajectories.loc[(trajectories.frame == test_frame), ["x", "y", "r"]]
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
    plt.savefig(f'./simulation/synthetic_image_from_simulation_{resolution}_resolution.png', dpi = 300)
    plt.close()


if generate_synthetic_images:
    parallel(generate_synthetic_image_from_simulation_data(trajectories=trajectories,\
                    frame=frame, height=500*sc, width=500*sc, gaussian_sigma=5*sc,\
                    gaussian_amplitude=20, color=100, scale = sc, save_path = f'./simulation/synthetic_dataset_{fps}_fps/',\
                    sharp_verb=True) for frame in tqdm(frames))