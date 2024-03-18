import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import matplotlib as mpl
import joblib
import multiprocessing
n_jobs = int(multiprocessing.cpu_count()*0.9)
parallel = joblib.Parallel(n_jobs=n_jobs, backend='loky', verbose=0)
import random
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from tifffile import imsave, imwrite
from tqdm import tqdm
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from csbdeep.utils import Path, normalize
from stardist.models import StarDist2D
from tracking_utils import *
import trackpy as tp 
np.random.seed(42)
lbl_cmap = random_label_cmap()

run_simulation_verb = False
generate_synthetic_images = False
run_detection_verb = True
run_linking_verb = True

if run_simulation_verb:

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
    fps = 100
    simulated_trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_{fps}_fps.parquet')


resolution = 1000
frames = simulated_trajectories.frame.unique()
frames = frames[:2]

if 1:
    test_frame = 5000
    test_img, test_mask, circles_array = zip(*parallel(generate_synthetic_image_from_simulation_data(trajectories=simulated_trajectories,\
                                                                            frame=test_frame, height=resolution, width=resolution,\
                                                                            gaussian_sigma=5*resolution/500,\
                                                                            gaussian_amplitude=20, color=100, scale = resolution/500,
                                                                            save_imgs_path = None,\
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
    parallel(
            generate_synthetic_image_from_simulation_data(trajectories=simulated_trajectories,\
                                                        frame=frame, height=resolution, width=resolution,\
                                                        gaussian_sigma=5*resolution/500, scale = resolution/500, \
                                                        gaussian_amplitude=20, color=100,\
                                                        save_imgs_path = f'./simulation/synthetic_dataset_{fps}_fps/image_{resolution}/',\
                                                        sharp_verb=True)
            for frame in tqdm(frames)
    )

if run_detection_verb:
    model_name = 'modified_2D_versatile_fluo_gpu' # stardist model trained for 150 epochs on simulated dataset starting from the pretrained 2D versatile fluo model
    model = StarDist2D(None, name = model_name, basedir = './models/')
    print(model.config)
    raw_detection_df = detect_features_from_images(frames, f'./simulation/synthetic_dataset_{fps}_fps/image_{resolution}/', model)
    raw_detection_df.to_parquet(f'./simulation/simulated_video_raw_detection_{fps}_fps_{resolution}.parquet')

if run_linking_verb:
    print('Linking stardist_trajectories...')
    cutoff = 100
    print(raw_detection_df)
    stardist_trajectories = tp.link_df(raw_detection_df, cutoff, memory = 1, link_strategy = 'hybrid', neighbor_strategy = 'KDTree', adaptive_stop = 1)
    stardist_trajectories = stardist_trajectories.sort_values(['frame', 'particle'])
    # CREATE COLOR COLUMN AND SAVE DF
    n = max(stardist_trajectories.particle)
    print(f'N of droplets: {n + 1}')
    random.seed(5)
    colors = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    for i in range(max(stardist_trajectories.particle)+1-n):
        colors.append('#00FFFF')
    c = []
    for p in stardist_trajectories.particle:
        c.append(colors[p])
    stardist_trajectories['color'] = c
    stardist_trajectories = stardist_trajectories.reset_index(drop=True)
    stardist_trajectories.to_parquet(f'./simulation/simulated_video_linked_detection_{fps}_fps_{resolution}.parquet')
else:
    stardist_trajectories = pd.read_parquet(f'./simulation/simulated_video_linked_detection_{fps}_fps_{resolution}.parquet')


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for i in range(50):
    df = stardist_trajectories.loc[stardist_trajectories.particle == i]
    ax.plot(df.x, df.y)
plt.savefig('test.png', dpi = 500)
plt.close()