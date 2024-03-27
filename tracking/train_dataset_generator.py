from matplotlib import pyplot as plt
import matplotlib.animation
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import multiprocessing
n_jobs = int(multiprocessing.cpu_count()*0.8)
print(f'Using {n_jobs} cores for parallel jobs')
parallel = joblib.Parallel(n_jobs=n_jobs, backend='loky', verbose=0)
from stardist import random_label_cmap
lbl_cmap = random_label_cmap()
from tracking_utils import *


run_simulation_verb = False
generate_synthetic_images = True
test_verb = True

resolution = 1000
fps = 100
sc = int(resolution/500)


if run_simulation_verb:
    nInstances = 50
    time_steps = 100000
    fps = 100
    dt = 1 / fps
    mean_instance_radius = 10 # radius of the instances
    time_constant_r = 3*time_steps # time constant for the exponential decay of the instance radius
    outer_radius = 250
    v0_init = 10 # magnitude of the self-propulsion velocity
    time_constant = time_steps # time constant for the exponential decay of the self-propulsion velocity

    D_r = .1 # rotational diffusion coefficient
    D_t = .1 # translational diffusion coefficient

    T0 = 10 # magnitude of the short-range alignment force
    align_radius = 7*mean_instance_radius  # radius for short-range alignment

    repulsion_radius = outer_radius #50
    repulsion_strength = 5*10**3
    # parameters for the Lennard-Jones potential
    epsilon = 2
    sigma = 2*mean_instance_radius
    # Initialize instance positions and orientations
    radius_of_instances = np.random.normal(mean_instance_radius, 0.1, nInstances)

    positions = initial_instance_positions(nInstances, radius_of_instances, outer_radius)
    orientations = np.random.uniform(0, 2*np.pi, size = nInstances)

    if 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(np.arange(time_steps), v0_init * np.exp(-np.array(range(time_steps)) / time_constant))
        ax.set(xlabel='time', ylabel='v0', title='Self-propulsion velocity decay')
        plt.savefig('./simulation/self_propulsion_velocity_decay.png')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(nInstances):
            ax.plot(np.arange(time_steps), radius_of_instances[i] * np.exp(-np.arange(time_steps)/(time_constant_r)))
        ax.set(xlabel='time', ylabel='radius', title='instance radius decay')
        plt.savefig('./simulation/radius_decay.png')
        plt.close()

        test_pos = initial_instance_positions( nInstances, radius_of_instances, outer_radius)
        distances = np.linalg.norm(test_pos, axis=1)
        boundary_indices = distances > outer_radius - repulsion_radius
        if np.any(boundary_indices):
            directions = - test_pos / distances[:, np.newaxis]
            forces = repulsion_strength / ((outer_radius - distances) ** 2)[:, np.newaxis]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(nInstances):
            if boundary_indices[i]:
                ax.add_artist(plt.Circle((test_pos[i, 0], test_pos[i, 1]), radius_of_instances[i], color='r', fill=True, alpha=0.5))
            else:
                ax.add_artist(plt.Circle((test_pos[i, 0], test_pos[i, 1]), radius_of_instances[i], color='r', fill=False))
        ax.add_artist(plt.Circle((0, 0), outer_radius, color='b', fill=False))
        ax.quiver(test_pos[boundary_indices, 0], test_pos[boundary_indices, 1], (forces * directions)[boundary_indices, 0], (forces * directions)[boundary_indices, 1])
        ax.add_artist(plt.Circle((0, 0), outer_radius - repulsion_radius, color='b', fill=False))
        ax.set(xlim = (-outer_radius , outer_radius), ylim = (-outer_radius, outer_radius), title = 'Initial instance positions and boundary repulsion')
        plt.savefig('./simulation/initial_instance_positions.png', dpi = 300)
        plt.close()

        r_ij = test_pos[:, np.newaxis] - test_pos
        r_ij_m = np.linalg.norm(r_ij, axis=2)
        directions = r_ij / r_ij_m[:, :, np.newaxis]
        directions[np.isnan(directions)] = 0
        lj_force = 4 * epsilon * (12 * sigma**12 / r_ij_m**13 - 6 * sigma**6 / r_ij_m**7)
        lj_force[np.isnan(lj_force)] = 0
        forces = np.sum(lj_force[:, :, np.newaxis] * directions, axis=1)

        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for i in range(nInstances):
            ax.add_artist(plt.Circle((test_pos[i, 0], test_pos[i, 1]), radius_of_instances[i], color='r', fill=False))
        ax.quiver(test_pos[:, 0], test_pos[:, 1], forces[:, 0], forces[:, 1], color='r')
        ax.set(xlim = (-outer_radius , outer_radius), ylim = (-outer_radius, outer_radius), title = 'Initial instance positions and Lennard-Jones forces')
        for i in range(nInstances):
            ax1.add_artist(plt.Circle((test_pos[i, 0] + forces[i, 0]*dt, test_pos[i, 1] + forces[i, 1]*dt), radius_of_instances[i], color='r', fill=False))
        ax1.set(xlim = (-outer_radius , outer_radius), ylim = (-outer_radius, outer_radius), title = 'instance positions after Lennard-Jones forces')
        plt.close()

    frames, x, y, r, label = [], [], [], [], []
    for step in tqdm(range(time_steps)):
        # update magnitude of self-propulsion velocity
        v0 = v0_init * np.exp(-step / time_constant)
        instance_radius = radius_of_instances * np.exp(-step / time_constant_r)
        # Update positions
        positions += v0 * np.array([np.cos(orientations), np.sin(orientations)]).T * dt + \
                        np.random.normal(scale=np.sqrt(2 * D_t * dt), size=(nInstances, 2))

        # Handle collisions with the boundary and between instances
        handle_instance_collisions(positions, instance_radius)
        handle_boundary_repulsion(positions, repulsion_radius, outer_radius, repulsion_strength, dt)
        lj_interaction(positions, epsilon, sigma, dt)
        
        # Update orientations with rotational diffusion and short-range alignment
        short_range_align(T0, positions, orientations, align_radius)
        orientations += np.random.normal(scale=np.sqrt(2 * D_r * dt), size=nInstances)

        # Ensure orientations stay within [0, 2*pi)
        orientations %= 2 * np.pi

        frames += [step for i in range(nInstances)]
        x += list(positions[:, 0])
        y += list(positions[:, 1])
        r += list(instance_radius)
        label += [i for i in range(nInstances)]
        
    trajectories = pd.DataFrame({'frame': frames, 'x': x, 'y': y, 'r': r, 'label': label})
    trajectories.to_parquet(f'./simulation/simulated_trajectories_{fps}_fps_r_decay_r_gaussian_2.parquet')
else:
    trajectories = pd.read_parquet(f'./simulation/simulated_trajectories_{fps}_fps_r_decay_r_gaussian_2.parquet')

if 0:
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True
            
    def update_graph(frame):
        df = trajectories.loc[(trajectories.frame == frame), ["x", "y", "r"]]
        for i in range(len(df)):
            graph[i].center = (df.x.values[i], df.y.values[i])
            graph[i].radius = df.r.values[i]
        title.set_text(f'Tracking -- step = {frame} ')
        return graph

    title = ax.set_title(f'Tracking -- step = {0} ')
    ax.set(xlabel = 'X [px]', ylabel = 'Y [px]')
    df = trajectories.loc[(trajectories.frame == 0), ["x", "y", "r"]]

    graph = []
    for i in range(len(df)):
        graph.append(ax.add_artist(plt.Circle((df.x.values[i], df.y.values[i]), df.r.values[i], fill = True, alpha = 0.5, color = 'b')))
    ax.add_artist(plt.Circle((0, 0), outer_radius, color='r', fill=False))
    ax.set_xlim(-outer_radius, outer_radius)
    ax.set_ylim(-outer_radius, outer_radius)

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, range(time_steps)[::100], interval = 5, blit = False)
    writer = matplotlib.animation.FFMpegWriter(fps = 30, metadata = dict(artist='Matteo Scandola'), extra_args = ['-vcodec', 'libx264'])
    ani.save(f'./simulation/test.mp4', writer = writer)
    plt.close()

if test_verb:
    test_frame = 5000
    test_img, test_mask = generate_synthetic_image_from_simulation_data(trajectories=trajectories,\
                                            frame=test_frame, height=500*sc, width=500*sc, gaussian_sigma=5*sc,\
                                            gaussian_amplitude=20, color=100, scale = sc, save_path = None,\
                                            sharp_verb=True)

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
    ax[0].set(xticks=[], yticks=[], title='Simulated instance positions')
    ax[1].set(xticks=[], yticks=[], title='Synthetic image')
    ax[2].set(xticks=[], yticks=[], title='Synthetic mask')
    plt.tight_layout()
    plt.savefig(f'./simulation/synthetic_image_from_simulation_{resolution}_resolution.png', dpi = 300)
    plt.close()


if generate_synthetic_images:
    frames = trajectories.frame.unique()
    parallel(joblib.delayed(generate_synthetic_image_from_simulation_data)(trajectories=trajectories,\
                    frame=frame, height=500*sc, width=500*sc, gaussian_sigma=5*sc,\
                    gaussian_amplitude=20, color=100, scale = sc, save_path = f'./simulation/synthetic_dataset_{fps}_fps_r_decay_r_gaussian_2/',\
                    sharp_verb=True) for frame in tqdm(frames))