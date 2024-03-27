import joblib
import multiprocessing
n_jobs = int(multiprocessing.cpu_count()*0.8)
parallel = joblib.Parallel(n_jobs=n_jobs, backend='loky', verbose=0)
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from csbdeep.utils import normalize
import skimage
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tifffile import imwrite, imread


##################################################################################################################
#                                        PREPROCESSING FUNCTIONS                                                 #
##################################################################################################################

def get_frame(cap, frame, x1, y1, x2, y2, w, h, resolution, preprocess):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()
    if preprocess:
        npImage = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        alpha = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice(((x1, y1), (x2, y2)), 0, 360, fill=255)
        npAlpha = np.array(alpha)
        npImage = npImage*npAlpha
        ind = np.where(npImage == 0)
        npImage[ind] = npImage[200, 200]
        npImage = npImage[y1:y2, x1:x2]
        if npImage.shape[0] > resolution: # if the image is too large --> shrinking with INTER_AREA interpolation
            npImage = cv2.resize(npImage, (resolution, resolution), interpolation = cv2.INTER_AREA)
        else: # if the image is too small --> enlarging with INTER_LINEAR interpolation
            npImage = cv2.resize(npImage, (resolution, resolution), interpolation = cv2.INTER_CUBIC)
        return npImage
    elif not preprocess:
        npImage = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        npImage = npImage[y1:y2, x1:x2]
        npImage = cv2.resize(npImage, (500, 500))
        return npImage
    else:
        raise ValueError("preprocess must be a boolean")


##################################################################################################################
#                                           STARDIST FUNCTIONS                                                   #
##################################################################################################################

def detect_instances_frame(instance_properties_dict, frame, img, model):
    segmented_image, dict_test = model.predict_instances(normalize(img), predict_kwargs = {'verbose' : False})
    instance_properties = skimage.measure.regionprops_table(segmented_image, \
                                                            properties=('area', 'area_bbox', 'area_convex', 'area_filled',\
                                                                        'axis_major_length', 'axis_minor_length',\
                                                                        'bbox', 'centroid', 'eccentricity', \
                                                                        'equivalent_diameter_area', 'euler_number', 'extent',\
                                                                        'feret_diameter_max', 'inertia_tensor',\
                                                                        'inertia_tensor_eigvals', 'label'))

    for key in instance_properties.keys():
        instance_properties_dict[key] += list(instance_properties[key])
    #instance_properties_dict.update({key: instance_properties_dict.get(key, []) + list(instance_properties[key]) for key in instance_properties.keys()})
    instance_properties_dict['prob']  += list(dict_test['prob'])
    instance_properties_dict['frame'] += list(np.ones(len(list(instance_properties['centroid-0'])))*frame)
    return instance_properties_dict



def detect_instances_from_images(frames, imgs_path, resolution, model):
    instance_properties_dict = {'frame':[], 'centroid-1':[], 'centroid-0':[], 'area':[], 'r':[], 'eccentricity':[],\
                                'prob':[], 'area_bbox':[], 'area_convex':[], 'area_filled':[], 'axis_major_length':[],\
                                'axis_minor_length':[], 'bbox-0':[], 'bbox-1':[], 'bbox-2':[], 'bbox-3':[],\
                                'equivalent_diameter_area':[], 'euler_number':[], 'extent':[], 'feret_diameter_max':[],\
                                'inertia_tensor-0-0':[], 'inertia_tensor-0-1':[], 'inertia_tensor-1-0':[],\
                                'inertia_tensor-1-1':[], 'inertia_tensor_eigvals-0':[], 'inertia_tensor_eigvals-1':[],\
                                'label':[]}

    for frame in tqdm(frames):
        img = imread(imgs_path + f'frame_{frame}_{resolution}_resolution.tif')
        instance_properties_dict = detect_instances_frame(instance_properties_dict, frame, img, model)

    instance_properties_dict['r'] = np.sqrt(np.array(instance_properties_dict['area'])/np.pi)
    raw_detection_df = pd.DataFrame(instance_properties_dict)
    raw_detection_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
    raw_detection_df['frame'] = raw_detection_df.frame.astype('int')
    raw_detection_df.sort_values(by=['frame', 'prob'], ascending=[True, False], inplace=True)
    return raw_detection_df

def detect_instances(frames, test_verb, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, resolution, save_path):
    instance_properties_dict = {'frame':[], 'centroid-1':[], 'centroid-0':[], 'area':[], 'r':[], 'eccentricity':[],\
                                'prob':[], 'area_bbox':[], 'area_convex':[], 'area_filled':[], 'axis_major_length':[],\
                                'axis_minor_length':[], 'bbox-0':[], 'bbox-1':[], 'bbox-2':[], 'bbox-3':[],\
                                'equivalent_diameter_area':[], 'euler_number':[], 'extent':[], 'feret_diameter_max':[],\
                                'inertia_tensor-0-0':[], 'inertia_tensor-0-1':[], 'inertia_tensor-1-0':[],\
                                'inertia_tensor-1-1':[], 'inertia_tensor_eigvals-0':[], 'inertia_tensor_eigvals-1':[],\
                                'label':[]}

    for frame in tqdm(frames):
        img = get_frame(video, frame, xmin, ymin, xmax, ymax, w, h, resolution, True)
        instance_properties_dict = detect_instances_frame(instance_properties_dict, frame, img, model)
                
    instance_properties_dict['r'] = np.sqrt(np.array(instance_properties_dict['area'])/np.pi)
    raw_detection_df = pd.DataFrame(instance_properties_dict)
    raw_detection_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
    raw_detection_df['frame'] = raw_detection_df.frame.astype('int')
    raw_detection_df.sort_values(by=['frame', 'prob'], ascending=[True, False], inplace=True)
    if test_verb:
        pass
    else:
        if save_path is not None:
            raw_detection_df.to_parquet(save_path + f'raw_detection_{video_selection}_{model_name}_{frames[0]}_{frames[-1]}.parquet', index=False)
    return raw_detection_df


def detect_instances_parallel(frames, test_verb, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, resolution, save_path):
    instance_properties_dict = {'frame':[], 'centroid-1':[], 'centroid-0':[], 'area':[], 'r':[], 'eccentricity':[],\
                                'prob':[], 'area_bbox':[], 'area_convex':[], 'area_filled':[], 'axis_major_length':[],\
                                'axis_minor_length':[], 'bbox-0':[], 'bbox-1':[], 'bbox-2':[], 'bbox-3':[],\
                                'equivalent_diameter_area':[], 'euler_number':[], 'extent':[], 'feret_diameter_max':[],\
                                'inertia_tensor-0-0':[], 'inertia_tensor-0-1':[], 'inertia_tensor-1-0':[],\
                                'inertia_tensor-1-1':[], 'inertia_tensor_eigvals-0':[], 'inertia_tensor_eigvals-1':[],\
                                'label':[]}

    instance_properties_dict = parallel(
                        joblib.delayed(detect_instances_frame)(instance_properties_dict, frame, get_frame(video, frame, xmin, ymin, xmax, ymax, w, h, resolution, True), model) 
                        for frame in tqdm(frames) )
                
    instance_properties_dict['r'] = np.sqrt(np.array(instance_properties_dict['area'])/np.pi)
    raw_detection_df = pd.DataFrame(instance_properties_dict)
    raw_detection_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
    raw_detection_df['frame'] = raw_detection_df.frame.astype('int')
    raw_detection_df.sort_values(by=['frame', 'prob'], ascending=[True, False], inplace=True)
    if test_verb:
        pass
    else:
        if save_path is not None:
            raw_detection_df.to_parquet(save_path + f'raw_detection_{video_selection}_{model_name}_{frames[0]}_{frames[-1]}.parquet', index=False)
    return raw_detection_df

def test_detection(n_samples, n_frames, nDrops, video_selection, merge_frame, model, model_name, video, xmin, ymin, xmax, ymax, w, h, resolution, save_path):
    print(f"Testing detection on {n_samples} random frames")
    sample_frames = np.sort(np.random.choice(np.arange(0, n_frames, 1, dtype=int), n_samples, replace=False))
    raw_detection_df = detect_instances(frames = sample_frames, test_verb = True, video_selection = video_selection,\
                                        model = model, model_name = model_name, video = video, xmin = xmin, ymin = ymin,\
                                        xmax = xmax, ymax = ymax, w = w, h = h, resolution = resolution, save_path = save_path)

    n_instances_per_frame = raw_detection_df.groupby('frame').count().x.values
    if merge_frame is not None:
        pre_merge_df = raw_detection_df.loc[raw_detection_df.frame < merge_frame]
        pre_merge_frames = pre_merge_df.frame.unique()
        post_merge_df = raw_detection_df.loc[raw_detection_df.frame >= merge_frame]
        post_merge_frames = post_merge_df.frame.unique()
        n_instances_per_frame_pre_merge = pre_merge_df.groupby('frame').count().x.values 
        n_instances_per_frame_post_merge = post_merge_df.groupby('frame').count().x.values
        err_pre_merge = len(np.where(n_instances_per_frame_pre_merge != 50)[0])
        err_post_merge = len(np.where(n_instances_per_frame_post_merge != 49)[0])
        tot_err = err_pre_merge + err_post_merge
        print(f"Frames with spurious effects pre merge:", err_pre_merge , "/", len(pre_merge_frames))
        print(f"Frames with spurious effects post merge:", err_post_merge, "/", len(post_merge_frames))        
    else:
        tot_err = len(np.where(n_instances_per_frame != nDrops)[0])
        print(f"Frames with spurious effects:", tot_err, "/", n_samples)

    fig, ax = plt.subplots(2, 2, figsize = (8, 4))
    ax[0, 0].plot(raw_detection_df.frame.unique(), n_instances_per_frame, '.')
    ax[0, 0].set(xlabel = 'Frame', ylabel = 'N', title = f'N per frame -- {tot_err} / {n_samples} errors')
    ax[0, 1].plot(raw_detection_df.r, '.')
    ax[0, 1].set(xlabel = 'Feature index', ylabel = 'Radius [px]', title = 'Radius of instances detected')
    ax[1, 0].scatter(raw_detection_df.r, raw_detection_df.eccentricity, s=0.1)
    ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title='R-eccentricity correlation')
    ax[1, 1].scatter(raw_detection_df.r, raw_detection_df.prob, s=0.1)
    ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title='R-Probability correlation')
    plt.tight_layout()
    plt.savefig(save_path + f'test_detection.png', dpi = 500)
    plt.close()

    try:
        selected_frame = sample_frames[np.where(raw_detection_df.groupby('frame').count().x.values != nDrops)[0][0]]
        img = get_frame(video, selected_frame, xmin, ymin, xmax, ymax, w, h, resolution, True)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title(f"Example of spurious effect at frame {selected_frame}") 
        ax.imshow(img, cmap='gray')
        for i in range(len(raw_detection_df.loc[raw_detection_df.frame == selected_frame])):
            ax.add_artist(plt.Circle((raw_detection_df.loc[raw_detection_df.frame == selected_frame].x.values[i], raw_detection_df.loc[raw_detection_df.frame == selected_frame].y.values[i]), \
                                        raw_detection_df.loc[raw_detection_df.frame == selected_frame].r.values[i], color='r', fill=False))
        plt.savefig(save_path + f'example_of_spurious_effect.png', dpi = 500)
        plt.close()
    except:
        print('No spurious effect detected')
    return raw_detection_df

def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y


##################################################################################################################
#                                        POSTPROCESSING FUNCTIONS                                                 #
##################################################################################################################

def get_smooth_trajs(trajs, windLen, orderofPoly):
    trajs_ret = trajs.copy()
    trajs_ret['x'] = trajs_ret.groupby('particle')['x'].transform(lambda x: savgol_filter(x, windLen, orderofPoly))
    trajs_ret['y'] = trajs_ret.groupby('particle')['y'].transform(lambda y: savgol_filter(y, windLen, orderofPoly))
    return trajs_ret

def interpolate_trajectory(group):
    interp_method = 'linear'
    all_frames = pd.DataFrame({"frame": range(group["frame"].min(), group["frame"].max() + 1)})
    merged = pd.merge(all_frames, group, on="frame", how="left")
    merged = merged.sort_values(by="frame")
    # Interpolate missing values
    merged["x"]        = merged["x"].interpolate(method = interp_method)
    merged["y"]        = merged["y"].interpolate(method = interp_method)
    merged["r"]        = merged["r"].interpolate(method = interp_method)
    merged["area"]     = merged["area"].interpolate(method = interp_method)
    merged["prob"]     = merged["prob"].interpolate(method = interp_method)
    merged["frame"]    = merged["frame"].interpolate(method = interp_method)
    # ffill() --> Fill NA/NaN values by propagating the last valid observation to next valid.
    merged["particle"] = merged["particle"].ffill()
    merged["color"]    = merged["color"].ffill()
    return merged


##################################################################################################################
#                                        SIMULATION FUNCTIONS                                                    #
##################################################################################################################
"""
def overlap_between_circles(existing_circles, center, radius):
    for existing_center in existing_circles:
        distance = np.linalg.norm(np.array(existing_center) - np.array(center))
        if distance < 2*radius:
            return True
    return False

def initial_instance_positions(nInstances, rFeature, rMax):
    list_of_centers = []
    for i in range(nInstances):
        while True:
            # Generate a random position inside the outer circle 
            theta = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, rMax - rFeature)
            center = (r * np.cos(theta), r * np.sin(theta))
            if not overlap_between_circles(list_of_centers, center, rFeature):
                list_of_centers.append(center)
                break
    return np.array(list_of_centers)
"""

def overlap_between_circles(existing_circles, circle):
    for existing_circle in existing_circles:
        distance = np.linalg.norm(np.array(existing_circle[:2]) - np.array(circle[:2]))
        if distance < existing_circle[2] + circle[2]:
            return True
    return False

def initial_instance_positions(nInstances, radius_of_instances, outer_radius):
    list_of_centers = []
    for i in range(nInstances):
        while True:
            # Generate a random position inside the outer circle 
            theta = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, outer_radius - max(radius_of_instances))
            circle = (r * np.cos(theta), r * np.sin(theta), radius_of_instances[i])
            if not overlap_between_circles(list_of_centers, circle):
                list_of_centers.append(circle)
                break
    return np.array(list_of_centers)[:, :2]

"""
# Function to check for collisions between instances
def handle_instance_collisions_old(pos, instance_radius):
    r_ij_m = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2)
    mask = np.tril(r_ij_m < 2 * instance_radius, k=-1)
    r_ij = (pos[:, np.newaxis] - pos) * mask[:, :, np.newaxis]
    # Normalize displacements
    norms = np.linalg.norm(r_ij, axis=2)
    norms[norms == 0] = 1 # Avoid division by zero
    r_ij_v = r_ij / norms[:, :, np.newaxis]
    # Calculate adjustment factor
    adjustment = (2 * instance_radius - r_ij_m) * mask
    # Apply adjustments to positions
    pos += np.sum(r_ij_v * (adjustment / 2)[:, :, np.newaxis], axis=1)
    pos -= np.sum(r_ij_v * (adjustment / 2)[:, :, np.newaxis], axis=0)
"""

# generalized to different radii
def handle_instance_collisions(pos, instance_radius):
    n = len(pos)
    # Calculate pairwise distances between instances
    r_ij_m = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2)
    # Define a mask to identify instances within the sum of their radii
    mask = np.tril(r_ij_m < (instance_radius[:, np.newaxis] + instance_radius), k=-1)
    # Calculate displacements between instances
    r_ij = (pos[:, np.newaxis] - pos) * mask[:, :, np.newaxis]
    # Normalize displacements
    norms = np.linalg.norm(r_ij, axis=2)
    norms[norms == 0] = 1  # Avoid division by zero
    r_ij_v = r_ij / norms[:, :, np.newaxis]
    # Calculate adjustment factor
    adjustment = ((instance_radius[:, np.newaxis] + instance_radius) - r_ij_m) * mask
    # Apply adjustments to positions
    pos += np.sum(r_ij_v * (adjustment / 2)[:, :, np.newaxis], axis=1)
    pos -= np.sum(r_ij_v * (adjustment / 2)[:, :, np.newaxis], axis=0)
    

def handle_boundary_collisions(pos, outer_radius, instance_radius):
    distances = np.linalg.norm(pos, axis=1)
    # Find indices where distances exceed the boundary
    out_of_boundary_mask = distances > outer_radius - instance_radius
    # Calculate adjustment factor for positions exceeding the boundary
    adjustment = (outer_radius - instance_radius) / distances[out_of_boundary_mask]
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

def handle_boundary_repulsion(pos, repulsion_radius, outer_radius, repulsion_strength, dt):    
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


##################################################################################################################
#                                      SYNTHETIC DATASET GENERATION FUNCTIONS                                    #
##################################################################################################################

def create_gaussian(center, img_width, img_height, sigma, ampl):
    center_x, center_y = center
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2.0 * sigma**2))
    return np.round(ampl*(gaussian / np.max(gaussian))).astype(np.uint8)

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
    
    # add gaussian profile to instances
    image += circles_array 
    if save_path is not None: 
        imwrite(save_path + f'image/frame_{frame}_{height}_resolution.tif', image, compression='zlib')
        imwrite(save_path + f'mask/frame_{frame}_{height}_resolution.tif', mask, compression='zlib')
    return image, mask