import joblib
parallel = joblib.Parallel(n_jobs=4, backend='loky', verbose=0)
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from csbdeep.utils import normalize
import skimage
import pandas as pd
import matplotlib.pyplot as plt

##################################################################################################################
#                                        PREPROCESSING FUNCTIONS                                                 #
##################################################################################################################

def get_frame_sharp(cap, frame, x1, y1, x2, y2, w, h, preprocess):
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
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        # sharpen image https://en.wikipedia.org/wiki/Kernel_(image_processing)
        npImage = cv2.filter2D(src=npImage, ddepth=-1, kernel=2*kernel)
        npImage = npImage[y1:y2, x1:x2]
        return normalize(npImage)
    elif not preprocess:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("preprocess must be a boolean")

def get_frame(cap, frame, x1, y1, x2, y2, w, h, preprocess):
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
        npImage = cv2.resize(npImage, (500, 500))
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

def detect_features_frame(feature_properties_dict, frame, img, model):
    segmented_image, dict_test = model.predict_instances(normalize(img), predict_kwargs = {'verbose' : False})

    feature_properties = skimage.measure.regionprops_table(segmented_image, \
                                                            properties=('area', 'area_bbox', 'area_convex', 'area_filled',\
                                                                        'axis_major_length', 'axis_minor_length',\
                                                                        'bbox', 'centroid', 'eccentricity', \
                                                                        'equivalent_diameter_area', 'euler_number', 'extent',\
                                                                        'feret_diameter_max', 'inertia_tensor',\
                                                                        'inertia_tensor_eigvals', 'label'))

    for key in feature_properties.keys():
        feature_properties_dict[key] += list(feature_properties[key])
    feature_properties_dict['prob']  += list(dict_test['prob'])
    feature_properties_dict['frame'] += list(np.ones(len(list(feature_properties['centroid-0'])))*frame)
    return feature_properties_dict


def detect_features(frames, test_verb, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, save_path):
    feature_properties_dict = {'frame':[], 'centroid-1':[], 'centroid-0':[], 'area':[], 'r':[], 'eccentricity':[],\
                                'prob':[], 'area_bbox':[], 'area_convex':[], 'area_filled':[], 'axis_major_length':[],\
                                'axis_minor_length':[], 'bbox-0':[], 'bbox-1':[], 'bbox-2':[], 'bbox-3':[],\
                                'equivalent_diameter_area':[], 'euler_number':[], 'extent':[], 'feret_diameter_max':[],\
                                'inertia_tensor-0-0':[], 'inertia_tensor-0-1':[], 'inertia_tensor-1-0':[],\
                                'inertia_tensor-1-1':[], 'inertia_tensor_eigvals-0':[], 'inertia_tensor_eigvals-1':[],\
                                'label':[]}

    for frame in tqdm(frames):
        img = get_frame(video, frame, xmin, ymin, xmax, ymax, w, h, True)
        feature_properties_dict = detect_features_frame(feature_properties_dict, frame, img, model)
                
    feature_properties_dict['r'] = np.sqrt(np.array(feature_properties_dict['area'])/np.pi)
    raw_detection_df = pd.DataFrame(feature_properties_dict)
    raw_detection_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
    raw_detection_df['frame'] = raw_detection_df.frame.astype('int')
    raw_detection_df.sort_values(by=['frame', 'prob'], ascending=[True, False], inplace=True)
    if test_verb:
        pass
        #raw_detection_df.to_parquet(save_path + f'raw_detection_{video_selection}_{model_name}_test.parquet', index=False)
    else:
        raw_detection_df.to_parquet(save_path + f'raw_detection_{video_selection}_{model_name}_{frames[0]}_{frames[-1]}.parquet', index=False)
    return raw_detection_df

def test_detection(n_samples, n_frames, nDrops, video_selection, model, model_name, video, xmin, ymin, xmax, ymax, w, h, save_path):
    print(f"Testing detection on {n_samples} random frames")
    sample_frames = np.sort(np.random.choice(np.arange(0, n_frames, 1, dtype=int), n_samples, replace=False))
    raw_detection_df = detect_features(frames = sample_frames, test_verb = True, video_selection = video_selection,\
                                       model = model, model_name = model_name, video = video, xmin = xmin, ymin = ymin,\
                                       xmax = xmax, ymax = ymax, w = w, h = h, save_path = save_path)

    n_feature_per_frame = raw_detection_df.groupby('frame').count().x.values
    print(f"Frames with spurious effects:", len(np.where(n_feature_per_frame != nDrops)[0]), "/", len(sample_frames))
    fig, ax = plt.subplots(2, 2, figsize = (8, 4))
    ax[0, 0].plot(raw_detection_df.frame.unique(), n_feature_per_frame, '.')
    ax[0, 0].set(xlabel = 'Frame', ylabel = 'N of droplets', title = 'N of droplets per frame')
    ax[0, 1].plot(raw_detection_df.r, '.')
    ax[0, 1].set(xlabel = 'Feature index', ylabel = 'Radius [px]', title = 'Radius of features detected')
    ax[1, 0].scatter(raw_detection_df.r, raw_detection_df.eccentricity, s=0.1)
    ax[1, 0].set(xlabel = 'Radius [px]', ylabel='Eccentricity', title='R-eccentricity correlation')
    ax[1, 1].scatter(raw_detection_df.r, raw_detection_df.prob, s=0.1)
    ax[1, 1].set(xlabel = 'Radius [px]', ylabel='Probability', title='R-Probability correlation')
    plt.tight_layout()
    plt.savefig(save_path + f'test.png', dpi = 500)
    plt.close()

    try:
        selected_frame = sample_frames[np.where(raw_detection_df.groupby('frame').count().x.values < nDrops)[0][0]]
        img = get_frame(video, selected_frame, xmin, ymin, xmax, ymax, w, h, True)
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


def filter_detection_data(r_min, r_max, raw_detection_df, nDrops):
    # filter found features
    print("Frames with spurious effects pre filtering:", len(np.where(raw_detection_df.groupby('frame').count().x.values != nDrops)[0]), "/", len(raw_detection_df.frame.unique()))
    filtered_df = raw_detection_df.loc[raw_detection_df.r.between(rmin, rmax)]
    filtered_df = filtered_df.groupby('frame').apply(lambda x: x.nlargest(nDrops, 'prob'))
    filtered_df = filtered_df.reset_index(drop=True)
    print("Frames with spurious effects after filtering:", len(np.where(filtered_df.groupby('frame').count().x.values != nDrops)[0]), "/", len(filtered_df.frame.unique()))
    return filtered_df


def interpolate_trajectory(group):
    interp_method = 'quadratic'
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
    merged["particle"] = merged["particle"].ffill()
    merged["color"]    = merged["color"].ffill()
    return merged

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
#                                              SIMULATION FUNCTIONS                                              #
##################################################################################################################

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


def handle_boundary_repulsion(pos, outer_radius, repulsion_radius, repulsion_strength, dt):    
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

def generate_synthetic_image(outer_radius, n_feature, height, width, rmin, rmax, color, gaussian_sigma, gaussian_amplitude, sharp_verb=False):
    # create background image
    image = np.random.randint(65, 75, (height, width), dtype=np.uint8)
    # initialize mask
    mask = np.zeros((height, width), dtype=np.uint8)
    list_of_centers = []
    list_of_distances = []

    for i in range(n_feature):
        while True:
            # Generate a random position inside the outer circle 
            theta = random.uniform(0, 2 * np.pi)
            r = random.randint(0, outer_radius)
            center = (int(width/2 + r * np.cos(theta)), int(height/2 + r * np.sin(theta)))

            feature_radius = random.randint(rmin, rmax)

            if not overlap_between_circles(list_of_centers, center, feature_radius, feature_radius):
                list_of_centers.append([center, feature_radius])
                break
        #color = 110 #(random.randint(0, 255))
        index = i + 1  # Assign unique index (starting from 1)
        # Draw the circle on the image 
        # lineType = 4 for 4-connected line, 8 for 8-connected line, LINE_AA for antialiased line
        cv2.circle(image, center, feature_radius, color, -1, lineType=4) 

        # draw circles as gaussian distribution
        cv2.add(image, create_gaussian(center, width, height, gaussian_sigma, gaussian_amplitude))

        # Draw the circle with its index on the mask
        #draw_circle_with_index(mask, center, feature_radius, index)
        cv2.circle(mask, center, feature_radius, (index), -1)
        
    # Draw the outer circle mimicking the petri dish
    cv2.circle(image, (int(height/2), int(width/2)), int(width/2), 150) 
    cv2.circle(image, (int(height/2), int(width/2)), int(width/2)-4, 150) 
    if sharp_verb:
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    image = cv2.GaussianBlur(image, (5, 5), 2)
    return image, mask
    
@joblib.delayed
def generate_synthetic_image_from_simulation_data_parallel(trajectories, frame, height, width, gaussian_sigma, gaussian_amplitude, color, sharp_verb=False):
    trajs = trajectories.loc[(trajectories.frame == frame), ["x", "y", "r"]]
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
        feature_radius = int(trajs.r.values[i])
        cv2.circle(image, center, feature_radius, color, -1, lineType=8) 
        circles_array += create_gaussian(center, width, height, gaussian_sigma, gaussian_amplitude)
        cv2.circle(mask, center, feature_radius, (index), -1)
    
    if sharp_verb:
        image = cv2.GaussianBlur(image, (5, 5), 2)
        kernel = np.array([[0, -1, 0],
                          [-1, 5,-1],
                          [0, -1, 0]])
        image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    
    # add gaussian profile to droplets
    image += circles_array 
    return image, mask, circles_array