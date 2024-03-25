import subprocess as sp
import os
import numpy as np
from tqdm import tqdm
import shutil
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D
from csbdeep.utils import Path, normalize
from tifffile import imread
from glob import glob
from tracking_utils import plot_img_label, random_fliprot, random_intensity_change, augmenter
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) # this goes *before* tf import
import tensorflow as tf
import matplotlib.pyplot as plt
from stardist import _draw_polygons

np.random.seed(42)
lbl_cmap = random_label_cmap()

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


if 0:
    print(get_gpu_memory())
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(fraction = 0.5, total_memory=5000)

model_name = 'modified_2D_versatile_fluo_1000x1000_v3'

train_verb = True
optimize_verb = True

X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))

if model_name == 'new':
    n_rays = 32
    use_gpu = True
    grid = (2, 2)
    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
    )
    model = StarDist2D(conf, name=f'stardist_trained_from_zero', basedir='models')
else:
    if 1:
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, f'./models/{model_name}')
        model = StarDist2D(None, f'./models/{model_name}')
    else:
        model = StarDist2D(None, name = f'{model_name}', basedir = 'models')

print(model.config)

assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x, 1, 99.8, axis = axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
ax[0, 0].imshow(X[0])
ax[0, 1].imshow(Y[0])
ax[1, 0].imshow(X[-1])
ax[1, 1].imshow(Y[-1])
plt.savefig(f'./models/{model_name}/test.png', dpi = 500)
plt.close()

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

if train_verb: 
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=150, steps_per_epoch=100)

if optimize_verb:
    model.optimize_thresholds(X_val, Y_val)

if 1:
    img = X_val[0]
    label, details = model.predict_instances(img, predict_kwargs = {'verbose' : False})
    coord, points, prob = details['coord'], details['points'], details['prob']
    fig, (ax,ax1) = plt.subplots(1, 2, figsize = (10, 5), sharex=True, sharey=True)
    ax.imshow(img, cmap = 'gray')
    ax1.imshow(img, cmap = 'gray')
    _draw_polygons(coord, points, prob, show_dist=True)
    plt.savefig(f'./models/{model_name}/example_X_val[0]_{model_name}.pdf', format = 'pdf')
    plt.close()