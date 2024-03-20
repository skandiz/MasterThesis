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

training_model = 'modified_2D_versatile_fluo_1000x1000_v4'
train_verb = False
optimize_verb = True


if training_model == 'new':
    n_rays = 32
    use_gpu = False and gputools_available()
    grid = (2, 2)
    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
    )
    print(conf)
    vars(conf)
    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)
    model = StarDist2D(conf, name=f'stardist_trained_from_zero', basedir='models')

elif training_model == 'modified_2D_versatile_fluo_1000x1000':
    if 0:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_1000_resolution/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_1000_resolution/mask/*.tif"))
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, './models/modified_2D_versatile_fluo_1000x1000')
        model = StarDist2D(None, './models/modified_2D_versatile_fluo_1000x1000')
    else:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_1000_resolution/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_1000_resolution/mask/*.tif"))
        model = StarDist2D(None, name = 'modified_2D_versatile_fluo_1000x1000', basedir = 'models')

elif training_model == 'modified_2D_versatile_fluo_1000x1000_v2':
    if 0:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, './models/modified_2D_versatile_fluo_1000x1000_v2')
        model = StarDist2D(None, './models/modified_2D_versatile_fluo_1000x1000_v2')
    else:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model = StarDist2D(None, name = 'modified_2D_versatile_fluo_1000x1000_v2', basedir = 'models')

elif training_model == 'modified_2D_versatile_fluo_1000x1000_v3':
    if 0:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, './models/modified_2D_versatile_fluo_1000x1000_v3')
        model = StarDist2D(None, './models/modified_2D_versatile_fluo_1000x1000_v3')
    else:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model = StarDist2D(None, name = 'modified_2D_versatile_fluo_1000x1000_v3', basedir = 'models')

elif training_model == 'modified_2D_versatile_fluo_1000x1000_v4':
    if 0:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, './models/modified_2D_versatile_fluo_1000x1000_v4')
        model = StarDist2D(None, './models/modified_2D_versatile_fluo_1000x1000_v4')
    else:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model = StarDist2D(None, name = 'modified_2D_versatile_fluo_1000x1000_v4', basedir = 'models')
    
elif training_model == 'modified_2D_versatile_fluo_1000x1000_v5':
    if 0:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, './models/modified_2D_versatile_fluo_1000x1000_v5')
        model = StarDist2D(None, './models/modified_2D_versatile_fluo_1000x1000_v5')
    else:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model = StarDist2D(None, name = 'modified_2D_versatile_fluo_1000x1000_v5', basedir = 'models')

elif training_model == 'modified_2D_versatile_fluo_1000x1000_v6':
    if 0:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_fluo')
        shutil.copytree(model_pretrained.logdir, './models/modified_2D_versatile_fluo_1000x1000_v6')
        model = StarDist2D(None, './models/modified_2D_versatile_fluo_1000x1000_v6')
    else:
        X = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/image/*.tif"))
        Y = sorted(glob("./simulation/synthetic_dataset_100_fps_r_decay_r_gaussian/mask/*.tif"))
        model = StarDist2D(None, name = 'modified_2D_versatile_fluo_1000x1000_v6', basedir = 'models')


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
plt.savefig('./test.png', dpi = 500)
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
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=100, steps_per_epoch=100)

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
    plt.savefig(f'example_X_val[0]_{training_model}.pdf', format = 'pdf')
    plt.close()