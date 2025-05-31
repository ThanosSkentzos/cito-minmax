# %%
import numpy as np
from scipy.ndimage import zoom
import os

import glob
import re

# Regex pattern to extract components
pattern = r"reconstruct_(\d+)_(\d+)_(True|False)_(\d+)\.npy"
# Find matching files
files = glob.glob(os.path.join("reconstructions", "reconstruct_*.npy"))

# Define down/up-sampling ratios
down_ratio = 1 / 4  # simulate 4mm slices over 1mm voxels
up_ratio = 4        # restore to original

def degrade(img2d, axis, r_down, r_up):
    """
    Simulate loss of resolution along 'axis' (0 for vertical, 1 for horizontal),
    then upsample back.
    """
    zoom_factors = [1, 1]
    zoom_factors[axis] = r_down
    low = zoom(img2d, zoom=zoom_factors, order=1)
    zoom_factors[axis] = r_up
    high = zoom(low, zoom=zoom_factors, order=1)
    return high

def make_pairs(volume, plane):
    """
    Generate input-target pairs for a given plane:
        - 'axial': slices along Z  (slice = volume[z, :, :])
        - 'coronal': slices along X (slice = volume[:, :, x])
        - 'sagittal': slices along Y (slice = volume[:, y, :])
    Returns inputs (2*N, H, W) and targets (2*N, H, W)
    """

    volume = 255/volume.max()*volume
    volume = volume.round().astype(np.uint8)
    if plane == 'axial':
        slices = [volume[z, :, :] for z in range(volume.shape[0])]
    elif plane == 'coronal':
        slices = [volume[:, :, x] for x in range(volume.shape[2])]
    elif plane == 'sagittal':
        slices = [volume[:, y, :] for y in range(volume.shape[1])]
    else:
        raise ValueError("Unknown plane")

    nonzero = [s for s in slices if s.mean()>0.1]
    print(len(slices))
    hr = np.stack(nonzero, axis=0)  # (N, H, W)
    
    # degrade along vertical (axis=0) then horizontal (axis=1)
    degV = np.stack([degrade(img, axis=0, r_down=down_ratio, r_up=up_ratio) for img in hr], axis=0)
    degH = np.stack([degrade(img, axis=1, r_down=down_ratio, r_up=up_ratio) for img in hr], axis=0)

    # rotate horizontal-degraded and hr so input blur is always vertical
    degH_rot = np.rot90(degH, k=1, axes=(1, 2))
    hr_rot = np.rot90(hr, k=1, axes=(1, 2))

    inputs = np.concatenate([degV, degH_rot], axis=0)
    targets = np.concatenate([hr, hr_rot], axis=0)

    # Shuffle
    indices = np.random.permutation(inputs.shape[0])
    inputs = inputs[indices]
    targets = targets[indices]
    
    return inputs, targets


for f in files:
    filename = os.path.basename(f)
    match = re.match(pattern, filename)
    if match:
        resolution, radius, fibre_str, n_axis = match.groups()
        resolution = int(resolution)
        radius = int(radius)
        fibre = fibre_str.lower() == 'true'  # Convert 'True'/'False' to boolean
        n_axis = int(n_axis)

        FOLDER = f"sr4zct_data_{resolution}_{radius}_{fibre}_{n_axis}"
        os.makedirs(FOLDER, exist_ok=True)

        # Load 3D phantom volume
        V = np.load(f"reconstructions/reconstruct_{resolution}_{radius}_{fibre}_{n_axis}.npy")  # shape (100,100,100), order (z, y, x)
        # Generate and save pairs for each plane
        for plane in ['axial', 'coronal', 'sagittal']:
            inp, tgt = make_pairs(V, plane)
            np.save(f"{FOLDER}/inputs_{plane}.npy", inp)
            np.save(f"{FOLDER}/targets_{plane}.npy", tgt)
            print(f"{plane.capitalize()}: inputs shape {inp.shape}, targets shape {tgt.shape}")

        # List saved files
        print("\nSaved files:")
        for fn in sorted(os.listdir(f"{FOLDER}")):
            print(f" - {FOLDER}/{fn}")

# %%
