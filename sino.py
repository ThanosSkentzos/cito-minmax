
# %%
import astra
astra.test()

import os
import numpy as np
from utils import load_file, save_file, plot_vol, plot_slice, play_slices
from scipy.ndimage import zoom
from reconstruct import SIRT
import matplotlib.pyplot as plt

BRONCHI_LEVEL = 1
LUNG_LEVEL = 0.3
FACTOR = 4
RESOLUTION = 128

def upscale_z(volume, factor=2, order=1):
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array (X, Y, Z)")
    zoom_factors = (1, 1, factor)  # Only change Z axis
    return zoom(volume, zoom_factors, order=order)

def downsample_z_average(volume, factor=2):
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array (X, Y, Z)")
    x, y, z = volume.shape
    new_z = z // factor
    volume = volume[:, :, :new_z * factor]  # truncate extra slices
    volume = volume.reshape(x, y, new_z, factor)
    return volume.mean(axis=3)

def make_volume(bro_filename,lung_filename,bro_level=BRONCHI_LEVEL,lung_level=LUNG_LEVEL,shape=[RESOLUTION,RESOLUTION,RESOLUTION]):
    bro = load_file(bro_filename)
    lung = load_file(lung_filename)
    volume = np.zeros(shape).astype(float)
    volume[*lung.T] = lung_level            # in xyz
    volume[*bro.T] = bro_level              # in xyz [RESOLUTION,RESOLUTION,RESOLUTION]
    # volume = downsample_z_average(volume,FACTOR)   # in [RESOLUTION,RESOLUTION,50]
    # volume_LD = upscale_z(volume,FACTOR)
    return volume

def make_astra_volume(volume):
    x,y,z = volume.shape
    vol_geom = astra.create_vol_geom(y,z,x)   #takes rows,cols,slices
    phantom_id = astra.data3d.create('-vol',vol_geom,volume)
    return vol_geom,phantom_id

def make_projection(type,num_detectors_X,num_detectors_Y,spacingX,spacingY,
                    start_radian,end_radian,amount,include_end=False):
    angles = np.linspace(start_radian, end_radian, amount,include_end)
    proj_geom = astra.create_proj_geom(type, spacingX, spacingY, num_detectors_X, num_detectors_Y, angles)
    return proj_geom

# %%
def make_sinograms(volume, numX, numY, scale):
    x, y, z = 0, 1, 2
    spacingx = 1
    spacingy = 1

    volume = volume.transpose([z,x,y])
    vol_geom, phantom_id = make_astra_volume(volume)

    proj_geom = make_projection("parallel3d", numX, numY, spacingx, spacingy, 0, 2*np.pi, 180, True)
    proj_id = astra.create_projector('cuda3d', proj_geom=proj_geom, vol_geom=vol_geom)
    sino_id, sino = astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

    # sino = sino.transpose([1,0,2])

    sz, sa, sr = sino.shape
    s = sino.reshape(sz//scale, scale, sa, sr).mean(axis=1)
    proj_geom = make_projection("parallel3d", numX//scale, numY, spacingx, spacingy, 0,2*np.pi, 180, True)

    x,y,z = volume.shape
    new_shape = (x, y, z//scale)
    vol_geom = astra.create_vol_geom(y, z//scale, x)   #takes rows, cols, slices

    proj_id = astra.create_projector('cuda3d', proj_geom=proj_geom, vol_geom=vol_geom)
    sino_id = astra.data3d.create('-sino', proj_geom, s)
    empty_volume = np.zeros(shape=new_shape)

    return vol_geom, empty_volume, proj_id, sino_id

#%% # Test
def test():
    volume = make_volume("bro.npy", "lung.npy")
    for scale in [1, 2, 4, 5, 10]:
        sino = make_sinograms(volume, RESOLUTION, RESOLUTION, scale)
        plt.figure()
        print(sino.shape)
        angles, z, r = sino.shape
        print(z//scale//2)
        plt.imshow(sino[:,z//2,:],cmap='gray')
    plt.show()
    # plt.figure()
    # for i in range(0,180,4):
        # plt.imshow(s[i,:,:],cmap='gray')
        # plt.pause(0.1)
    # plt.show()

#%% # MAIN
def main():
    r = "reconstruct.npy"
    if not os.path.exists(r):
        volume = make_volume("bro.npy","lung.npy")
        vol_geom, empty_volume, proj_id, sino_id = make_sinograms(volume, RESOLUTION, RESOLUTION, scale=1)
        vol_id, vol = SIRT(vol_geom, empty_volume, proj_id, sino_id)
        print("Done.")
        save_file(r,vol)
    else:
        vol = load_file(r)
        print(vol.shape)
    play_slices(r)

if __name__ == "__main__":
    main()

# %%
#NEXT STEPS 
# generate data by downscaling
# train neural network find image2image or use MS-D like paper with  layers
# %%
