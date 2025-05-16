
# %%
import astra
astra.test()
import numpy as np
from utils import load_file
from scipy.ndimage import zoom
BRONCHI_LEVEL = 1
LUNG_LEVEL = 0.3
FACTOR = 4
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

def make_volume(bro_filename,lung_filename,bro_level=BRONCHI_LEVEL,lung_level=LUNG_LEVEL,shape=[100,100,100]):
    bro = load_file(bro_filename)
    lung = load_file(lung_filename)
    volume = np.zeros(shape).astype(float)
    volume[*lung.T] = lung_level            # in xyz
    volume[*bro.T] = bro_level              # in xyz [100,100,100]
    # volume = downsample_z_average(volume,FACTOR)   # in [100,100,50]
    # volume_LD = upscale_z(volume,FACTOR)
    return volume
def make_astra_volume(volume):
    x,y,z = volume.shape
    vol = astra.create_vol_geom(y,z,x)   #takes rows,cols,slices
    phantom_id = astra.data3d.create('-vol',vol,volume)
    return vol,phantom_id
def make_projection(type,num_detectors_X,num_detectors_Y,spacingX,spacingY,
                    start_radian,end_radian,amount,include_end=False):
    angles = np.linspace(start_radian, end_radian, amount,include_end)
    proj_geom = astra.create_proj_geom(type, spacingX, spacingY, num_detectors_X, num_detectors_Y, angles)
    return proj_geom
# %%
def make_sinograms(volume,numX,numY,scale):
    x,y,z=0,1,2
    volume = volume.transpose([z,x,y])
    vol,phantom_id = make_astra_volume(volume)
    spacingx = 1
    spacingy = 1
    proj_geom = make_projection("parallel3d",numX,numY,spacingx,spacingy,0,2*np.pi,180,True)
    id,sino = astra.creators.create_sino3d_gpu(volume,proj_geom,vol)
    sino = sino.transpose([1,0,2])
    angles,z,r = sino.shape
    s = sino.reshape(angles,z//scale,scale,r).mean(axis=2)
    return s
#%%
def main():
    import matplotlib.pyplot as plt
    volume = make_volume("bro.npy","lung.npy")
    # volume = volume[:-2,:-1,:]
    for scale in [1,2,4,5,10]:
        sino= make_sinograms(volume,100,100,scale)
        plt.figure()
        print(sino.shape)
        angles,z,r = sino.shape
        print(z//scale//2)
        plt.imshow(sino[:,z//2,:],cmap='gray')
    plt.show()
    # plt.figure()
    # for i in range(0,180,4):
        # plt.imshow(s[i,:,:],cmap='gray')
        # plt.pause(0.1)
    # plt.show()
if __name__=="__main__":
    main()
# %%
#NEXT STEPS 
# generate data by downscaling
# train neural network find image2image or use MS-D like paper with 100 layers
# %%
