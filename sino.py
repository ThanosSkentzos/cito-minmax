
# %%
import astra
astra.test()
import numpy as np
from utils import load_file
BRONCHI_LEVEL = 1
LUNG_LEVEL = 0.3

def make_volume(bro_filename,lung_filename,bro_level=BRONCHI_LEVEL,lung_level=LUNG_LEVEL,shape=[100,100,100]):
    bro = load_file(bro_filename)
    lung = load_file(lung_filename)
    volume = np.zeros(shape).astype(float)
    volume[*lung.T] = lung_level            # in xyz
    volume[*bro.T] = bro_level              # in xyz
    volume = volume.transpose([2,1,0])      # in zyx for astra
    return volume
def make_astra_volume(volume):
    vol = astra.create_vol_geom(*volume.shape)
    phantom_id = astra.data3d.create('-vol',vol,volume)
    return vol,phantom_id
def make_projection(type,num_detectors,detector_width,
                    start_radian,end_radian,amount,include_end=False):
    angles = np.linspace(start_radian, end_radian, amount,include_end)
    proj_geom = astra.create_proj_geom(type, detector_width, detector_width, num_detectors, num_detectors, angles)
    return proj_geom
# %%
def make_sinograms():
    volume = make_volume("bro.npy","lung.npy")
    vol,phantom_id = make_astra_volume(volume)
    proj_geom = make_projection("parallel3d",100,1,0,2*np.pi,180,True)
    id,sino = astra.creators.create_sino3d_gpu(volume,proj_geom,vol)
    sino = sino.transpose([1,0,2])
    return sino
#%%
if __name__=="__main__":
    import matplotlib.pyplot as plt
    sino = make_sinograms()
    for i in range(0,180,4):
        s = sino.flatten()
        plt.imshow(sino[i,:,:],cmap='gray')
        plt.pause(0.001)
# %%
