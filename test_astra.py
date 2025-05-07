# %%
import astra

astra.test()
# %%
import numpy as np

r1 = [0, 1, 0, 0, 0, 0]
r2 = [0, 1, 1, 0, 0, 1]
r3 = [1, 1, 1, 1, 1, 0]
r4 = [0, 1, 1, 1, 1, 1]
r5 = [0, 0, 1, 1, 0, 0]
r6 = [0, 0, 0, 1, 0, 0]
rows = r1 + r2 + r3 + r4 + r5 + r6
data = np.array(rows).reshape(6, 6)

# %% astra volume
vol = astra.create_vol_geom(6, 6)
phantom_id = astra.data2d.create("-vol", vol)  # initialize variable
phantom_id = astra.data2d.create("-vol", vol, 0)  # or initialize with value
phantom_id = astra.data2d.create("-vol", vol, data)  # or initialize with value
astra.data2d.store(phantom_id, data)  # other way to initialize

# %% astra projection
angles = np.linspace(0, np.pi, 8, False)
det_count = 6
det_width = 1
proj_geom = astra.create_proj_geom("parallel", det_width, det_count, angles)
projector = astra.create_projector("cuda", proj_geom, vol)
# %%
b = astra.creators.create_sino(phantom_id,projector)
b
# %%
import numpy as np
from utils import load_file
bro = load_file("bro.npy")
bro
volume = np.zeros([100,100,100]).astype(float)
volume[*bro.T] = 1 # in xyz
volume = volume.transpose([2,1,0])
vol = astra.create_vol_geom(*volume.shape)
vol
# %%
phantom_id = astra.data3d.create('-vol',vol,volume)

# %% astra projection
angles = np.linspace(0, np.pi, 180,False)
det_count = 100
det_width = 1
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 100, 100, angles)
# proj_geom = astra.create_proj_geom("parallel3d", det_width,det_width,det_count, det_count, angles)
# projector = astra.create_projector("line", proj_geom, vol)
# %%
id,sino = astra.creators.create_sino3d_gpu(volume,proj_geom,vol)
sino = sino.transpose([1,0,2])
sino.shape
#%%
import matplotlib.pyplot as plt
for i in range(180):
    s = sino.flatten()
    plt.imshow(sino[i,:,:],cmap='gray')
    plt.pause(0.001)
# %%
