import numpy as np
import matplotlib.pyplot as plt
def load_file(name):
    with open(name,"rb") as f:
        return np.load(f)

def save_file(name,array):
    with open(name,"wb") as f:
        return np.save(f,array)

def plot_vol(vol,limit=0.1):
    coords = np.argwhere(vol>limit)[::100]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*[[p[dim] for p in coords] for dim in range(coords.shape[-1])])
    plt.show()

def plot_slice(vol,axis,num, show=False):
    axes = [0,1,2]
    axes.pop(axis)
    axes = [axis] + axes # [1,0,2]
    vol = vol.transpose(axes)
    img = vol[num]
    plt.imshow(img,cmap='gray',vmin=vol.min(),vmax=vol.max())
    if show: plt.show()