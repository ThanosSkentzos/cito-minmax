import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def load_file(name):
    with open(name,"rb") as f:
        return np.load(f)

def save_file(name, array):
    with open(name,"wb") as f:
        return np.save(f,array)

def plot_vol(vol, limit=0.1):
    coords = np.argwhere(vol>limit)[::100]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*[[p[dim] for p in coords] for dim in range(coords.shape[-1])])
    plt.show()

def plot_slice(vol, axis, num, show=False):
    axes = [0,1,2]
    axes.pop(axis)
    axes = [axis] + axes # [1,0,2]
    vol = vol.transpose(axes)
    img = vol[num]
    plt.imshow(img, cmap='gray', vmin=vol.min(), vmax=vol.max(), origin='lower')
    if show: plt.show()

def play_slices(r):
    vol = load_file(r)
    print(vol.shape)

    # plot_vol(vol,0.1)
    for ax in range(3):
        plt.figure()
        for i in range(0,vol.shape[ax],1):
            plot_slice(vol,ax,i)
            plt.pause(0.01)
        plt.close()

def plot_lung_vista(lung_mesh, bronchi, jupyter=False):
    BODY = 0.1
    LUNG = 0.3
    BRONCHI = 0.7
    plotter = pv.Plotter()
    plotter.set_background([BODY,BODY,BODY])
    plotter.add_mesh(lung_mesh, opacity=LUNG)
    for tube in bronchi:
        plotter.add_mesh(tube, opacity=BRONCHI)
    plotter.add_axes()
    if jupyter:
        plotter.show(jupyter_backend='trame')
    else:
        plotter.show()
