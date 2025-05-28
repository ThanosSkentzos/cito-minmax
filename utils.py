import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def load_file(name):
    with open(name,"rb") as f:
        return np.load(f)

def save_file(name, array):
    with open(name,"wb") as f:
        return np.save(f,array)

def plot_vol(vol, limit=0.1,stride=1000):
    coords = np.argwhere(vol>limit)[::stride]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*[[p[dim] for p in coords] for dim in range(coords.shape[-1])])
    plt.show()

def plot_bronchi(bro,stride=1000):
    coords = bro[::stride]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*[[p[dim] for p in coords] for dim in range(coords.shape[-1])])

def plot_sino(sino,stride=4):
    sino = sino.transpose([1,0,2])
    for angle in range(0,sino.shape[0],stride):
        plt.imshow(sino[angle],cmap='gray',origin='lower')
        plt.pause(0.001)

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
        for i in range(0,vol.shape[ax],10):
            plot_slice(vol,ax,i)
            plt.pause(0.001)
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

from scipy import interpolate
from scipy.ndimage import gaussian_filter

def resample_img_ax0(img, org_res=0.7421875, org_overlap=0, exp_res=3, exp_overlap=2, keep_dim=True, num_points_overlap=9):
    (w, h) = img.shape
    length = org_res + (w -1) * (org_res - org_overlap)
    num_new = 1 + int((length-exp_res) / (exp_res - exp_overlap))
    x_old = np.linspace(org_res/2, length-org_res/2, w, endpoint=True)
    x_new = np.linspace(exp_res/2, length-exp_res/2, num_new, endpoint=True)
    if keep_dim:
        image_new = np.zeros((w,h))
    else:
        image_new = np.zeros((len(x_new), h))
    vmin, vmax = img.min(), img.max()
    # go for every column
    for i in range(h):
        f = interpolate.interp1d(x_old, img[:,i], kind='linear',fill_value='extrapolate')
        y_new = f(x_new)
        if exp_overlap > 0:
            for j in range(num_points_overlap // 2):
                # average the pixel values
                x_new_l = x_new - exp_res / num_points_overlap * (j + 1)
                x_new_r = x_new + exp_res / num_points_overlap * (j + 1)
                y_new_l = f(x_new_l)
                y_new_r = f(x_new_r)
                y_new += (y_new_l + y_new_r)
            y_new /=num_points_overlap
        if keep_dim:
            f2 = interpolate.interp1d(x_new, y_new, kind='linear',fill_value='extrapolate')
            y_new = f2(x_old)
        image_new[:,i] = y_new
    image_new[image_new<vmin] = vmin
    image_new[image_new>vmax] = vmax
    return image_new

def resample_img_ax1(img,org_res=0.7421875,org_overlap=0,exp_res=3,exp_overlap=2,keep_dim=True, num_points_overlap=9):
    (w, h) = img.shape
    length = org_res + (h -1) * (org_res - org_overlap)
    num_new = 1 + int((length-exp_res) / (exp_res - exp_overlap))
    x_old = np.linspace(org_res/2, length-org_res/2, h, endpoint=True)
    x_new = np.linspace(exp_res/2, length-exp_res/2, num_new, endpoint=True)
    if keep_dim:
        image_new = np.zeros((w,h))
    else:
        image_new = np.zeros((w,len(x_new)))
    vmin, vmax = img.min(), img.max()
    # go for every row
    for i in range(w):
        f = interpolate.interp1d(x_old, img[i], kind='linear',fill_value='extrapolate')
        y_new = f(x_new)
        if exp_overlap > 0:
            for j in range(num_points_overlap // 2):
                # average the pixel values
                x_new_l = x_new - exp_res / num_points_overlap * (j + 1)
                x_new_r = x_new + exp_res / num_points_overlap * (j + 1)
                y_new_l = f(x_new_l)
                y_new_r = f(x_new_r)
                y_new += (y_new_l + y_new_r)
            y_new /= num_points_overlap
        if keep_dim:
            f2 = interpolate.interp1d(x_new, y_new, kind='linear',fill_value='extrapolate')
            y_new = f2(x_old)
        image_new[i] = y_new
    image_new[image_new<vmin] = vmin
    image_new[image_new>vmax] = vmax
    return image_new


# first blur
def resample_img_ax0_with_blurr(img, org_res=0.7421875, org_overlap=0, exp_res=3, exp_overlap=2, keep_dim=True, num_points_overlap=9, blurr_sigma=0):
    (w, h) = img.shape
    length = org_res + (w -1) * (org_res - org_overlap)
    num_new = 1 + int((length-exp_res) / (exp_res - exp_overlap))
    x_old = np.linspace(org_res/2, length-org_res/2, w, endpoint=True)
    x_new = np.linspace(exp_res/2, length-exp_res/2, num_new, endpoint=True)
    if keep_dim:
        image_new = np.zeros((w,h))
    else:
        image_new = np.zeros((len(x_new), h))
    vmin, vmax = img.min(), img.max()
    if blurr_sigma > 0:
        for i in range(h):
            img[:, i] = gaussian_filter(img[:, i], sigma=blurr_sigma)

    # go for every column
    for i in range(h):
        f = interpolate.interp1d(x_old, img[:,i], kind='linear',fill_value='extrapolate')
        y_new = f(x_new)
        if exp_overlap > 0:
            for j in range(num_points_overlap // 2):
                # average the pixel values
                x_new_l = x_new - exp_res / num_points_overlap * (j + 1)
                x_new_r = x_new + exp_res / num_points_overlap * (j + 1)
                y_new_l = f(x_new_l)
                y_new_r = f(x_new_r)
                y_new += (y_new_l + y_new_r)
            y_new /=num_points_overlap

        if keep_dim:
            f2 = interpolate.interp1d(x_new, y_new, kind='linear',fill_value='extrapolate')
            y_new = f2(x_old)

        image_new[:,i] = y_new

    image_new[image_new<vmin] = vmin
    image_new[image_new>vmax] = vmax
    return image_new

def resample_img_ax1_with_blurr(img,org_res=0.7421875,org_overlap=0,exp_res=3,exp_overlap=2,keep_dim=True, num_points_overlap=9, blurr_sigma=0):
    (w, h) = img.shape
    length = org_res + (h -1) * (org_res - org_overlap)
    num_new = 1 + int((length-exp_res) / (exp_res - exp_overlap))
    x_old = np.linspace(org_res/2, length-org_res/2, h, endpoint=True)
    x_new = np.linspace(exp_res/2, length-exp_res/2, num_new, endpoint=True)
    if keep_dim:
        image_new = np.zeros((w,h))
    else:
        image_new = np.zeros((w,len(x_new)))
    vmin, vmax = img.min(), img.max()
    if blurr_sigma > 0:
        for i in range(w):
            img[i] = gaussian_filter(img[i], sigma=blurr_sigma)

    # go for every row
    for i in range(w):
        x_old = np.arange(org_res/2,length-org_res/4,org_res-org_overlap)
        x_new = np.arange(exp_res/2,length-exp_res/4,exp_res-exp_overlap)
        f = interpolate.interp1d(x_old, img[i], kind='linear',fill_value='extrapolate')
        y_new = f(x_new)
        if exp_overlap > 0:
            for j in range(num_points_overlap // 2):
                # average the pixel values
                x_new_l = x_new - exp_res / num_points_overlap * (j + 1)
                x_new_r = x_new + exp_res / num_points_overlap * (j + 1)
                y_new_l = f(x_new_l)
                y_new_r = f(x_new_r)
                y_new += (y_new_l + y_new_r)
            y_new /= num_points_overlap

        if keep_dim:
            f2 = interpolate.interp1d(x_new, y_new, kind='linear',fill_value='extrapolate')
            y_new = f2(x_old)
        
        image_new[i] = y_new
    image_new[image_new<vmin] = vmin
    image_new[image_new>vmax] = vmax
    return image_new
