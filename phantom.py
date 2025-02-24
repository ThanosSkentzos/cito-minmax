# %%
import sklearn.datasets as data
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import draw

random.seed(42)
np.random.seed(42)
N = 201
img = np.zeros((N, N))
vol = np.zeros((N, N, N))
intensity = 20
intensity = 0
half = N // 2

# %%
moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(
    n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25
)
test_data = np.vstack([moons, blobs])
plt.scatter(test_data.T[0], test_data.T[1], color="b")

# %% 2d


def please_draw(func, img, intensity, level, *args, mode=0):
    args = [np.array(i) + random.randint(0, intensity) for i in args]
    rr, cc = func(*args, shape=img.shape)
    if mode == 0:
        img[rr, cc] = level
    elif mode == 1:
        img[rr, cc] += level
    elif mode == -1:
        img[rr, cc] -= level
    else:
        raise ValueError
    return img


def gen_draw(f, img, intensity):
    def draw_shape(*args, **kwargs):
        if len(kwargs.keys()) == 1 and list(kwargs.keys())[0] == "mode":
            return please_draw(f, img, intensity, *args, **kwargs)
        return please_draw(f, img, intensity, *args)

    return draw_shape


ellipse = gen_draw(draw.ellipse, img, intensity)
disk = gen_draw(draw.disk, img, intensity)
rectangle = gen_draw(draw.rectangle, img, intensity)

ellipse(0.5, half, half, 70, 90, mode=0)
# disk(0.8, (60, 100), 30, mode=1)
# rectangle(0.3, (50, 100), (140, 50), mode=-1)
offset = 35
lung_width = 35
lung_length = 42

ellipse(0.3, half, half - offset, lung_length, lung_width, mode=0)
ellipse(0.3, half, half + offset, lung_length, lung_width, mode=0)
ellipse(0.3, 0.8 * half, half, 30, 50, mode=0)
ellipse(0.5, 0.85 * half, 100, 30, 20)


img = img * 255
img[img > 255] = 255
img[img < 0] = 0
img = img.astype(np.int16)

plt.imshow(img, cmap="gray")
plt.colorbar()
plt.close("all")
# %%
shape = draw.ellipsoid(half - 1, half - 1, half - 1, spacing=(1, 1, 1))
shape.shape
vol[shape] = 0.5

# %%
import time

plt.ion()
fig = plt.figure()
ax = fig.add_subplot()
for x in range(vol.shape[0]):
    ax.imshow(shape[x, :, :], cmap="gray")
    # plt.draw()
    plt.pause(0.03)
    ax.clear()
    print(x)


# %%
