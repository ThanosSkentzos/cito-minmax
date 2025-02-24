# %%
import matplotlib.pyplot as plt
import skimage.draw as draw
import numpy as np

N = 200
half = N // 2

plane = np.zeros((N, N, N))
x = np.linspace(1, N, N)
y = np.linspace(1, N, N)
z = np.linspace(1, N, N)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
M = [X, Y, Z]


# %%
def add_shape(plane, vol, level=0.5, mode=0):
    """
    c are the coordinates for the center of the ellipsoid
    s are the scales (a,b,c) with which we divide the terms
    level is the pixel level
    mode is an integer
        1: add
        0: set
       -1: substract
    """
    if mode == 0:
        plane[vol] = level
    elif mode == 1:
        plane[vol] += level
    elif mode == -1:
        plane[vol] -= level
    else:
        raise ValueError
    return plane


def ellipsoid(c, s, noise=0):
    noise_offset = np.zeros((3))
    noise_scale = np.ones((3))
    if noise:
        noise_offset = np.random.randint(-noise, noise, size=len(c))
        noise_scale = 1 + np.random.randint(-noise, noise, size=len(c)) / 100
    ellipsoid = (
        sum(
            [
                (M[i] - c[i] + noise_offset[i]) ** 2 / (noise_scale[i] * s[i]) ** 2
                for i in range(len(s))
            ]
        )
        <= 1
    )
    return ellipsoid


x_offset = 20
centers = [half - x_offset, half, half]
centers2 = [half + x_offset, half, half]
scales = [16, 25, 49]
level = 0.5
noise=10

plane = add_shape(plane, ellipsoid(centers, scales,noise), 0.5)
plane = add_shape(plane, ellipsoid(centers2, scales))


vol = np.argwhere(plane)
# %% PLOT 3D
sample = vol[::200]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*[sample[:, i] for i in range(sample.shape[1])])
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
ax.set_zlim([0, 200])
fig.set_facecolor("black")
ax.set_facecolor("black")
ax.grid(False)

plt.show()

# %% PLOT 2d
plt.ion()
from tqdm import tqdm

fig = plt.figure()
ax = fig.add_subplot()
step = 5
for x in tqdm(range(0, plane.shape[0], step)):
    ax.imshow(plane[x, :, :], cmap="gray")
    # plt.draw()
    plt.pause(0.03)
    ax.clear()
    # print(x,end="\r")
for y in tqdm(range(0, plane.shape[1], step)):
    ax.imshow(plane[:, y, :], cmap="gray")
    # plt.draw()
    plt.pause(0.03)
    ax.clear()
    # print(y,end="\r")
for z in tqdm(range(0, plane.shape[2], step)):
    ax.imshow(plane[:, :, z], cmap="gray")
    # plt.draw()
    plt.pause(0.03)
    ax.clear()
    # print(z,end="\r")
# %%
