#%%
import numpy as np
import pyvista as pv
from skimage import measure
from utils import save_file,load_file

np.random.seed(42)
# --- Step 1: Create lung volume ---
x, y, z = np.ogrid[-1:1:100j, -1:1:100j, -1:1:100j]
volume = np.zeros((100, 100, 100), dtype=np.float32)

# Right lung
right_lung = (((x + 0.5) / 0.5) ** 2 + (y / 0.35) ** 2 + (z / 0.8) ** 2) < 1
volume[right_lung] = 1

# Left lung with cardiac notch
left_lung = (((x - 0.5) / 0.45) ** 2 + ((y + 0.05) / 0.35) ** 2 + (z / 0.75) ** 2) < 1
volume[left_lung] = 1

# Create lung mesh using marching cubes
verts, faces, _, _ = measure.marching_cubes(volume, level=0.8)
# faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).flatten()
faces_pv = np.array([[3,*faces[i].flatten()] for i in range(faces.shape[0])],dtype=np.int32).flatten()
lung_mesh = pv.PolyData(verts, faces_pv)


# --- Step 2: Generate bronchial tree with controlled main + primary + branching ---
def generate_bronchial_tree(root, depth, volume=volume, angle_variation=0.5, branches_per_node=10):
    points = []
    connections = []
    depths = []
    vol_shape = volume.shape

    def random_rotation(base_direction, angle):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        t = 1 - c
        x, y, z = axis
        R = np.array([
            [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,     t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c    ]
        ])
        return R @ base_direction
    
    def is_inside_lung(pt):
        i, j, k = np.round(pt).astype(int)
        if 0 <= i < vol_shape[0] and 0 <= j < vol_shape[1] and 0 <= k < vol_shape[2]:
            return volume[i, j, k] > 0
        return False
    def distance(start_point,vector):
        step = 2
        is_in = True
        current_pt = start_point
        initial_scale = 0
        scale = 1 
        while True:
            scale *= 1.1
            current_pt = start_point + vector * (initial_scale + scale)
            if not is_inside_lung(current_pt):
                initial_scale += scale/2
                if scale < 4:
                    return initial_scale 
                scale = 1

    def branch(p, d, level):
        if level == 0:
            return
        if level != MAX_DEPTH:
            scale = 0.7*distance(p,d)
        else:
            scale = SCALE_FACTOR
        end = p + d * scale * (level/MAX_DEPTH)
        if is_inside_lung(end):
            points.append(p)
            points.append(end)
            connections.append((len(points) - 2, len(points) - 1))
            depths.append(depth - level + 1)
            for _ in range(branches_per_node):
                angle = np.random.uniform(-angle_variation * np.pi, angle_variation * np.pi)
                new_dir = random_rotation(d, angle)
                new_dir = new_dir / np.linalg.norm(new_dir)

                branch(end, new_dir, level - 1)

    # Main trachea up the Z axis
    main_start = root
    main_end = main_start + np.array([0, 0, 20])
    points.extend([main_start, main_end])
    connections.append((0, 1))
    depths.append(1)

    # Left and Right Primary Bronchi (±X + Z)
    primary_dirs = [
        np.array([-1, 0, 1]),
        np.array([1, 0, 1])
    ]
    for dir in primary_dirs:
        dir = dir / np.linalg.norm(dir)
        branch(main_end, dir, depth)

    return np.array(points), connections, depths

# --- Step 3: Draw cylinders, clip depth >= 3 outside lungs ---
def draw_cylinders(points, connections, depths, radius=1.5):
    tubes = []

    for i, (start_i, end_i) in enumerate(connections):
        start = points[start_i]
        end = points[end_i]
        depth = depths[i]
        direction = end - start
        center = (start + end) / 2
        level_scale = max(0.1, 1 - 0.2*depth)
        tube = pv.Cylinder(center=center,
                        direction=direction,
                        radius=radius * level_scale,
                        height=np.linalg.norm(direction),
                        resolution=16)
        tubes.append(tube)
    return tubes

# --- Create Bronchial Tree -----
MAX_DEPTH = 5
SCALE_FACTOR = 10
BRANCHES = 7
points, conns, depths = generate_bronchial_tree(
    root=np.array([50, 50, 0]),
    depth=MAX_DEPTH,
    volume=volume,
    angle_variation=0.5,
    branches_per_node=BRANCHES
)
bronchi = draw_cylinders(points, conns, depths)
print("Drew cylinders.")
# vol = fill_cylinders(volume,points,conns,depths)
#%%
# # Plot
# plotter = pv.Plotter()
# plotter.set_background([BODY,BODY,BODY])
# plotter.add_mesh(lung_mesh,  opacity=LUNG)
# for tube in bronchi:
#     plotter.add_mesh(tube, opacity=BRONCI)
# plotter.add_axes()
# plotter.show(jupyter_backend='trame')

# %%
grid = pv.StructuredGrid()
grid.dimensions = np.array(volume.shape) + 1  # ASTRA/VTK expects +1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)  # Each voxel = 1x1x1


# %%
import numpy as np
import pyvista as pv
import vtk

# Assume lung_mesh is a closed, watertight PyVista PolyData object
# and volume.shape is (100, 100, 100)

# Step 1: Create the VTK implicit function from the lung mesh
lung_mesh.flip_normals()

lung_vtk = lung_mesh  # Already PolyData
implicit_function = vtk.vtkImplicitPolyDataDistance()
implicit_function.SetInput(lung_vtk)

# Step 2: Create a 3D binary array and fill it using the implicit function
filled_volume = np.zeros(volume.shape, dtype=np.float16)
#%%
# Step 3: Check each voxel center (i, j, k) to see if it's inside the mesh
import os
from tqdm import tqdm
if not os.path.exists("lung.npy"):
    for i in tqdm(range(volume.shape[0])):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                point = [i, j, k]
                if implicit_function.EvaluateFunction(point) < 0:
                    filled_volume[i, j, k] = 0.3
    lung_coords = np.argwhere(filled_volume>0.0)
    save_file("lung.npy",lung_coords)
else:
    lung_coords = load_file("lung.npy")
print("Filled lungs.")
#%%
filled_bronchi = np.zeros(volume.shape, dtype=np.float16)
if not os.path.exists("bro.npy"):
    for tube in tqdm(bronchi):
        xmin,xmax,ymin,ymax,zmin,zmax = tube.bounds
        # tube.flip_normals()
        implicit_function = vtk.vtkImplicitPolyDataDistance()
        implicit_function.SetInput(tube)
        for i in range(int(xmin-1),int(xmax+2)):
            for j in range(int(ymin-1),int(ymax+2)):
                for k in range(int(zmin-1),int(zmax+2)):
                    point = [i, j, k]
                    if filled_bronchi[i,j,k]>0:continue
                    if implicit_function.EvaluateFunction(point) <= 0:
                        filled_bronchi[i, j, k] = 1
    bronchi_coords=np.argwhere(filled_bronchi>0.0)
    save_file("bro.npy",bronchi_coords)
else:
    bronchi_coords=load_file("bro.npy")

print("Filled bronchi.")

# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*[[p[dim] for p in bronchi_coords] for dim in range(bronchi_coords.shape[-1])])
# ax.scatter(*[[p[dim] for p in lung_coords] for dim in range(lung_coords.shape[-1])])
plt.show()

# %%
