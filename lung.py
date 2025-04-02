#%%
import numpy as np
import pyvista as pv
from skimage import measure

np.random.seed(42)
# --- Step 1: Create lung volume ---
x, y, z = np.ogrid[-1:1:100j, -1:1:100j, -1:1:100j]
volume = np.zeros((100, 100, 100), dtype=np.uint8)

# Right lung
right_lung = (((x + 0.5) / 0.5) ** 2 + (y / 0.35) ** 2 + (z / 0.8) ** 2) < 1
volume[right_lung] = 1

# Left lung with cardiac notch
left_lung = (((x - 0.5) / 0.45) ** 2 + ((y + 0.05) / 0.35) ** 2 + (z / 0.75) ** 2) < 1
volume[left_lung] = 1

# Create lung mesh using marching cubes
verts, faces, _, _ = measure.marching_cubes(volume, level=0.8)
faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).flatten()
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

    def branch(p, d, level):
        if level == 0:
            return
        end = p + d * SCALE_FACTOR * (level/MAX_DEPTH)
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
MAX_DEPTH = 10
SCALE_FACTOR = 10
BRANCHES = 5
points, conns, depths = generate_bronchial_tree(
    root=np.array([50, 50, 0]),
    depth=MAX_DEPTH,
    volume=volume,
    angle_variation=0.5,
    branches_per_node=BRANCHES
)

bronchi = draw_cylinders(points, conns, depths)

# Plot
plotter = pv.Plotter()
plotter.add_mesh(lung_mesh, color='pink', opacity=0.8)
for tube in bronchi:
    plotter.add_mesh(tube, color='purple', opacity=0.9)
plotter.add_axes()
plotter.show(jupyter_backend='trame')

# %%
