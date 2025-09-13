# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:48:27 2025

@author: shiqi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import random
import math
from collections import defaultdict

# Configuration
main_folder = "D:/hippo data/bio_image/output_folder_sigma8"
subfolders = sorted([f.path for f in os.scandir(main_folder) if f.is_dir()],
                   key=lambda x: int(os.path.basename(x)))

# Create 3D plot with transparent background
fig = plt.figure(figsize=(16, 12), facecolor='none')
ax = fig.add_subplot(111, projection='3d')

# ======================
# Configure 3D Axes
# ======================
ax.set_facecolor('none')
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Transparent panes
    axis.line.set_color((0.3, 0.3, 0.3, 0.5))  # Gray axis lines

# Set limits and labels
ax.set(xlim=[0, 512], ylim=[0, 512], zlim=[0.5, 5.5],
       xlabel='X Position', ylabel='Y Position', zlabel='Folder Layer')
ax.invert_zaxis()  # Makes higher z-values appear lower
ax.view_init(elev=30, azim=-70)

# ======================
# Create Gray Floor Only
# ======================
x_floor = np.linspace(0, 512, 100)
y_floor = np.linspace(0, 512, 100)
X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
Z_floor = np.full(X_floor.shape, 7.5)  # Bottom of zlim range

# Plot semi-transparent floor
ax.plot_surface(X_floor, Y_floor, Z_floor, color='#d3d3d3',
                alpha=0.1, edgecolor='none', antialiased=True)

# ======================
# Trajectory Processing
# ======================
all_trajectories = []
for z_idx, folder in enumerate(subfolders, 1):
    for file in glob.glob(os.path.join(folder, "*.txt")):
        data = np.loadtxt(file)
        if data.size > 0:
            all_trajectories.append({
                'file': file,
                'folder': z_idx,
                'data': data,
                'duration': np.ptp(data[:, 2])
            })

# Group trajectories by folder
traj_groups = defaultdict(list)
for idx, traj in enumerate(all_trajectories):
    traj_groups[traj['folder']].append(idx)

# Select 25% from each subfolder
highlight_indices = []
for folder_idx, indices in traj_groups.items():
    n_trajs = len(indices)
    if n_trajs == 0:
        continue
    n_select = math.ceil(n_trajs * 0.1)
    highlight_indices.extend(random.sample(indices, n_select))

# ======================
# Plotting Parameters
# ======================
highlight_style = {
    'cmap': plt.cm.ocean_r,
    'line_width': 1.5,
    'point_size': 100,
    'alpha': 0.8
}

dimmed_style = {
    'color': (1, 0.5, 0.2),
    'line_width': 0.3,
    'point_size': 6,
    'alpha': 0.2
}

# Create colormap normalization
max_duration = max(t['duration'] for t in all_trajectories)
norm = plt.Normalize(0, max_duration)

# ======================
# Plot Trajectories
# ======================
for idx, traj in enumerate(all_trajectories):
    data = traj['data']
    x = data[:, 0]
    y = data[:, 1]
    frames = data[:, 2]
    z = np.full_like(x, traj['folder'])
    rel_frames = frames - np.min(frames)
    
    if idx in highlight_indices:
        style = highlight_style
        colors = style['cmap'](norm(rel_frames))
    else:
        style = dimmed_style
        colors = [style['color']] * len(x)
    
    # Plot lines
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                color=colors[i],
                linewidth=style['line_width'],
                alpha=style['alpha'])
    
    # Plot points
    ax.scatter(x, y, z, c=colors,
              s=style['point_size'],
              alpha=style['alpha'],
              edgecolors='none')

# ======================
# Final Configuration
# ======================
ax.set(xlim=[0, 512], ylim=[0, 512], zlim=[0.5, 7.5],
       xlabel='X Position', ylabel='Y Position', zlabel='Folder Layer')
ax.invert_zaxis()
ax.view_init(elev=25, azim=-75)

# Add colorbar for highlighted trajectories
sm = plt.cm.ScalarMappable(cmap=highlight_style['cmap'], norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Time Since Trajectory Start (frames)')

plt.title('3D Trajectories with 25% Highlighted per Folder', pad=20)
plt.tight_layout()

# Save as vector file
fig.savefig("3d_trajectories_vector_test.pdf", format='pdf', 
            bbox_inches='tight', transparent=True)
plt.show()