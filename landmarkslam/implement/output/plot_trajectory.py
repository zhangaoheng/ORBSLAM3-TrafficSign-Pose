#!/usr/bin/env python3
"""
ORB-SLAM3 Trajectory Visualizer
Reads TUM-format trajectory file and plots it.

Usage:
  python3 plot_trajectory.py <trajectory.txt>
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_tum_trajectory(filepath):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]
    return timestamps, positions, quaternions


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_trajectory.py <trajectory.txt>")
        print("Example: python3 plot_trajectory.py AllFrames_dataset_140500_mono.txt")
        sys.exit(1)

    filepath = sys.argv[1]
    print("Loading:", filepath)
    timestamps, positions, quaternions = load_tum_trajectory(filepath)
    print(f"  Frames: {len(timestamps)}")
    print(f"  Duration: {(timestamps[-1] - timestamps[0]) / 1e9:.2f} s")

    t_sec = (timestamps - timestamps[0]) / 1e9
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # ---- Figure 1: Top view (XY) + Side view (XZ) ----
    fig1 = plt.figure(figsize=(12, 5))

    ax1 = fig1.add_subplot(1, 2, 1)
    sc = ax1.scatter(x, y, c=t_sec, cmap="viridis", s=2, alpha=0.7)
    ax1.plot(x, y, "b-", alpha=0.3, linewidth=0.5)
    ax1.scatter(x[0], y[0], c="green", marker="o", s=100, label="Start", zorder=5)
    ax1.scatter(x[-1], y[-1], c="red", marker="x", s=100, label="End", zorder=5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Top View (X-Y)")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    ax1.legend()
    plt.colorbar(sc, ax=ax1, label="Time (s)")

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.scatter(x, z, c=t_sec, cmap="viridis", s=2, alpha=0.7)
    ax2.plot(x, z, "b-", alpha=0.3, linewidth=0.5)
    ax2.scatter(x[0], z[0], c="green", marker="o", s=100, label="Start")
    ax2.scatter(x[-1], z[-1], c="red", marker="x", s=100, label="End")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Side View (X-Z)")
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")
    ax2.legend()
    plt.colorbar(sc, ax=ax2, label="Time (s)")

    plt.tight_layout()

    # ---- Figure 2: 3D view ----
    fig2 = plt.figure(figsize=(10, 8))
    ax3 = fig2.add_subplot(111, projection="3d")

    colors = plt.cm.viridis(t_sec / t_sec.max())
    for i in range(len(x) - 1):
        ax3.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], alpha=0.6, linewidth=0.5)

    ax3.scatter(x[0], y[0], z[0], c="green", marker="o", s=150, label="Start")
    ax3.scatter(x[-1], y[-1], z[-1], c="red", marker="x", s=150, label="End")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title("ORB-SLAM3 Trajectory (3D)")

    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax3.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax3.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax3.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    ax3.legend()

    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(t_sec)
    plt.colorbar(mappable, ax=ax3, label="Time (s)", shrink=0.7)

    # ---- Figure 3: Position vs time ----
    fig3, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    labels = ["X", "Y", "Z"]
    colors_pos = ["red", "green", "blue"]
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors_pos)):
        ax.plot(t_sec, positions[:, i], color=color, linewidth=0.8)
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Position vs Time")
    plt.tight_layout()

    # ---- Stats ----
    total_dist = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    print(f"\n=== Trajectory Stats ===")
    print(f"  Total path length: {total_dist:.2f} m")
    print(f"  X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Z range: [{z.min():.3f}, {z.max():.3f}]")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
