#!/usr/bin/env python3
"""
plot_coverage.py

Quick static sanity-check plot: overlays the room boundary polygon
(room_boundary.yaml) with the generated coverage waypoints
(coverage_waypoints.yaml), so you can visually confirm the swath
pattern looks right BEFORE spinning up the full sim/Nav2 stack.

Usage:
    python3 plot_coverage.py \
        --boundary room_boundary.yaml \
        --waypoints coverage_waypoints.yaml \
        --out coverage_preview.png
"""

import argparse
import yaml
import matplotlib
matplotlib.use("Agg")  # no display needed, just save a png
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default="room_boundary.yaml")
    ap.add_argument("--waypoints", default="coverage_waypoints.yaml")
    ap.add_argument("--out", default="coverage_preview.png")
    args = ap.parse_args()

    with open(args.boundary) as f:
        boundary = yaml.safe_load(f)["corners"]
    with open(args.waypoints) as f:
        wps = yaml.safe_load(f)["waypoints"]

    bx = [c[0] for c in boundary]
    by = [c[1] for c in boundary]

    wx = [w["x"] for w in wps]
    wy = [w["y"] for w in wps]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Room boundary
    ax.plot(bx, by, color="red", linewidth=2, label="Room boundary")
    ax.fill(bx, by, color="red", alpha=0.05)

    # Coverage path, colored by order (start=blue -> end=yellow) so you
    # can see the actual traversal sequence, not just the lane pattern.
    sc = ax.scatter(wx, wy, c=range(len(wx)), cmap="viridis", s=12, zorder=3)
    ax.plot(wx, wy, color="gray", linewidth=0.6, alpha=0.6, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Waypoint order (start -> end)")

    # Start / end markers
    ax.scatter([wx[0]], [wy[0]], color="blue", s=80, marker="^", zorder=4, label="Start")
    ax.scatter([wx[-1]], [wy[-1]], color="orange", s=80, marker="s", zorder=4, label="End")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Coverage path preview ({len(wx)} waypoints)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved -> {args.out}")
    print(f"Boundary corners: {len(boundary)}")
    print(f"Waypoints: {len(wx)}")


if __name__ == "__main__":
    main()