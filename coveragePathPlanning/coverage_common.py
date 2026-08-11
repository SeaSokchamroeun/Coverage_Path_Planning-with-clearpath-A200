#!/usr/bin/env python3
"""
coverage_common.py

Algorithm-agnostic post-processing shared by every coverage-path
generator (generate_room_coverage_bcd.py, generate_room_coverage_bsa.py,
and any future candidate -- STC, spiral, whatever comes next).

This exists so that no generator script ever imports from a SIBLING
generator script again. That cross-import (generate_room_coverage_bsa.py
importing from generate_room_coverage.py by hardcoded filename) is
exactly what broke when the baseline got renamed to
generate_room_coverage_bcd.py -- renaming or adding an algorithm should
never be able to break a different algorithm's script.

Split rule: bcd_route_server.py holds the lower-level primitives (unsafe
grid build, A*, connect_points, is_safe, segment_clear) that were already
shared correctly. This module holds the layer above that -- boundary
loading, path cleanup (simplify/smooth/clearance-repair), coverage
verification, and the debug/preview image writers -- which is equally
algorithm-agnostic but had been living inside the boustrophedon-specific
file instead of its own module.

Each generator script keeps ONLY what's actually specific to it:
  generate_room_coverage_bcd.py: build_tracks_two_layer,
      build_tracks_multi_interval, build_room_cell (lane/column logic)
  generate_room_coverage_bsa.py: coarsen, bsa_traverse,
      sequence_to_track (spiral/backtrack logic)
Everything else -- boundary I/O, path cleanup, verification, both debug
images -- lives here exactly once.
"""

import math
import os

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bcd_route_server as brs


# ---------------------------------------------------------------------
# Boundary I/O
# ---------------------------------------------------------------------

def load_boundary(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return [tuple(c) for c in data["corners"]], data.get("frame_id", "map")


def polygon_to_mask(corners_world, resolution, origin, img_shape):
    """Rasterize the world-frame polygon into a pixel mask matching the
    map's coordinate convention (same world_to_px used everywhere else)."""
    h, w = img_shape
    ox, oy = origin[0], origin[1]

    def world_to_px(wx, wy):
        col = int(round((wx - ox) / resolution))
        row = int(round(h - 1 - (wy - oy) / resolution))
        return col, row  # cv2 wants (x, y) = (col, row)

    pts = np.array([world_to_px(x, y) for x, y in corners_world], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def _pgm_path(map_yaml_path, meta):
    map_dir = os.path.dirname(os.path.abspath(map_yaml_path))
    image_path = meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(map_dir, image_path)
    return image_path


# ---------------------------------------------------------------------
# Path cleanup: line-of-sight simplification + safety-checked corner
# smoothing. Both take an `essential` set of pixels that can never be
# removed or moved -- each generator computes its own essential set
# (lane endpoints for BCD, turn/jump points for BSA) and passes it in.
# ---------------------------------------------------------------------

def simplify_path(path_px, unsafe_grid, essential, max_lookahead=25):
    """Greedy line-of-sight shortcutting ('string pulling'), bounded and
    endpoint-protected. Grid-based A* detours naturally zigzag in a
    staircase; this removes that by jumping ahead when the straight
    line is fully clear.

    Two safety rails, both required:
      1. `essential` pixels can never be jumped over -- only connector/
         transit points in between are eligible for removal.
      2. `max_lookahead` bounds how far ahead a jump can be attempted,
         keeping this a LOCAL cleanup, not a global shortest-path search.
    """
    if len(path_px) < 3:
        return path_px

    out = [path_px[0]]
    i = 0
    n = len(path_px)
    while i < n - 1:
        limit = i + 1
        while limit < n - 1 and path_px[limit] not in essential:
            limit += 1
        max_j = min(i + max_lookahead, limit)

        j = max_j
        while j > i + 1 and not brs.segment_clear(unsafe_grid, path_px[i], path_px[j]):
            j -= 1
        out.append(path_px[j])
        i = j
    return out


CORNER_ANGLE_THRESHOLD_DEG = 25.0   # skip smoothing near-straight runs
CORNER_CUT_RATIOS = [0.3, 0.2, 0.1, 0.05]  # tried largest-to-smallest


def _angle_at(a, b, c):
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def smooth_corners(path_px, unsafe_grid, essential):
    """Round sharp direction changes via safety-checked corner cutting --
    but NEVER at an essential point. Full coverage/route integrity takes
    priority over turn smoothness at those points."""
    if len(path_px) < 3:
        return path_px

    out = [path_px[0]]
    i = 1
    while i < len(path_px) - 1:
        a, b, c = path_px[i - 1], path_px[i], path_px[i + 1]

        if b in essential:
            out.append(b)
            i += 1
            continue

        angle = 180.0 - _angle_at(a, b, c)

        if angle < CORNER_ANGLE_THRESHOLD_DEG:
            out.append(b)
            i += 1
            continue

        placed = False
        for ratio in CORNER_CUT_RATIOS:
            p_in = (round(a[0] + (b[0] - a[0]) * (1 - ratio)),
                    round(a[1] + (b[1] - a[1]) * (1 - ratio)))
            p_out = (round(b[0] + (c[0] - b[0]) * ratio),
                     round(b[1] + (c[1] - b[1]) * ratio))
            if (brs.is_safe(unsafe_grid, p_in) and brs.is_safe(unsafe_grid, p_out)
                    and brs.segment_clear(unsafe_grid, a, p_in)
                    and brs.segment_clear(unsafe_grid, p_in, p_out)
                    and brs.segment_clear(unsafe_grid, p_out, c)):
                out.append(p_in)
                out.append(p_out)
                placed = True
                break
        if not placed:
            out.append(b)
        i += 1

    out.append(path_px[-1])
    return out


def repair_tight_clearance(path_px, unsafe_grid, resolution, min_clearance_m=0.15, search_radius_px=6):
    """Catch-all safety pass: any waypoint left with less than
    min_clearance_m to the nearest inflated-obstacle boundary gets
    nudged to the nearest sufficiently-clear pixel within a small local
    search window, provided the line back to its neighbors stays clear."""
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~unsafe_grid) * resolution

    out = list(path_px)
    n_fixed = 0
    n_unfixable = []
    for i in range(len(out)):
        r, c = out[i]
        if dist[r, c] >= min_clearance_m:
            continue

        best = None
        best_d = dist[r, c]
        for dr in range(-search_radius_px, search_radius_px + 1):
            for dc in range(-search_radius_px, search_radius_px + 1):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < unsafe_grid.shape[0] and 0 <= nc < unsafe_grid.shape[1]):
                    continue
                if dist[nr, nc] <= best_d:
                    continue
                prev_ok = brs.segment_clear(unsafe_grid, out[i - 1], (nr, nc)) if i > 0 else True
                next_ok = brs.segment_clear(unsafe_grid, (nr, nc), out[i + 1]) if i < len(out) - 1 else True
                if prev_ok and next_ok:
                    best, best_d = (nr, nc), dist[nr, nc]

        if best is not None and best_d >= min_clearance_m:
            out[i] = best
            n_fixed += 1
        elif best is not None:
            out[i] = best
            n_unfixable.append((out[i], best_d))
        else:
            n_unfixable.append(((r, c), dist[r, c]))

    return out, n_fixed, n_unfixable


# ---------------------------------------------------------------------
# Verification + debug/preview image writers
# ---------------------------------------------------------------------

def verify_coverage(waypoints_world, room_mask, unsafe, resolution, origin,
                     height_full, effective_width_m):
    """Rasterize the robot's actual swept footprint and measure what
    fraction of the room's real free floor area it covers."""
    def w2p(x, y):
        return (int(round((x - origin[0]) / resolution)),
                int(round(height_full - 1 - (y - origin[1]) / resolution)))

    swept = np.zeros(unsafe.shape, dtype=np.uint8)
    thickness_px = max(1, int(round(effective_width_m / resolution))) + 1
    pts = [w2p(w["x"], w["y"]) for w in waypoints_world]
    for i in range(len(pts) - 1):
        cv2.line(swept, pts[i], pts[i + 1], 255, thickness=thickness_px)

    target = room_mask & (~unsafe)
    covered = (swept > 0) & target
    target_area = target.sum()
    covered_area = covered.sum()
    pct = 100.0 * covered_area / max(1, target_area)

    missed = target & (~(swept > 0))
    return pct, missed


def save_coverage_debug(path, room_mask, unsafe, swept_missed, out_path="coverage_debug.png"):
    vis = np.zeros((*unsafe.shape, 3), dtype=np.uint8)
    vis[room_mask & ~unsafe] = (200, 200, 200)
    vis[unsafe] = (60, 60, 60)
    vis[swept_missed] = (0, 0, 255)
    cv2.imwrite(out_path, vis)


def save_preview_plot(waypoints, corners, out_path, algorithm_label, jump_marks=None):
    """Route preview: room boundary + fill, path colored by waypoint
    order (start->end), Start/End markers. `algorithm_label` goes in the
    title so every algorithm's preview is self-identifying at a glance.
    `jump_marks` (optional list of waypoint indices) draws the segment
    INTO that index as a dashed red line -- for algorithms like BSA that
    have a distinct "jump" event a plain sweep doesn't have. Pass None
    (or leave the default) for algorithms with no such concept."""
    xs = [w["x"] for w in waypoints]
    ys = [w["y"] for w in waypoints]

    fig, ax = plt.subplots(figsize=(7, 6))

    bx = [c[0] for c in corners]
    by = [c[1] for c in corners]
    ax.fill(bx, by, color="pink", alpha=0.15, zorder=0)
    ax.plot(bx, by, "r-", linewidth=1.5, label="Room boundary", zorder=1)

    ax.plot(xs, ys, "-", color="gray", linewidth=0.7, alpha=0.6, zorder=2)
    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis", s=18, zorder=3)

    if jump_marks:
        for i in jump_marks:
            if 0 < i < len(xs):
                ax.plot([xs[i - 1], xs[i]], [ys[i - 1], ys[i]],
                        "--", color="red", linewidth=1.3, zorder=3)
        ax.plot([], [], "--", color="red", linewidth=1.3, label="Backtrack jump")

    ax.scatter([xs[0]], [ys[0]], marker="^", color="blue", s=110,
              label="Start", zorder=4, edgecolors="black", linewidths=0.5)
    ax.scatter([xs[-1]], [ys[-1]], marker="s", color="orange", s=110,
              label="End", zorder=4, edgecolors="black", linewidths=0.5)

    plt.colorbar(sc, ax=ax, label="Waypoint order (start -> end)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Coverage path preview - {algorithm_label} ({len(xs)} waypoints)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
