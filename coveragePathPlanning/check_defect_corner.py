#!/usr/bin/env python3
"""
check_defect_corner.py

The one comparison metric evaluate_coverage_path.py doesn't produce:
behavior at the known Room 2 pixel-scale defect corner (world ~(10.45,
8.5), pixel ~(70, 510)). For each candidate path, reports:

  - map clearance AT the defect coordinate itself (algorithm-independent
    ground truth -- should be near zero, that's the pinch)
  - the nearest waypoint to the defect, its distance, and its clearance
  - minimum waypoint clearance within a radius of the defect
  - how many waypoints enter that radius at all

Reading the result: if BOTH algorithms place their nearest local
waypoints at comparably tight clearance (or both are forced to stand
off), that independently confirms a genuine map-scale pinch. If the
baseline sits tight there but BSA routes it cleanly (or vice versa),
the algorithm itself contributes to the defect.

Usage:
    python3 check_defect_corner.py --map office_map.yaml \\
        --compare boustrophedon:coverage_waypoints.yaml \\
                  bsa:coverage_waypoints_bsa.yaml \\
        [--world 10.454 8.663] [--radius 1.0]
"""

import argparse
import math

import numpy as np
import yaml
from scipy import ndimage

import bcd_route_server as brs
from coverage_common import _pgm_path


def find_true_pinch(clearance_m, resolution, origin, height_full, approx_world, search_radius_m):
    """The --world coordinate is only an approximate anchor (e.g. a
    midpoint between two flagged waypoints). Search a window around it
    for the actual local-minimum-clearance pixel and report THAT as the
    ground-truth defect location, rather than trusting the guess --
    a run against the real Room 2 map showed these can differ by
    several pixels and several hundredths of a meter of clearance."""
    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = int(round(height_full - 1 - (wy - origin[1]) / resolution))
        return row, col

    ar, ac = world_to_px(*approx_world)
    rad_px = max(1, int(round(search_radius_m / resolution)))
    r0, r1 = max(0, ar - rad_px), min(clearance_m.shape[0], ar + rad_px + 1)
    c0, c1 = max(0, ac - rad_px), min(clearance_m.shape[1], ac + rad_px + 1)
    window = clearance_m[r0:r1, c0:c1]
    fr, fc = np.unravel_index(np.argmin(window), window.shape)
    return (r0 + int(fr), c0 + int(fc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="office_map.yaml")
    ap.add_argument("--compare", nargs="+", required=True,
                    help="name:waypoints.yaml pairs")
    ap.add_argument("--world", nargs=2, type=float, default=[10.454, 8.663],
                    help="Approximate anchor coordinate to search around "
                         "(default: midpoint of the two known "
                         "silently-skipped waypoints 23 and 46). The "
                         "script auto-locates the true worst-clearance "
                         "pixel near this anchor rather than trusting it "
                         "directly -- see --search-radius.")
    ap.add_argument("--search-radius", type=float, default=0.3,
                    help="Window (m) around --world to search for the "
                         "true local-minimum-clearance pixel (default 0.3m)")
    ap.add_argument("--radius", type=float, default=1.0,
                    help="Analysis radius around the CONFIRMED defect (m)")
    args = ap.parse_args()

    meta = brs.load_map_meta(args.map)
    resolution = meta["resolution"]
    origin = meta["origin"]
    pgm_path = _pgm_path(args.map, meta)
    unsafe, height_full = brs.build_unsafe_grid(pgm_path, resolution)
    clearance_m = ndimage.distance_transform_edt(~unsafe) * resolution

    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = int(round(height_full - 1 - (wy - origin[1]) / resolution))
        return row, col

    dr, dc = find_true_pinch(clearance_m, resolution, origin, height_full,
                             args.world, args.search_radius)
    dx = origin[0] + dc * resolution
    dy = origin[1] + (height_full - 1 - dr) * resolution
    print(f"Anchor (approximate): world({args.world[0]:.3f}, {args.world[1]:.3f})")
    print(f"Confirmed pinch (searched +/-{args.search_radius:.2f}m around anchor): "
          f"world({dx:.3f}, {dy:.3f}) = pixel({dr},{dc})")
    print(f"Map clearance at the confirmed pinch: {clearance_m[dr, dc]:.3f}m "
          f"(ground truth, algorithm-independent)")

    for item in args.compare:
        name, path = item.split(":", 1)
        with open(path) as f:
            wps = yaml.safe_load(f)["waypoints"]

        best_i, best_d = None, None
        in_radius = []
        for i, w in enumerate(wps):
            d = math.hypot(w["x"] - dx, w["y"] - dy)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
            if d <= args.radius:
                r, c = world_to_px(w["x"], w["y"])
                in_radius.append((i, d, clearance_m[r, c]))

        print(f"\n--- {name} ({path}) ---")
        w = wps[best_i]
        r, c = world_to_px(w["x"], w["y"])
        print(f"  nearest waypoint:      #{best_i} at ({w['x']:.3f}, {w['y']:.3f}), "
              f"{best_d:.3f}m from defect, clearance {clearance_m[r, c]:.3f}m")
        if in_radius:
            i_min = min(in_radius, key=lambda t: t[2])
            print(f"  waypoints within {args.radius:.1f}m: {len(in_radius)}")
            print(f"  tightest of those:     #{i_min[0]} "
                  f"clearance {i_min[2]:.3f}m ({i_min[1]:.3f}m from defect)")
        else:
            print(f"  waypoints within {args.radius:.1f}m: 0 -- this path "
                  f"stands off the defect corner entirely")


if __name__ == "__main__":
    main()