#!/usr/bin/env python3
"""
evaluate_coverage_path.py

Standardized metrics for a coverage path, so different CPP algorithms or
parameter choices can be compared apples-to-apples instead of eyeballing
plots. Works on any coverage_waypoints.yaml + room_boundary.yaml + map,
regardless of which generator produced the waypoints.

Usage (single path):
    python3 evaluate_coverage_path.py --map office_map.yaml \\
        --boundary room_boundary.yaml --waypoints coverage_waypoints.yaml

Usage (compare multiple candidates side by side):
    python3 evaluate_coverage_path.py --map office_map.yaml \\
        --boundary room_boundary.yaml \\
        --compare boustrophedon:coverage_waypoints_boustrophedon.yaml \\
                  spiral:coverage_waypoints_spiral.yaml

Metrics reported per path:
  - coverage_pct       : % of drivable room floor actually swept
                          (ground truth, not just "path looks dense")
  - path_length_m      : total travel distance
  - n_waypoints        : waypoint count (proxy for message/plan size)
  - n_turns            : direction changes above a "real turn" threshold
  - sharp_turn_pct     : fraction of turns that are >90 deg (the ones
                          that force the robot to slow down hardest)
  - overlap_pct        : how much swept area is covered MORE than once
                          (redundant driving -- wasted time/battery)
  - est_time_s         : rough time estimate at a given cruise speed,
                          with a per-turn time penalty, so you can
                          compare "similar coverage, faster" tradeoffs
"""

import argparse
import math
import yaml
import numpy as np
import cv2
from scipy import ndimage


def load_map(map_yaml):
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    import os
    map_dir = os.path.dirname(os.path.abspath(map_yaml))
    image_path = meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(map_dir, image_path)
    from PIL import Image
    img = np.array(Image.open(image_path))
    return img, meta


def build_unsafe(img, resolution, robot_radius=0.4, clearance=0.2, cleanup_iter=2):
    free = img > 250
    occ = img < 50
    occ_clean = ndimage.binary_closing(occ, iterations=cleanup_iter)
    occ_clean = ndimage.binary_opening(occ_clean, iterations=cleanup_iter)
    free_clean = ndimage.binary_opening(free, iterations=cleanup_iter)
    inflate_px = max(1, int(round((robot_radius + clearance) / resolution)))
    occ_inflated = ndimage.binary_dilation(occ_clean, iterations=inflate_px)
    return occ_inflated | (~free_clean & ~occ_clean)


def load_boundary_mask(boundary_yaml, resolution, origin, shape):
    with open(boundary_yaml) as f:
        corners = yaml.safe_load(f)["corners"]
    ox, oy = origin[0], origin[1]
    h = shape[0]

    def w2p(x, y):
        return (int(round((x - ox) / resolution)), int(round(h - 1 - (y - oy) / resolution)))

    pts = np.array([w2p(x, y) for x, y in corners], dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def path_length(wps):
    total = 0.0
    for i in range(len(wps) - 1):
        total += math.hypot(wps[i + 1]["x"] - wps[i]["x"], wps[i + 1]["y"] - wps[i]["y"])
    return total


def turn_angles(wps):
    angles = []
    for i in range(1, len(wps) - 1):
        a, b, c = wps[i - 1], wps[i], wps[i + 1]
        v1 = (b["x"] - a["x"], b["y"] - a["y"])
        v2 = (c["x"] - b["x"], c["y"] - b["y"])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        deviation = 180.0 - math.degrees(math.acos(cos_a))
        angles.append(deviation)
    return angles


def coverage_and_overlap(wps, room_mask, unsafe, resolution, origin, height_full, effective_width_m):
    def w2p(x, y):
        return (int(round((x - origin[0]) / resolution)),
                int(round(height_full - 1 - (y - origin[1]) / resolution)))

    thickness_px = max(1, int(round(effective_width_m / resolution))) + 1
    pts = [w2p(w["x"], w["y"]) for w in wps]

    # count how many times each pixel gets swept, to measure overlap
    hit_count = np.zeros(unsafe.shape, dtype=np.uint16)
    for i in range(len(pts) - 1):
        layer = np.zeros(unsafe.shape, dtype=np.uint8)
        cv2.line(layer, pts[i], pts[i + 1], 1, thickness=thickness_px)
        hit_count += layer.astype(np.uint16)

    target = room_mask & (~unsafe)
    swept = hit_count > 0
    covered = swept & target
    coverage_pct = 100.0 * covered.sum() / max(1, target.sum())

    overlapped = (hit_count >= 2) & target
    overlap_pct = 100.0 * overlapped.sum() / max(1, covered.sum())

    return coverage_pct, overlap_pct


def evaluate(map_yaml, boundary_yaml, waypoints_yaml, cruise_speed_mps=0.5,
             turn_time_penalty_s=1.5, sharp_turn_threshold_deg=90.0,
             turn_threshold_deg=15.0):
    img, meta = load_map(map_yaml)
    resolution = meta["resolution"]
    origin = meta["origin"]
    height_full = img.shape[0]

    unsafe = build_unsafe(img, resolution)
    room_mask = load_boundary_mask(boundary_yaml, resolution, origin, unsafe.shape)

    with open(waypoints_yaml) as f:
        wps = yaml.safe_load(f)["waypoints"]

    length_m = path_length(wps)
    angles = turn_angles(wps)
    n_turns = sum(1 for a in angles if a >= turn_threshold_deg)
    n_sharp = sum(1 for a in angles if a >= sharp_turn_threshold_deg)
    sharp_pct = 100.0 * n_sharp / max(1, n_turns)

    # rough effective width: reuse whatever the room-coverage generator
    # would have used (robot_width 0.67 + margin 0.20 = 0.87), matches
    # what generate_room_coverage.py defaults to
    effective_width = 0.87
    coverage_pct, overlap_pct = coverage_and_overlap(
        wps, room_mask, unsafe, resolution, origin, height_full, effective_width)

    est_time_s = (length_m / cruise_speed_mps) + n_turns * turn_time_penalty_s

    return {
        "n_waypoints": len(wps),
        "path_length_m": length_m,
        "coverage_pct": coverage_pct,
        "overlap_pct": overlap_pct,
        "n_turns": n_turns,
        "sharp_turn_pct": sharp_pct,
        "est_time_s": est_time_s,
    }


def print_report(name, m):
    print(f"\n--- {name} ---")
    print(f"  waypoints:        {m['n_waypoints']}")
    print(f"  path length:      {m['path_length_m']:.1f} m")
    print(f"  coverage:         {m['coverage_pct']:.1f}%")
    print(f"  overlap:          {m['overlap_pct']:.1f}% (redundant driving over already-covered floor)")
    print(f"  turns:            {m['n_turns']}  ({m['sharp_turn_pct']:.0f}% of them >90 deg)")
    print(f"  est. time:        {m['est_time_s']:.0f}s  (@0.5 m/s cruise + 1.5s/turn penalty)")


def print_comparison_table(results):
    print("\n" + "=" * 78)
    print(f"{'metric':<18}" + "".join(f"{name:>18}" for name in results))
    print("-" * 78)
    keys = ["coverage_pct", "path_length_m", "n_waypoints", "n_turns", "overlap_pct", "est_time_s"]
    labels = {"coverage_pct": "coverage %", "path_length_m": "path length (m)",
              "n_waypoints": "waypoints", "n_turns": "turns",
              "overlap_pct": "overlap %", "est_time_s": "est. time (s)"}
    for k in keys:
        row = f"{labels[k]:<18}"
        for name in results:
            v = results[name][k]
            row += f"{v:>18.1f}"
        print(row)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--waypoints", default=None, help="single path to evaluate")
    ap.add_argument("--compare", nargs="+", default=None,
                     help="multiple name:path.yaml pairs to compare side by side")
    args = ap.parse_args()

    if args.compare:
        results = {}
        for item in args.compare:
            name, path = item.split(":", 1)
            results[name] = evaluate(args.map, args.boundary, path)
            print_report(name, results[name])
        print_comparison_table(results)
    elif args.waypoints:
        m = evaluate(args.map, args.boundary, args.waypoints)
        print_report(args.waypoints, m)
    else:
        print("Provide either --waypoints (single) or --compare name:path ... (multiple)")


if __name__ == "__main__":
    main()