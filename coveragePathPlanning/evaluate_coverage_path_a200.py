#!/usr/bin/env python3
"""
evaluate_coverage_path_a200.py

Standardised metrics for a coverage path on the Clearpath A200 (Husky),
so BCD / BSA / parameter variants can be compared apples-to-apples.

TWO FIXES over the TurtleBot evaluate_coverage_path.py
-----------------------------------------------------
1. OBSTACLE MODEL.  The old build_unsafe() ran
       binary_closing(occ, iterations=2)  then  binary_opening(..., 2)
   to clean speckle. That was tuned for turtlebot3_world, whose walls are
   several pixels thick. A SLAM map of the office world has walls ONE
   pixel thick sitting on the image border, and the erosion half of each
   operation deletes them outright:

       raw occupied px 1093 -> closing 8 -> opening 0

   With zero obstacles nothing gets inflated, and the evaluator grades
   coverage against 126.2 m^2 of "drivable floor" that includes the walls
   themselves, instead of the real 106.2 m^2. Every number it printed for
   this map was wrong.

   This version uses an exact Euclidean distance transform and no
   morphology at all, matching generate_room_coverage_bcd_v2.py exactly.
   Speckle is handled by dropping connected components below a size
   threshold, which cannot delete a thin wall.

2. PROVENANCE IS REQUIRED, NOT OPTIONAL.  Coverage is measured by painting
   a stripe of the robot's coverage width along the path. Assume a wider
   stripe than the generator planned and you do not merely mislabel a
   number -- you report floor as covered that was never touched. If the
   waypoints file has no `generated_with` block this script refuses to
   guess unless you pass --assume-defaults.

Usage
    python3 evaluate_coverage_path_a200.py \
        --map office_mapEmpty.yaml \
        --waypoints coverage_waypoints_bcd.yaml

    # side by side
    python3 evaluate_coverage_path_a200.py --map office_mapEmpty.yaml \
        --compare bcd:coverage_waypoints_bcd.yaml \
                  bsa:coverage_waypoints_bsa.yaml

--boundary is optional. With a single room the inflated free space IS the
room, so the polygon adds nothing; pass it only if you want the clip.
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


# --- A200 platform profile -------------------------------------------
# Husky A200: 0.99 x 0.67 m chassis, max 1.0 m/s. CRUISE is what Nav2 is
# actually configured to command, not the datasheet maximum -- overstate
# it and every algorithm looks faster than it can possibly run, which
# breaks comparison against a real timed run.
A200_CHASSIS_WIDTH_M = 0.67
A200_CRUISE_MPS = 0.50
A200_TURN_PENALTY_S = 2.0      # heavier than TB3: more mass to stop/spin

DEFAULTS = dict(robot_radius=0.40, extra_clearance=0.10,
                coverage_width=A200_CHASSIS_WIDTH_M, swath_width=0.57)

MIN_BLOB_PX = 4                # speckle filter, component-based not morphological


def load_map(map_yaml):
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img_rel = meta["image"]
    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(
        os.path.dirname(os.path.abspath(map_yaml)), img_rel)
    img = np.array(Image.open(img_path).convert("L")).astype(np.float64)
    return img, meta


def build_free_and_unsafe(img, meta, resolution, inflation_m):
    """Free mask + inflated-unsafe mask, using an exact EDT.

    Unknown counts as obstacle, matching both Nav2's costmap reading of an
    unmarked map and what the generator planned against.
    """
    negate = int(meta.get("negate", 0))
    free_th = float(meta.get("free_thresh", 0.196))
    occ = img / 255.0 if negate else (255.0 - img) / 255.0
    free = occ < free_th

    obstacle = ~free

    # Component-based speckle removal. Unlike binary_opening this cannot
    # erase a one-pixel-thick wall, because a wall is one large connected
    # component, not a scatter of tiny ones.
    if MIN_BLOB_PX > 1:
        lbl, n = ndimage.label(obstacle)
        if n:
            sizes = np.bincount(lbl.ravel())
            too_small = np.isin(lbl, np.where(sizes < MIN_BLOB_PX)[0])
            obstacle = obstacle & ~too_small

    dist_px = ndimage.distance_transform_edt(~obstacle)
    unsafe = dist_px * resolution < inflation_m
    return free, unsafe


def load_boundary_mask(boundary_yaml, resolution, origin, shape):
    import cv2
    with open(boundary_yaml) as f:
        corners = yaml.safe_load(f)["corners"]
    ox, oy = origin[0], origin[1]
    h = shape[0]

    def w2p(x, y):
        return (int(round((x - ox) / resolution)),
                int(round(h - 1 - (y - oy) / resolution)))

    pts = np.array([w2p(x, y) for x, y in corners], dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def path_length(wps):
    return sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"])
               for a, b in zip(wps[:-1], wps[1:]))


def turn_angles(wps):
    out = []
    for i in range(1, len(wps) - 1):
        a, b, c = wps[i - 1], wps[i], wps[i + 1]
        v1 = (b["x"] - a["x"], b["y"] - a["y"])
        v2 = (c["x"] - b["x"], c["y"] - b["y"])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        out.append(180.0 - math.degrees(math.acos(cos_a)))
    return out


def _dense_samples(wps, resolution, origin, height, shape):
    """Path centreline as an ordered list of pixel samples, ~1 px apart."""
    h, w = shape

    def w2p(x, y):
        return (int(round(height - 1 - (y - origin[1]) / resolution)),
                int(round((x - origin[0]) / resolution)))

    pts = [w2p(p["x"], p["y"]) for p in wps]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for i in range(n + 1):
            t = i / n if n else 0.0
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= r < h and 0 <= c < w and (not out or (r, c) != out[-1]):
                out.append((r, c))
    return out


def coverage_and_overlap(wps, target, resolution, origin, height, width_m):
    """Coverage, and overlap defined as DISTINCT VISITS.

    Overlap must mean "the robot drove over this floor on two separate
    occasions". Accumulating a band per path segment does not measure
    that: consecutive segments share a waypoint, so the disc around every
    waypoint is counted twice, and a lane-end connector's band lies
    entirely inside the two lanes it joins. Validated against a synthetic
    sweep with lane spacing set EQUAL to coverage width -- zero overlap by
    construction -- where the per-segment version reported 16.9%.

    Instead: walk the centreline in order, stamping a disc of half the
    coverage width. A pixel scores a new visit only when the stamping
    sample is far enough along the path from the last one that touched it
    to be a genuinely separate pass.
    """
    h, w = target.shape
    rad_f = (width_m / 2.0) / resolution      # float: 0.335 m -> 6.7 px
    rad = max(1, int(math.ceil(rad_f)))       # integer bounding box only
    gap = max(4 * rad, 8)                     # samples; ~1 sample per pixel

    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    # Float radius test. Rounding rad to 7 px would paint a 0.75 m band for
    # a 0.67 m robot and report ~7% of overlap that is pure quantisation.
    disc = (yy * yy + xx * xx) <= rad_f * rad_f

    last = np.full((h, w), -10 ** 9, dtype=np.int64)
    visits = np.zeros((h, w), np.uint16)

    for i, (r, c) in enumerate(_dense_samples(wps, resolution, origin,
                                              height, target.shape)):
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        sub = disc[r0 - (r - rad):r1 - (r - rad), c0 - (c - rad):c1 - (c - rad)]
        window_last = last[r0:r1, c0:c1]
        fresh = sub & ((i - window_last) > gap)
        visits[r0:r1, c0:c1][fresh] += 1
        window_last[sub] = i

    swept = visits > 0
    covered = swept & target
    coverage_pct = 100.0 * covered.sum() / max(1, target.sum())
    overlap_pct = 100.0 * ((visits >= 2) & target).sum() / max(1, covered.sum())
    return coverage_pct, overlap_pct


def min_clearance(wps, free, resolution, origin, height):
    """Smallest distance from any point ALONG the path to an obstacle.
    Sampled per pixel, not per waypoint: a path can have every waypoint
    clear and still cut a corner through a wall between two of them."""
    dist = ndimage.distance_transform_edt(free) * resolution
    h, w = free.shape

    def w2p(x, y):
        return (int(round(height - 1 - (y - origin[1]) / resolution)),
                int(round((x - origin[0]) / resolution)))

    pts = [w2p(p["x"], p["y"]) for p in wps]
    worst = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for i in range(n + 1):
            t = i / n if n else 0.0
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= r < h and 0 <= c < w:
                worst = min(worst, dist[r, c])
    return worst


def evaluate(map_yaml, waypoints_yaml, boundary_yaml=None,
             cruise=A200_CRUISE_MPS, turn_penalty=A200_TURN_PENALTY_S,
             turn_threshold=15.0, sharp_threshold=90.0,
             assume_defaults=False):
    img, meta = load_map(map_yaml)
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    height = img.shape[0]

    with open(waypoints_yaml) as f:
        doc = yaml.safe_load(f)
    wps = doc["waypoints"]
    prov = doc.get("generated_with")

    if not prov:
        if not assume_defaults:
            sys.exit(
                f"ERROR: {waypoints_yaml} has no 'generated_with' block.\n"
                "  Coverage is measured by painting the robot's coverage\n"
                "  width along the path. Guessing that width fabricates\n"
                "  coverage rather than mismeasuring it, so this script\n"
                "  will not guess.\n"
                "  Regenerate with generate_room_coverage_bcd_v2.py (which\n"
                "  writes provenance), or re-run with --assume-defaults to\n"
                "  accept "
                f"robot_radius={DEFAULTS['robot_radius']}, "
                f"clearance={DEFAULTS['extra_clearance']}, "
                f"coverage_width={DEFAULTS['coverage_width']}.")
        print(f"  NOTE: no provenance in {waypoints_yaml}; assuming A200 "
              "defaults. Numbers are valid ONLY if the generator used them.")
        prov = dict(DEFAULTS)

    robot_radius = float(prov.get("robot_radius", DEFAULTS["robot_radius"]))
    clearance = float(prov.get("extra_clearance", DEFAULTS["extra_clearance"]))
    inflation = float(prov.get("inflation", robot_radius + clearance))
    # The stripe width is what the ROBOT COVERS, not the lane spacing.
    # Lane spacing is narrower (it includes overlap); painting with the
    # spacing under-reports, painting with something wider invents coverage.
    cover_w = float(prov.get("coverage_width",
                             prov.get("swath_width", DEFAULTS["coverage_width"])))
    lane_spacing = float(prov.get("swath_width", DEFAULTS["swath_width"]))

    free, unsafe = build_free_and_unsafe(img, meta, resolution, inflation)
    target = free & ~unsafe
    if boundary_yaml:
        target = target & load_boundary_mask(boundary_yaml, resolution,
                                             origin, target.shape)

    length_m = path_length(wps)
    angles = turn_angles(wps)
    n_turns = sum(1 for a in angles if a >= turn_threshold)
    n_sharp = sum(1 for a in angles if a >= sharp_threshold)

    coverage_pct, overlap_pct = coverage_and_overlap(
        wps, target, resolution, origin, height, cover_w)
    clear_m = min_clearance(wps, free, resolution, origin, height)

    return {
        "platform": prov.get("platform", "a200"),
        "algorithm": prov.get("algorithm", "?"),
        "coverage_width": cover_w,
        "lane_spacing": lane_spacing,
        "inflation": inflation,
        "drivable_m2": float(target.sum() * resolution ** 2),
        "n_waypoints": len(wps),
        "path_length_m": length_m,
        "coverage_pct": coverage_pct,
        "overlap_pct": overlap_pct,
        "n_turns": n_turns,
        "sharp_turn_pct": 100.0 * n_sharp / max(1, n_turns),
        "min_clearance_m": clear_m,
        "est_time_s": length_m / cruise + n_turns * turn_penalty,
    }


def print_report(name, m):
    print(f"\n--- {name} ---")
    print(f"  algorithm:        {m['algorithm']} on {m['platform']}")
    print(f"  drivable floor:   {m['drivable_m2']:.1f} m2 "
          f"(after {m['inflation']:.2f} m inflation)")
    print(f"  waypoints:        {m['n_waypoints']}")
    print(f"  path length:      {m['path_length_m']:.1f} m")
    print(f"  coverage:         {m['coverage_pct']:.1f}%")
    print(f"  overlap:          {m['overlap_pct']:.1f}% "
          "(floor driven over more than once)")
    print(f"  turns:            {m['n_turns']}  "
          f"({m['sharp_turn_pct']:.0f}% of them >90 deg)")
    print(f"  min clearance:    {m['min_clearance_m']:.2f} m "
          "(worst point ALONG the path, not just at waypoints)")
    print(f"  est. time:        {m['est_time_s']:.0f}s  "
          f"(@{A200_CRUISE_MPS} m/s + {A200_TURN_PENALTY_S}s/turn)")
    print(f"  measured as:      {m['coverage_width']:.2f} m coverage width, "
          f"{m['lane_spacing']:.2f} m lane spacing")

    if m["coverage_pct"] < 95.0:
        print("  !! coverage below 95% -- check coverage_debug.png")
    if m["min_clearance_m"] < m["inflation"] - 1e-6:
        print(f"  !! path comes within {m['min_clearance_m']:.2f} m of an "
              f"obstacle, tighter than the {m['inflation']:.2f} m it was "
              "planned for")


def print_comparison(results):
    names = list(results)
    print("\n" + "=" * (20 + 16 * len(names)))
    print(f"{'metric':<20}" + "".join(f"{n:>16}" for n in names))
    print("-" * (20 + 16 * len(names)))
    rows = [("coverage %", "coverage_pct"), ("path length (m)", "path_length_m"),
            ("waypoints", "n_waypoints"), ("turns", "n_turns"),
            ("overlap %", "overlap_pct"), ("min clearance (m)", "min_clearance_m"),
            ("est. time (s)", "est_time_s")]
    for label, key in rows:
        print(f"{label:<20}" + "".join(f"{results[n][key]:>16.2f}" for n in names))
    print("=" * (20 + 16 * len(names)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--boundary", default=None,
                    help="Optional room_boundary.yaml clip.")
    ap.add_argument("--waypoints", default=None)
    ap.add_argument("--compare", nargs="+", default=None,
                    metavar="NAME:PATH",
                    help="Several name:waypoints.yaml pairs.")
    ap.add_argument("--cruise", type=float, default=A200_CRUISE_MPS)
    ap.add_argument("--assume-defaults", action="store_true",
                    help="Proceed even if a waypoints file has no provenance.")
    args = ap.parse_args()

    if args.compare:
        results = {}
        for item in args.compare:
            if ":" not in item:
                sys.exit(f"ERROR: --compare wants NAME:PATH, got '{item}'")
            name, path = item.split(":", 1)
            results[name] = evaluate(args.map, path, args.boundary,
                                     cruise=args.cruise,
                                     assume_defaults=args.assume_defaults)
            print_report(name, results[name])
        if len(results) > 1:
            print_comparison(results)
    elif args.waypoints:
        print_report(os.path.basename(args.waypoints),
                     evaluate(args.map, args.waypoints, args.boundary,
                              cruise=args.cruise,
                              assume_defaults=args.assume_defaults))
    else:
        ap.error("give --waypoints or --compare")


if __name__ == "__main__":
    main()
