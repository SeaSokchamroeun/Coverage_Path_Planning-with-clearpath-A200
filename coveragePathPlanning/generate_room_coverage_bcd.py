#!/usr/bin/env python3
"""
generate_room_coverage.py

Generates coverage_waypoints.yaml for a SINGLE room, using:
  - room_boundary.yaml (the polygon from extract_boundary.py)
  - office_map.pgm/.yaml (for real obstacle avoidance -- furniture
    inside the room still needs to be dodged, the polygon alone
    doesn't know about interior obstacles)

This is the single-room counterpart to bcd_route_server.py's full-map
decomposition: same guidance-track generation and elbow/local-A*
stitching (imported directly, not reimplemented), but scoped to one
polygon instead of running BCD across the whole map. Use this when you
only want CPP in one room, not the full multi-room sweep.

Usage:
    python3 generate_room_coverage.py \\
        --map office_map.yaml --boundary room_boundary.yaml \\
        --out coverage_waypoints.yaml
"""

import argparse
import math
import yaml
import numpy as np
import cv2

import bcd_route_server as brs  # reuse unsafe-grid build, tracks, stitching
from coverage_common import (
    load_boundary, polygon_to_mask, simplify_path, smooth_corners,
    repair_tight_clearance, verify_coverage, save_coverage_debug,
    save_preview_plot, _pgm_path,
)

WAYPOINT_CLEARANCE_M = 0.5   # was 0.15 -- too small for the real 0.45/0.55m
                              # footprint half-extents; a 0.15m repair target
                              # left waypoints inside the robot's own body
                              # envelope near interior furniture. Module-level
                              # so both build_tracks_two_layer() and main()'s
                              # log lines see the same value.

MIN_INTERVAL_M = 1.0  # a disjoint free run shorter than this (in meters,
                       # along a column) is treated as decomposition
                       # noise and dropped; taller runs are real separate
                       # floor zones (e.g. above/below a piece of
                       # furniture) and MUST each get their own pass.
                       # NOTE: raising this to 2.0 was tried and reverted
                       # -- it doesn't merge small intervals into
                       # neighbors, it DROPS the whole column, which
                       # fixed the box-area tangle but skipped a lane
                       # near the shelf (creating both a new coverage
                       # gap and a long diagonal jump to the next
                       # surviving lane). The real fix is reordering
                       # lane visits locally, not filtering which lanes
                       # exist -- see LANE_REORDER_WINDOW below.


def build_tracks_multi_interval(unsafe_grid, cell, resolution, warnings):
    """Like bcd_route_server.cell_guidance_tracks, but handles columns
    with multiple disjoint free intervals correctly. The shared
    function's dedup keeps only the tallest interval per column -- fine
    for filtering single-pixel noise, but wrong here: a column split by
    a piece of furniture has TWO real floor zones (above and below it),
    and discarding the shorter one leaves an entire zone permanently
    uncovered (found via verify_coverage() -- a large solid gap that
    traced back to exactly this column pattern, not just corner noise).
    Returns (waypoints, essential_set) -- essential_set is the ground
    truth for corner-smoothing/simplification protection, computed
    directly alongside generation so it can never drift out of sync
    with what was actually produced (unlike the earlier standalone
    recomputation, which had exactly that bug once already).
    """
    lane_spacing_px = max(1, int(round(brs.SWATH_WIDTH_M / resolution)))

    col_intervals = {}
    for col, rr in zip(cell['cols'], cell['row_ranges']):
        col_intervals.setdefault(col, []).append(rr)

    # merge touching intervals within the same column, keep all others
    # (this is the fix: no "keep tallest only" step)
    for col, ivs in col_intervals.items():
        ivs.sort()
        merged = [ivs[0]]
        for r0, r1 in ivs[1:]:
            lr0, lr1 = merged[-1]
            if r0 <= lr1 + 1:
                merged[-1] = (lr0, max(lr1, r1))
            else:
                merged.append((r0, r1))
        col_intervals[col] = [(r0, r1) for r0, r1 in merged
                               if (r1 - r0) * resolution >= MIN_INTERVAL_M]

    cols_sorted = sorted(cell['cols'])
    track_cols = cols_sorted[::lane_spacing_px]
    if track_cols[-1] != cols_sorted[-1]:
        track_cols.append(cols_sorted[-1])

    essential = set()
    waypoints = []
    for i, col in enumerate(track_cols):
        ivs = col_intervals.get(col) or []
        if not ivs:
            continue
        ivs = sorted(ivs)
        if i % 2 == 1:
            ivs = ivs[::-1]

        col_points = []
        for r0, r1 in ivs:
            top, bottom = (r0, col), (r1, col)
            a, b = (top, bottom) if i % 2 == 0 else (bottom, top)
            col_points.extend([a, b])
            essential.add(a)
            essential.add(b)

        for pt in col_points:
            if waypoints:
                waypoints.extend(brs.connect_points(unsafe_grid, waypoints[-1], pt, warnings))
            waypoints.append(pt)

    return waypoints, essential

# Corner smoothing config: sharp direction changes (boustrophedon lane
# turns) get rounded by corner-cutting, but ONLY if the cut segments
# stay in verified-safe space. Elbow points that were placed specifically
# to clear furniture will naturally fail the safety check at large cut
# ratios and fall back to smaller ones or no smoothing, so this can't
# round a path into an obstacle.
def build_tracks_two_layer(unsafe_grid, cell, resolution, warnings):
    """Two clean horizontal passes: sweep every column's TOP-zone
    interval first (full boustrophedon left-to-right), then every
    column's BOTTOM-zone interval (right-to-left, continuing smoothly
    from wherever the top pass ended instead of transiting back across
    the room). This replaces the per-column interleaving that produced
    a criss-cross "oval" loop around any obstacle splitting the room --
    that version alternated top/bottom on every single lane, repeatedly
    crossing the obstacle's row-band. Sweeping each layer to completion
    before moving to the next avoids that entirely and matches a
    standard single-obstacle boustrophedon reference pattern.
    """
    lane_spacing_px = max(1, int(round(brs.SWATH_WIDTH_M / resolution)))
    # Lane endpoints were being placed at the EXACT edge of free space
    # (the last pixel before the inflated-obstacle boundary) -- which is
    # technically "clear" but leaves ~0 real margin. Found via a sim run
    # that kept stalling at the same spots: checked actual clearance at
    # those waypoints against the map and found 0.05m (one pixel) at the
    # stuck point, and 23/80 waypoints in the same path had <0.15m
    # clearance. Real localization noise / control tracking error can't
    # be absorbed by that, so pull every lane endpoint back by a real
    # buffer instead of hugging the boundary.
    clearance_px = max(1, int(round(WAYPOINT_CLEARANCE_M / resolution)))

    col_intervals = {}
    for col, rr in zip(cell['cols'], cell['row_ranges']):
        col_intervals.setdefault(col, []).append(rr)
    for col, ivs in col_intervals.items():
        ivs.sort()
        merged = [ivs[0]]
        for r0, r1 in ivs[1:]:
            lr0, lr1 = merged[-1]
            if r0 <= lr1 + 1:
                merged[-1] = (lr0, max(lr1, r1))
            else:
                merged.append((r0, r1))
        col_intervals[col] = [(r0, r1) for r0, r1 in merged
                               if (r1 - r0) * resolution >= MIN_INTERVAL_M]

    # Determine the split row PER COLUMN rather than one global average --
    # the obstacle boundary is not flat (confirmed: it slopes ~16 rows
    # across its column span for this map), so a single global r_split
    # places some columns' top/bottom boundary well inside what is
    # locally still free space (or worse, right against the real local
    # edge with zero clearance), which is what was producing the
    # long forced A* detours and the criss-cross near the defect corner.
    # For columns with a real local split, use that column's own gap
    # midpoint. For single-interval columns (no local split), carry
    # forward the nearest real split value by column distance.
    cols_sorted_all = sorted(col_intervals.keys())
    local_split = {}
    for col in cols_sorted_all:
        ivs = col_intervals[col]
        if len(ivs) >= 2:
            local_split[col] = (ivs[0][1] + ivs[1][0]) / 2.0

    if local_split:
        split_cols = sorted(local_split.keys())
        r_split_by_col = {}
        for col in cols_sorted_all:
            nearest = min(split_cols, key=lambda c: abs(c - col))
            r_split_by_col[col] = local_split[nearest]
    else:
        all_r0 = min(r0 for ivs in col_intervals.values() for r0, r1 in ivs)
        all_r1 = max(r1 for ivs in col_intervals.values() for r0, r1 in ivs)
        fallback = (all_r0 + all_r1) / 2.0
        r_split_by_col = {col: fallback for col in cols_sorted_all}

    cols_sorted = sorted(col_intervals.keys())
    track_cols = cols_sorted[::lane_spacing_px]
    if track_cols[-1] != cols_sorted[-1]:
        track_cols.append(cols_sorted[-1])

    def layer_slice(ivs, want_top, r_split):
        """Clip this column's interval(s) to the requested side of
        r_split, then COLLAPSE to a single (min, max) span rather than
        keeping them as separate intervals. Any internal gap (e.g. a
        small obstacle sitting within this layer, not at the layer
        boundary) is left for connect_points' elbow/local-A* detour to
        route around as part of ONE continuous line -- this is what
        keeps the sweep as a single clean pass per column. Pre-splitting
        into separate essential sub-intervals here was producing a
        fragmented, erratic cluster of protected points right where a
        column had its own local obstacle-induced gap (found by
        comparing essential-vs-connector point density in that region:
        mostly disposable connector noise, not real separate zones)."""
        rows = []
        for r0, r1 in ivs:
            if want_top and r0 < r_split:
                rows.append((r0, min(r1, int(r_split))))
            elif not want_top and r1 > r_split:
                rows.append((max(r0, int(r_split) + 1), r1))
        if not rows:
            return None
        return (min(r for r, _ in rows), max(r for _, r in rows))

    essential = set()

    def sweep_layer(cols_order, want_top, start_dir_down):
        pts = []
        for i, col in enumerate(cols_order):
            span = layer_slice(col_intervals.get(col, []), want_top, r_split_by_col[col])
            if span is None:
                continue
            r0, r1 = span
            # pull both ends inward by the clearance buffer, but never
            # past the span's own midpoint (guarantees r0<=r1 even for
            # a span narrower than 2x the buffer)
            mid = (r0 + r1) // 2
            r0 = min(r0 + clearance_px, mid)
            r1 = max(r1 - clearance_px, mid)
            going_down = start_dir_down if i % 2 == 0 else not start_dir_down
            top, bottom = (r0, col), (r1, col)
            a, b = (top, bottom) if going_down else (bottom, top)
            pts.extend([a, b])
            essential.add(a)
            essential.add(b)
        return pts

    top_pts = sweep_layer(track_cols, want_top=True, start_dir_down=True)
    # continue the bottom pass from wherever the top pass ended: reverse
    # column order so the transition is a short local jump, not a
    # cross-room trip back to the start
    bottom_pts = sweep_layer(track_cols[::-1], want_top=False, start_dir_down=True)

    waypoints = []
    for pt in top_pts:
        if waypoints:
            waypoints.extend(brs.connect_points(unsafe_grid, waypoints[-1], pt, warnings))
        waypoints.append(pt)
    for pt in bottom_pts:
        if waypoints:
            waypoints.extend(brs.connect_points(unsafe_grid, waypoints[-1], pt, warnings))
        waypoints.append(pt)

    return waypoints, essential


    """Recompute the same top/bottom lane-endpoint pixels
    bcd_route_server.cell_guidance_tracks() generates internally, so we
    can protect them from simplification. These are the actual coverage
    points -- everything else in the generated path is a connector/
    transit point that exists only to get between them safely."""
    lane_spacing_px = max(1, int(round(brs.SWATH_WIDTH_M / resolution)))
    col_to_range = {}
    for col, rr in zip(cell['cols'], cell['row_ranges']):
        if col in col_to_range:
            er0, er1 = col_to_range[col]
            nr0, nr1 = rr
            if not (nr1 < er0 - 1 or nr0 > er1 + 1):
                col_to_range[col] = (min(er0, nr0), max(er1, nr1))
            elif (nr1 - nr0) > (er1 - er0):
                col_to_range[col] = rr
        else:
            col_to_range[col] = rr

    cols_sorted = sorted(cell['cols'])  # NOT deduped -- must match
    # bcd_route_server.cell_guidance_tracks() exactly, including its use
    # of the raw (duplicate-containing) cols list for the stride, or the
    # selected track_cols silently diverge and "essential" protection
    # doesn't actually match the real track points.
    track_cols = cols_sorted[::lane_spacing_px]
    if track_cols[-1] != cols_sorted[-1]:
        track_cols.append(cols_sorted[-1])

    essential = set()
    for col in track_cols:
        r0, r1 = col_to_range[col]
        essential.add((r0, col))
        essential.add((r1, col))
    return essential


def build_room_cell(unsafe, room_mask):
    """Build the same {'cols': [...], 'row_ranges': [...]} structure
    bcd_route_server's cell_guidance_tracks() expects, but restricted to
    the room polygon intersected with real free space (so interior
    furniture is still respected)."""
    passable_in_room = (~unsafe) & room_mask
    cols_with_data = np.where(passable_in_room.any(axis=0))[0]
    if len(cols_with_data) == 0:
        raise RuntimeError("No free space inside the room polygon -- check "
                            "the boundary/seed used in extract_boundary.py.")

    cell = {'cols': [], 'row_ranges': []}
    for col in cols_with_data:
        rows = np.where(passable_in_room[:, col])[0]
        # a column could have multiple disjoint free runs if furniture
        # splits it -- record each run separately, same convention BCD
        # uses, so cell_guidance_tracks's per-column dict handles it
        # (keeps the taller one on collision via the existing dedup fix)
        splits = np.where(np.diff(rows) > 1)[0]
        runs = np.split(rows, splits + 1)
        for run in runs:
            cell['cols'].append(int(col))
            cell['row_ranges'].append((int(run[0]), int(run[-1])))
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="office_map.yaml")
    ap.add_argument("--boundary", default="room_boundary.yaml")
    ap.add_argument("--out", default="coverage_waypoints.yaml")
    ap.add_argument("--spawn", nargs=2, type=float, default=[0.0, 0.0],
                     help="World-frame point the robot starts from, used "
                          "to pick which end of the coverage sweep to "
                          "connect to first.")
    ap.add_argument("--swath-width", type=float, default=None,
                     help="Distance between coverage lanes in meters. "
                          "Defaults to --robot-width + --margin (full "
                          "lawnmower-style coverage, no gaps). Override "
                          "only if you deliberately want sparser passes.")
    ap.add_argument("--robot-width", type=float, default=0.67,
                     help="Husky A200 chassis width in meters (default 0.67, "
                          "per the platform's physical footprint).")
    ap.add_argument("--margin", type=float, default=0.20,
                     help="Extra coverage margin added to robot width, in "
                          "meters (default 0.20 -- your requested 20cm).")
    args = ap.parse_args()

    if args.swath_width is None:
        args.swath_width = args.robot_width + args.margin
        print(f"Swath width defaulted to robot_width({args.robot_width}) + "
              f"margin({args.margin}) = {args.swath_width}m")

    old = brs.SWATH_WIDTH_M
    brs.SWATH_WIDTH_M = args.swath_width
    print(f"Lane spacing set to {args.swath_width}m (bcd_route_server default was {old}m)")

    meta = brs.load_map_meta(args.map)
    resolution = meta["resolution"]
    origin = meta["origin"]

    pgm_path = _pgm_path(args.map, meta)
    unsafe, height_full = brs.build_unsafe_grid(pgm_path, resolution)

    corners, frame_id = load_boundary(args.boundary)
    room_mask = polygon_to_mask(corners, resolution, origin, unsafe.shape)

    print(f"Room polygon rasterized: {room_mask.sum()} px "
          f"({room_mask.sum() * resolution**2:.2f} m^2 bbox-filled)")

    cell = build_room_cell(unsafe, room_mask)
    n_cols = len(set(cell['cols']))
    area_m2 = sum((r1 - r0 + 1) for r0, r1 in cell['row_ranges']) * resolution ** 2
    print(f"Free+in-room cell: {n_cols} columns, {area_m2:.2f} m^2 "
          f"(after removing interior obstacles)")

    warnings = []
    track_px, essential = build_tracks_two_layer(unsafe, cell, resolution, warnings)
    if not track_px:
        raise RuntimeError("No guidance tracks generated -- room may be "
                            "narrower than the robot's swath spacing.")

    def px_to_world(row, col):
        wx = origin[0] + col * resolution
        wy = origin[1] + (height_full - 1 - row) * resolution
        return wx, wy

    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = int(round(height_full - 1 - (wy - origin[1]) / resolution))
        return row, col

    # connect from spawn to the nearer end of the track sequence, same
    # pattern build_full_route uses per-cell in the multi-room version
    spawn_px = world_to_px(*args.spawn)
    d_start = math.hypot(track_px[0][0] - spawn_px[0], track_px[0][1] - spawn_px[1])
    d_end = math.hypot(track_px[-1][0] - spawn_px[0], track_px[-1][1] - spawn_px[1])
    if d_end < d_start:
        track_px = track_px[::-1]

    approach = brs.connect_points(unsafe, spawn_px, track_px[0], warnings)
    full_px = [spawn_px] + approach + track_px

    if warnings:
        print(f"WARNING: {len(warnings)} stitching warning(s):")
        for w in warnings:
            print(f"  - {w}")

    n_before = len(full_px)
    full_px = simplify_path(full_px, unsafe, essential)
    n_simplified = len(full_px)
    full_px = smooth_corners(full_px, unsafe, essential)
    print(f"Path cleanup: {n_before} -> {n_simplified} (line-of-sight simplify, "
          f"{len(essential)} lane endpoints protected) -> {len(full_px)} (corner smoothing)")

    full_px, n_fixed, unfixable = repair_tight_clearance(full_px, unsafe, resolution)
    print(f"Clearance repair: nudged {n_fixed} point(s) with <{WAYPOINT_CLEARANCE_M}m clearance to safer positions")
    if unfixable:
        print(f"  WARNING: {len(unfixable)} point(s) could not reach {WAYPOINT_CLEARANCE_M}m clearance "
              f"even after nudging -- genuinely tight geometry, check these in RViz:")
        for (r, c), d in unfixable:
            print(f"    pixel({r},{c}) best achievable clearance={d:.3f}m")

    world_pts = [px_to_world(r, c) for r, c in full_px]
    waypoints = []
    for i, (x, y) in enumerate(world_pts):
        if i < len(world_pts) - 1:
            nx, ny = world_pts[i + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = waypoints[-1]["yaw"] if waypoints else 0.0
        waypoints.append({"x": float(x), "y": float(y), "yaw": float(yaw)})

    with open(args.out, "w") as f:
        yaml.dump({"frame_id": frame_id, "waypoints": waypoints}, f)

    print(f"Generated {len(waypoints)} waypoints -> {args.out}")

    pct, missed = verify_coverage(waypoints, room_mask, unsafe, resolution,
                                   origin, height_full, args.swath_width)
    print(f"\nCoverage check: {pct:.1f}% of drivable room floor swept "
          f"by the robot's effective {args.swath_width:.2f}m coverage width")
    if pct < 95.0:
        print("  WARNING: this is below 95% -- gaps likely remain. "
              "See coverage_debug.png (red = missed floor).")
    save_coverage_debug(args.out, room_mask, unsafe, missed)
    print("  Saved coverage_debug.png (red = floor the robot's footprint never reaches)")

    preview_path = "coverage_preview_bcd.png"
    save_preview_plot(waypoints, corners, preview_path, "Boustrophedon")
    print(f"  Saved {preview_path}")


if __name__ == "__main__":
    main()