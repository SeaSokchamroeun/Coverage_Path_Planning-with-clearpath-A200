#!/usr/bin/env python3
"""
generate_room_coverage_bsa.py

Backtracking Spiral Algorithm (BSA) coverage generator for a SINGLE room.
This is the comparison candidate against the boustrophedon baseline in
generate_room_coverage.py -- it swaps ONLY the guidance-track generation
(the spiral traversal below) and reuses everything else unchanged, by
direct import:

  from bcd_route_server:        build_unsafe_grid, connect_points,
                                segment_clear, load_map_meta
  from generate_room_coverage:  load_boundary, polygon_to_mask,
                                simplify_path, smooth_corners,
                                repair_tight_clearance, verify_coverage,
                                save_coverage_debug

Same map, same boundary, same robot params, same clearance repair, same
YAML output schema ({frame_id, waypoints:[{x,y,yaw}]}) -- so the output
is drop-in compatible with run_coverage.py and evaluate_coverage_path.py,
and any metric difference vs the baseline traces to the algorithm alone.

NEVER writes to coverage_waypoints.yaml -- default output is
coverage_waypoints_bsa.yaml. The working baseline is untouched.

Algorithm (BSA, Gonzalez et al. 2005, adapted to this pipeline):
  1. Coarsen the passable-in-room mask into square cells of swath width
     (robot_width + margin = 0.87m by default, matching the baseline's
     lane spacing exactly -- the fairness knob).
  2. A coarse cell is coverable iff it contains at least one passable
     pixel with >= WAYPOINT_CLEARANCE_M (0.15m) real clearance; its
     waypoint is that cell's maximum-clearance pixel. Cells that have
     free floor but can't meet 0.15m anywhere are DROPPED AND LOGGED
     with world coords -- this is the explicit stress-test signal for
     the known x~10.45m defect corner: if BSA also can't stand there,
     it prints exactly where and why.
  3. Spiral traversal: from the start cell, repeatedly move to the
     first unvisited coverable 4-neighbor in priority order
     [turn-left, straight, turn-right] relative to current heading --
     this hugs the visited/obstacle boundary and grows a spiral.
  4. Dead end (no unvisited neighbor): BFS over already-visited cells
     to the NEAREST one that still has an unvisited coverable neighbor,
     jump there (the physical route for the jump is produced later by
     the shared connect_points elbow/local-A* stitching), and re-spiral
     from that point. This is BSA's "backtracking".
  5. Terminate when no reachable unvisited coverable cell remains.

Straight runs are collapsed by the shared simplify_path pass: only
turn cells, jump endpoints, and route endpoints are marked essential,
mirroring how the baseline protects lane endpoints.

Usage:
    python3 generate_room_coverage_bsa.py \\
        --map office_map.yaml --boundary room_boundary.yaml \\
        --spawn 1.303262 1.481998 \\
        --out coverage_waypoints_bsa.yaml
"""

import argparse
import math
from collections import deque

import numpy as np
import yaml
from scipy import ndimage

import bcd_route_server as brs
from coverage_common import (
    load_boundary, polygon_to_mask, simplify_path, smooth_corners,
    repair_tight_clearance, verify_coverage, save_coverage_debug,
    save_preview_plot, _pgm_path,
)

WAYPOINT_CLEARANCE_M = 0.15  # same target as the baseline's lane-endpoint
                             # buffer and repair_tight_clearance threshold

# 4-neighbor headings in (dr, dc); rotate helpers for the spiral priority
def _left_of(h):   return (-h[1], h[0])
def _right_of(h):  return (h[1], -h[0])


def coarsen(passable, clearance_m, cell_px):
    """Divide the passable-in-room mask into cell_px x cell_px coarse
    cells. The lattice is anchored to the passable area's bounding box
    with symmetric overhang (NOT to the map origin) so edge cells hug
    the free-space edges the same way the baseline's first/last lanes
    do -- origin-aligned cells produced thin dropped slivers along every
    wall, an artifact of the grid, not of BSA.

    Returns:
      coverable : 2D bool (coarse grid) -- cell has a >=0.15m-clear pixel
      rep       : dict (cr,cc) -> (row,col) waypoint pixel: the
                  sufficiently-clear passable pixel CLOSEST TO THE CELL
                  CENTER. For edge cells this lands right at the
                  clearance boundary next to the wall -- the exact
                  analog of the baseline pulling lane endpoints to the
                  free edge then buffering 0.15m inward.
      dropped   : list of ((cr,cc), (row,col), best_clr, free_frac) for
                  cells with free floor but no >=0.15m-clear pixel.
                  free_frac lets the caller separate genuine pinch
                  signals (substantial free area, nowhere clear enough
                  to stand) from edge-quantization noise (tiny sliver).
    """
    H, W = passable.shape
    rows_any = np.where(passable.any(axis=1))[0]
    cols_any = np.where(passable.any(axis=0))[0]
    rmin, rmax = int(rows_any[0]), int(rows_any[-1])
    cmin, cmax = int(cols_any[0]), int(cols_any[-1])
    n_cr = max(1, int(math.ceil((rmax - rmin + 1) / cell_px)))
    n_cc = max(1, int(math.ceil((cmax - cmin + 1) / cell_px)))
    # symmetric overhang: lattice span >= extent, centered on the bbox
    r_off = rmin - (n_cr * cell_px - (rmax - rmin + 1)) // 2
    c_off = cmin - (n_cc * cell_px - (cmax - cmin + 1)) // 2

    coverable = np.zeros((n_cr, n_cc), dtype=bool)
    rep = {}
    dropped = []

    for cr in range(n_cr):
        for cc in range(n_cc):
            r0 = max(0, r_off + cr * cell_px)
            c0 = max(0, c_off + cc * cell_px)
            r1 = min(r_off + (cr + 1) * cell_px, H)
            c1 = min(c_off + (cc + 1) * cell_px, W)
            if r1 <= r0 or c1 <= c0:
                continue
            block_pass = passable[r0:r1, c0:c1]
            n_pass = int(block_pass.sum())
            if n_pass == 0:
                continue  # no free floor at all -- not a stress signal
            block_clr = clearance_m[r0:r1, c0:c1].copy()
            block_clr[~block_pass] = -1.0
            clear_enough = block_clr >= WAYPOINT_CLEARANCE_M
            ctr = ((r1 - r0 - 1) / 2.0, (c1 - c0 - 1) / 2.0)
            if clear_enough.any():
                rr, cc2 = np.where(clear_enough)
                k = int(np.argmin((rr - ctr[0]) ** 2 + (cc2 - ctr[1]) ** 2))
                rep[(cr, cc)] = (r0 + int(rr[k]), c0 + int(cc2[k]))
                coverable[cr, cc] = True
            else:
                fr, fc = np.unravel_index(np.argmax(block_clr), block_clr.shape)
                pix = (r0 + int(fr), c0 + int(fc))
                free_frac = n_pass / float((r1 - r0) * (c1 - c0))
                dropped.append(((cr, cc), pix, float(block_clr[fr, fc]), free_frac))
    return coverable, rep, dropped


def bsa_traverse(coverable, start_cell):
    """Core BSA: spiral with backtracking over the coarse grid.
    Returns (sequence, jump_before) where sequence is the ordered list
    of visited coarse cells and jump_before is the set of sequence
    indices that were reached via a backtrack jump (not an adjacent
    spiral step)."""
    visited = {start_cell}
    seq = [start_cell]
    jump_before = set()
    cur = start_cell
    heading = (0, 1)  # arbitrary initial; first step fixes it

    def free(cell):
        cr, cc = cell
        return (0 <= cr < coverable.shape[0] and 0 <= cc < coverable.shape[1]
                and coverable[cr, cc] and cell not in visited)

    # make the initial heading point at an actual free neighbor if any
    for h in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        if free((cur[0] + h[0], cur[1] + h[1])):
            heading = h
            break

    while True:
        # spiral step: hug the visited/blocked boundary on the left
        moved = False
        for h in (_left_of(heading), heading, _right_of(heading)):
            nxt = (cur[0] + h[0], cur[1] + h[1])
            if free(nxt):
                visited.add(nxt)
                seq.append(nxt)
                cur, heading = nxt, h
                moved = True
                break
        if moved:
            continue

        # dead end: BFS over visited cells to the nearest one with an
        # unvisited coverable neighbor
        bfs = deque([cur])
        seen = {cur}
        parent = {}
        target, target_nb, target_h = None, None, None
        while bfs:
            cell = bfs.popleft()
            for h in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nb = (cell[0] + h[0], cell[1] + h[1])
                if free(nb):
                    target, target_nb, target_h = cell, nb, h
                    break
                if nb in visited and nb not in seen:
                    seen.add(nb)
                    parent[nb] = cell
                    bfs.append(nb)
            if target is not None:
                break
        if target is None:
            return seq, jump_before  # complete: nothing reachable left

        visited.add(target_nb)
        jump_before.add(len(seq))
        seq.append(target_nb)
        cur, heading = target_nb, target_h


def sequence_to_track(seq, jump_before, rep, unsafe, warnings):
    """Convert the coarse-cell sequence into the same (track_px,
    essential) pair build_tracks_two_layer returns, stitching every
    consecutive pair through the shared connect_points chain. Essential
    = turn cells + jump endpoints + route endpoints, so simplify_path
    can collapse straight runs exactly like the baseline collapses
    connector noise between lane endpoints."""
    essential = set()
    n = len(seq)
    for i in range(n):
        is_end = i == 0 or i == n - 1
        is_jump_edge = i in jump_before or (i + 1) in jump_before
        if is_end or is_jump_edge:
            essential.add(rep[seq[i]])
            continue
        d_in = (seq[i][0] - seq[i - 1][0], seq[i][1] - seq[i - 1][1])
        d_out = (seq[i + 1][0] - seq[i][0], seq[i + 1][1] - seq[i][1])
        if d_in != d_out:
            essential.add(rep[seq[i]])

    track = []
    for i, cell in enumerate(seq):
        pt = rep[cell]
        if track:
            track.extend(brs.connect_points(unsafe, track[-1], pt, warnings))
        track.append(pt)
    return track, essential


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="office_map.yaml")
    ap.add_argument("--boundary", default="room_boundary.yaml")
    ap.add_argument("--out", default="coverage_waypoints_bsa.yaml")
    ap.add_argument("--spawn", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--swath-width", type=float, default=None,
                    help="Coarse cell size in meters. Defaults to "
                         "robot_width + margin -- MUST match the "
                         "baseline's lane spacing for a fair comparison.")
    ap.add_argument("--robot-width", type=float, default=0.67)
    ap.add_argument("--margin", type=float, default=0.20)
    args = ap.parse_args()

    if args.out == "coverage_waypoints.yaml":
        raise SystemExit("Refusing to overwrite the boustrophedon baseline "
                         "output (coverage_waypoints.yaml). Pick another --out.")

    if args.swath_width is None:
        args.swath_width = args.robot_width + args.margin
        print(f"Coarse cell size defaulted to robot_width({args.robot_width}) + "
              f"margin({args.margin}) = {args.swath_width}m "
              f"(matches baseline lane spacing)")

    meta = brs.load_map_meta(args.map)
    resolution = meta["resolution"]
    origin = meta["origin"]

    pgm_path = _pgm_path(args.map, meta)
    unsafe, height_full = brs.build_unsafe_grid(pgm_path, resolution)

    corners, frame_id = load_boundary(args.boundary)
    room_mask = polygon_to_mask(corners, resolution, origin, unsafe.shape)
    passable = (~unsafe) & room_mask
    print(f"Room polygon rasterized: {room_mask.sum()} px; "
          f"passable in room: {passable.sum()} px "
          f"({passable.sum() * resolution**2:.2f} m^2)")

    # real metric clearance everywhere, for cell reps + the drop log
    clearance_m = ndimage.distance_transform_edt(~unsafe) * resolution

    cell_px = max(1, int(round(args.swath_width / resolution)))
    coverable, rep, dropped = coarsen(passable, clearance_m, cell_px)
    print(f"Coarse grid: {cell_px}px ({cell_px * resolution:.2f}m) cells, "
          f"{int(coverable.sum())} coverable")

    def px_to_world(row, col):
        wx = origin[0] + col * resolution
        wy = origin[1] + (height_full - 1 - row) * resolution
        return wx, wy

    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = int(round(height_full - 1 - (wy - origin[1]) / resolution))
        return row, col

    if dropped:
        signal = [d for d in dropped if d[3] >= 0.25]
        noise = [d for d in dropped if d[3] < 0.25]
        if signal:
            print(f"STRESS SIGNAL: {len(signal)} coarse cell(s) have substantial "
                  f"free floor (>=25% of the cell) but no pixel with >= "
                  f"{WAYPOINT_CLEARANCE_M}m clearance -- BSA cannot place a "
                  f"waypoint there:")
            for (crcc, (r, c), best, frac) in signal:
                wx, wy = px_to_world(r, c)
                print(f"    cell {crcc} @ pixel({r},{c}) = world({wx:.3f}, {wy:.3f})"
                      f"  best clearance {best:.3f}m  ({frac*100:.0f}% free)")
        if noise:
            print(f"({len(noise)} additional sliver cell(s) <25% free dropped "
                  f"as edge quantization -- not a pinch signal)")

    if not coverable.any():
        raise RuntimeError("No coverable coarse cells -- check the boundary "
                           "and map inputs.")

    # start at the coverable cell nearest the spawn point
    spawn_px = world_to_px(*args.spawn)
    spawn_cell = (spawn_px[0] // cell_px, spawn_px[1] // cell_px)
    cov_cells = np.argwhere(coverable)
    dists = np.hypot(cov_cells[:, 0] - spawn_cell[0], cov_cells[:, 1] - spawn_cell[1])
    start_cell = tuple(cov_cells[int(np.argmin(dists))])

    seq, jump_before = bsa_traverse(coverable, start_cell)
    n_jumps = len(jump_before)
    print(f"BSA traversal: {len(seq)}/{int(coverable.sum())} coverable cells "
          f"visited, {n_jumps} backtrack jump(s)")
    if len(seq) < coverable.sum():
        print(f"    WARNING: {int(coverable.sum()) - len(seq)} coverable "
              f"cell(s) unreachable from the start cell by 4-adjacency -- "
              f"they will show as missed floor in the coverage debug image.")

    # jump landing points in world coords, captured now while they're
    # still exact rep-pixel positions -- essential points survive
    # simplify_path/smooth_corners untouched, so these stay accurate
    # enough to locate in the final waypoint list for the preview plot.
    jump_world_targets = [px_to_world(*rep[seq[i]]) for i in jump_before]

    warnings = []
    track_px, essential = sequence_to_track(seq, jump_before, rep, unsafe, warnings)

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
          f"{len(essential)} turn/jump points protected) -> {len(full_px)} "
          f"(corner smoothing)")

    full_px, n_fixed, unfixable = repair_tight_clearance(full_px, unsafe, resolution)
    print(f"Clearance repair: nudged {n_fixed} point(s) with <0.15m clearance")
    if unfixable:
        print(f"  WARNING: {len(unfixable)} point(s) below 0.15m clearance even "
              f"after nudging -- genuinely tight geometry:")
        for (r, c), d in unfixable:
            wx, wy = px_to_world(r, c)
            print(f"    pixel({r},{c}) = world({wx:.3f}, {wy:.3f}) "
                  f"best clearance={d:.3f}m")

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
        yaml.dump({"frame_id": frame_id, "waypoints": waypoints,
                   "algorithm": "bsa"}, f)
    print(f"Generated {len(waypoints)} waypoints -> {args.out}")

    pct, missed = verify_coverage(waypoints, room_mask, unsafe, resolution,
                                  origin, height_full, args.swath_width)
    print(f"\nCoverage check: {pct:.1f}% of drivable room floor swept "
          f"(effective width {args.swath_width:.2f}m)")
    if pct < 95.0:
        print("  WARNING: below 95% -- see coverage_debug_bsa.png (red = missed)")
    save_coverage_debug(args.out, room_mask, unsafe, missed,
                        out_path="coverage_debug_bsa.png")
    print("  Saved coverage_debug_bsa.png")

    # match each jump target's captured world coord to its nearest final
    # waypoint index, so the preview can draw that approach as dashed
    jump_marks = []
    for (jx, jy) in jump_world_targets:
        d = [math.hypot(w["x"] - jx, w["y"] - jy) for w in waypoints]
        jump_marks.append(int(np.argmin(d)))

    preview_path = "coverage_preview_bsa.png"
    save_preview_plot(waypoints, corners, preview_path, "BSA", jump_marks=jump_marks)
    print(f"  Saved {preview_path} ({len(jump_marks)} backtrack jump(s) marked dashed)")


if __name__ == "__main__":
    main()