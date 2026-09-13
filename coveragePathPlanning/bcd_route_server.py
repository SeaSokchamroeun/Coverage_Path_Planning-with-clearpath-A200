#!/usr/bin/env python3
"""
Tier 2 Route Server core: BCD decomposition + guidance tracks + route ordering.

Replaces the hardcoded ROOM_ROW_SLICE crop from earlier versions with an
automatic boustrophedon cellular decomposition (BCD) over the WHOLE map.
This generalizes to any number of rooms/obstacles without per-room tuning.

Pipeline (mirrors the paper's Tier-2 scope: ROI decomposition + guidance
tracks + route planning -- no GTSP, no Dubins/NURBS, since a differential
-drive Husky has ~zero turning radius and doesn't need curvature-
constrained smoothing; Nav2's MPPI controller + goThroughPoses already
gives smooth motion from simple waypoints):

  1. Load map, build inflated "unsafe" grid (same as before).
  2. Light morphological cleanup of the obstacle mask BEFORE decomposition
     -- raw pixel noise on wall/obstacle edges creates spurious split/merge
     events during the sweep, fragmenting cells into slivers. This step
     matters a lot in practice; skipping it is the #1 cause of a BCD
     that "explodes" into hundreds of tiny cells.
  3. BCD: sweep column by column (vertical sweep line moving in x).
     Track free-space intervals per column. A cell continues only on a
     clean 1-to-1 interval mapping between consecutive columns; any
     split (one interval -> many) or merge (many -> one) closes the
     involved cell(s) and opens new one(s). This is the standard
     critical-point rule, just implemented directly on raster columns
     instead of polygon vertices.
  4. Filter degenerate cells below a minimum area (raster BCD produces
     some slivers even after cleanup; drop them rather than route to them).
  5. Per surviving cell: generate vertical guidance tracks (tracks are
     perpendicular to the sweep direction, spaced by SWATH_WIDTH_M),
     boustrophedon-ordered.
  6. Cell visiting order: greedy nearest-neighbor over cell centroids,
     starting from the robot's current position. (Not GTSP-optimal, but
     cheap and good enough at building-floor scale.)
  7. Stitch: connect consecutive cells' nearest entry/exit waypoints
     using the same elbow -> local A* -> warn-and-diagonal fallback
     chain from v3, generalized to handle both axis directions.

Outputs coverage_waypoints.yaml (frame_id: map), plus prints per-cell
stats so you can sanity check cell count/sizes before trusting it.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml
import math
import heapq
import colorsys

# ---- Config ----
PGM_PATH = "office_map.pgm"
YAML_PATH = "office_map.yaml"
OUT_PATH = "coverage_waypoints.yaml"

ROBOT_RADIUS_M = 0.40
EXTRA_CLEARANCE_M = 0.20
SWATH_WIDTH_M = 0.65
LOCAL_ASTAR_PAD_PX = 20
SPAWN_WORLD = (0.0, 0.0)

# Morphological cleanup before decomposition -- collapses single-pixel
# noise on obstacle edges so the sweep doesn't fragment cells at every
# stray occupied pixel. In grid cells (not meters); 2-3 is typical.
CLEANUP_ITER_PX = 2

# A cell smaller than this (in m^2) is treated as decomposition noise
# and dropped rather than routed to.
MIN_CELL_AREA_M2 = 0.5

# A cell narrower than this (in meters, along the sweep/x direction) is
# treated as a furniture-induced fragment and merged into its larger
# neighbor rather than kept as its own region. Set this comfortably
# above your largest expected furniture piece, comfortably below your
# smallest real room width.
MIN_CELL_WIDTH_M = 2.0


# ---------------------------------------------------------------------
# Map I/O (unchanged from v3)
# ---------------------------------------------------------------------

def load_map_meta(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def build_unsafe_grid(pgm_path, resolution):
    img = np.array(Image.open(pgm_path))
    free = img > 250
    occ = img < 50

    # Morphological cleanup: close small gaps in obstacle mask, then open
    # to remove stray single-pixel occupied specks. Do this BEFORE
    # inflation -- cleaning up after inflation just inflates the noise too.
    occ_clean = ndimage.binary_closing(occ, iterations=CLEANUP_ITER_PX)
    occ_clean = ndimage.binary_opening(occ_clean, iterations=CLEANUP_ITER_PX)

    # Same cleanup on the FREE mask. Raw maps can have single-pixel free
    # specks sitting inside what's otherwise a wall/unknown region (seen
    # on this map at row 0 for several columns) -- without this, those
    # specks register as a real 1px-tall "free interval" during the BCD
    # column sweep, and any code that unions/merges intervals per column
    # can stitch that phantom point into a real cell's row range,
    # producing a garbage waypoint deep inside a wall.
    free_clean = ndimage.binary_opening(free, iterations=CLEANUP_ITER_PX)

    total_margin_m = ROBOT_RADIUS_M + EXTRA_CLEARANCE_M
    inflate_px = max(1, int(round(total_margin_m / resolution)))
    occ_inflated = ndimage.binary_dilation(occ_clean, iterations=inflate_px)

    unsafe = occ_inflated | (~free_clean & ~occ_clean)
    return unsafe, img.shape[0]


# ---------------------------------------------------------------------
# A* + local A* + segment clearance (unchanged from v3 -- reused as the
# stitching primitives for both intra-cell and inter-cell connections)
# ---------------------------------------------------------------------

def astar(unsafe_grid, start_rc, goal_rc):
    rows, cols = unsafe_grid.shape
    if unsafe_grid[start_rc] or unsafe_grid[goal_rc]:
        return None

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    neighbors = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    open_set = [(h(start_rc, goal_rc), 0, start_rc)]
    came_from = {}
    g_score = {start_rc: 0}
    visited = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_rc:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dr, dc, cost in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if unsafe_grid[nr, nc]:
                continue
            ng = g + cost
            if (nr, nc) not in g_score or ng < g_score[(nr, nc)]:
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = current
                heapq.heappush(open_set, (ng + h((nr, nc), goal_rc), ng, (nr, nc)))
    return None


def local_astar(unsafe_grid, start_rc, goal_rc, pad_px=LOCAL_ASTAR_PAD_PX):
    r0 = max(0, min(start_rc[0], goal_rc[0]) - pad_px)
    r1 = min(unsafe_grid.shape[0] - 1, max(start_rc[0], goal_rc[0]) + pad_px)
    c0 = max(0, min(start_rc[1], goal_rc[1]) - pad_px)
    c1 = min(unsafe_grid.shape[1] - 1, max(start_rc[1], goal_rc[1]) + pad_px)

    sub = unsafe_grid[r0:r1 + 1, c0:c1 + 1]
    local_path = astar(sub, (start_rc[0] - r0, start_rc[1] - c0),
                        (goal_rc[0] - r0, goal_rc[1] - c0))
    if local_path is None:
        return None
    return [(r + r0, c + c0) for r, c in local_path]


def segment_clear(unsafe_grid, rc0, rc1):
    r0, c0 = rc0
    r1, c1 = rc1
    n_steps = max(abs(r1 - r0), abs(c1 - c0))
    if n_steps == 0:
        return not unsafe_grid[r0, c0]
    for i in range(n_steps + 1):
        t = i / n_steps
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if unsafe_grid[r, c]:
            return False
    return True


def is_safe(unsafe_grid, rc):
    r, c = rc
    return 0 <= r < unsafe_grid.shape[0] and 0 <= c < unsafe_grid.shape[1] and not unsafe_grid[r, c]


def downsample_path(path_rc, step_px):
    if len(path_rc) <= 2:
        return path_rc
    out = path_rc[::step_px]
    if out[-1] != path_rc[-1]:
        out = out + [path_rc[-1]]
    return out


def greedy_shortcut(unsafe_grid, path_rc):
    """Safety-checked greedy line-of-sight shortcutting on a raw grid
    path (e.g. straight from local_astar/astar). Unlike downsample_path's
    blind stride, every kept segment is verified via segment_clear --
    this can never introduce an unsafe shortcut through an obstacle, it
    can only remove genuinely redundant points the A* grid search added.

    Was: downsample_path()'s stride sample, which picks every Nth raw
    A* pixel with no check that the straight line BETWEEN two sampled
    points stays clear. An A* path curves smoothly around an obstacle;
    sampling it by blind stride and connecting those samples with
    straight lines can zigzag back and forth in a way the original path
    never did, since nothing verifies the shortcut segments are safe.
    Found via a real connector between a spawn point and the first lane
    waypoint: the fallback A* path was fine, but its downsampled output
    visibly oscillated (multiple up/down reversals) before settling into
    the actual route -- a sampling artifact, not a routing decision.
    """
    if len(path_rc) < 3:
        return path_rc
    out = [path_rc[0]]
    i = 0
    n = len(path_rc)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not segment_clear(unsafe_grid, path_rc[i], path_rc[j]):
            j -= 1
        out.append(path_rc[j])
        i = j
    return out


def connect_points(unsafe_grid, prev, pt, warnings):
    """General-purpose connector between two pixel points, used for both
    intra-cell lane transitions and inter-cell stitching. Tries, in order:
    direct line, both axis-aligned elbows (full-segment checked), a
    bounded local A* detour, then a full-grid A* as a last resort before
    giving up and warning."""
    if prev == pt:
        return []

    if segment_clear(unsafe_grid, prev, pt):
        return []  # nothing needed, straight shot is already safe

    if pt[0] != prev[0] and pt[1] != prev[1]:
        elbow_a = (pt[0], prev[1])
        elbow_b = (prev[0], pt[1])
        if (is_safe(unsafe_grid, elbow_a)
                and segment_clear(unsafe_grid, prev, elbow_a)
                and segment_clear(unsafe_grid, elbow_a, pt)):
            return [elbow_a]
        if (is_safe(unsafe_grid, elbow_b)
                and segment_clear(unsafe_grid, prev, elbow_b)
                and segment_clear(unsafe_grid, elbow_b, pt)):
            return [elbow_b]

    detour = local_astar(unsafe_grid, prev, pt)
    if detour is None:
        detour = astar(unsafe_grid, prev, pt)  # last-resort full search
    if detour is not None:
        interior = greedy_shortcut(unsafe_grid, detour)[1:-1]
        return interior

    warnings.append(f"no clear connector between {prev} and {pt}")
    return []


# ---------------------------------------------------------------------
# Boustrophedon cellular decomposition (raster column sweep)
# ---------------------------------------------------------------------

def free_intervals_in_column(unsafe_col):
    """Return list of (row_start, row_end) inclusive free runs in a
    single column of the unsafe grid."""
    intervals = []
    in_run = False
    start = None
    n = len(unsafe_col)
    for r in range(n):
        blocked = unsafe_col[r]
        if not blocked and not in_run:
            start = r
            in_run = True
        if blocked and in_run:
            intervals.append((start, r - 1))
            in_run = False
    if in_run:
        intervals.append((start, n - 1))
    return intervals


def bcd_decompose(unsafe_grid):
    """Column-sweep BCD. Returns dict: cell_id -> {'cols': [...],
    'row_ranges': [(r0, r1), ...]} (parallel lists, one entry per column
    the cell spans)."""
    n_rows, n_cols = unsafe_grid.shape
    cells = {}
    next_id = 0

    prev_intervals = []
    active_cell_ids = []

    for col in range(n_cols):
        cur_intervals = free_intervals_in_column(unsafe_grid[:, col])

        adjacency = [[] for _ in cur_intervals]
        prev_adjacency = [[] for _ in prev_intervals]
        for pi, (pr0, pr1) in enumerate(prev_intervals):
            for ci, (cr0, cr1) in enumerate(cur_intervals):
                if not (cr1 < pr0 or cr0 > pr1):
                    adjacency[ci].append(pi)
                    prev_adjacency[pi].append(ci)

        new_active_cell_ids = [None] * len(cur_intervals)
        for ci in range(len(cur_intervals)):
            preds = adjacency[ci]
            continues = (len(preds) == 1) and (len(prev_adjacency[preds[0]]) == 1)
            if continues:
                cid = active_cell_ids[preds[0]]
            else:
                cid = next_id
                next_id += 1
                cells[cid] = {'cols': [], 'row_ranges': []}
            new_active_cell_ids[ci] = cid
            cells[cid]['cols'].append(col)
            cells[cid]['row_ranges'].append(cur_intervals[ci])

        active_cell_ids = new_active_cell_ids
        prev_intervals = cur_intervals

    return cells


def merge_small_cells(cells, resolution, min_width_m=2.0, min_area_m2=MIN_CELL_AREA_M2):
    """Fold narrow/small cells back into a larger neighbor.

    A real room-dividing wall keeps its two sides permanently separate
    (the split cell spans many meters and never remerges). A piece of
    furniture only splits the free space locally -- the resulting
    fragment is narrow (on the order of the furniture's own size) and
    borders a much larger cell on one or both sides. This pass collapses
    those narrow fragments back into whichever larger neighbor they
    bordered, so "cells" converge on rooms/sections rather than every
    obstacle-induced sub-region. The per-column row_range data is kept
    as-is when merging -- it's still the real free-space interval at
    that column (already dodging the furniture), only the cell-ID
    bookkeeping changes.
    """
    def cell_width_m(cell):
        return len(set(cell['cols'])) * resolution

    def cell_area_m2(cell):
        return sum((r1 - r0 + 1) for (r0, r1) in cell['row_ranges']) * (resolution ** 2)

    col_map = {}
    for cid, cell in cells.items():
        for col in cell['cols']:
            col_map.setdefault(col, set()).add(cid)

    changed = True
    while changed:
        changed = False
        small_ids = [cid for cid, c in cells.items()
                     if cell_width_m(c) < min_width_m or cell_area_m2(c) < min_area_m2]
        for cid in small_ids:
            if cid not in cells:
                continue
            cell = cells[cid]
            cols_sorted = sorted(cell['cols'])
            first_col, last_col = cols_sorted[0], cols_sorted[-1]

            candidates = {}
            for probe_col in (first_col - 1, last_col + 1):
                for other_cid in col_map.get(probe_col, set()):
                    if other_cid != cid and other_cid in cells:
                        candidates[other_cid] = cell_area_m2(cells[other_cid])
            if not candidates:
                continue  # truly isolated small region -- leave for area filter to drop

            target_cid = max(candidates, key=candidates.get)
            cells[target_cid]['cols'].extend(cell['cols'])
            cells[target_cid]['row_ranges'].extend(cell['row_ranges'])
            for col in cell['cols']:
                s = col_map.setdefault(col, set())
                s.discard(cid)
                s.add(target_cid)
            del cells[cid]
            changed = True

    return cells


def filter_small_cells(cells, resolution, min_area_m2):
    kept = {}
    for cid, cell in cells.items():
        area_px = sum((r1 - r0 + 1) for (r0, r1) in cell['row_ranges'])
        area_m2 = area_px * (resolution ** 2)
        if area_m2 >= min_area_m2:
            kept[cid] = cell
    return kept


def cell_centroid_px(cell):
    rs = [((r0 + r1) / 2.0) for (r0, r1) in cell['row_ranges']]
    return (sum(rs) / len(rs), sum(cell['cols']) / len(cell['cols']))


# ---------------------------------------------------------------------
# Per-cell guidance tracks (boustrophedon, vertical -- perpendicular to
# the sweep direction, matching standard BCD coverage convention)
# ---------------------------------------------------------------------

def cell_guidance_tracks(unsafe_grid, cell, resolution, warnings):
    lane_spacing_px = max(1, int(round(SWATH_WIDTH_M / resolution)))
    col_to_range = {}
    for col, rr in zip(cell['cols'], cell['row_ranges']):
        if col in col_to_range:
            er0, er1 = col_to_range[col]
            nr0, nr1 = rr
            touching = not (nr1 < er0 - 1 or nr0 > er1 + 1)
            if touching:
                col_to_range[col] = (min(er0, nr0), max(er1, nr1))
            else:
                # Disjoint ranges for the same column -- almost certainly
                # one is decomposition/pixel noise. Keep whichever is
                # taller rather than bridging across the gap between them
                # (bridging would fabricate a path through unsafe space).
                existing_h = er1 - er0
                new_h = nr1 - nr0
                col_to_range[col] = rr if new_h > existing_h else (er0, er1)
        else:
            col_to_range[col] = rr
    cols_sorted = sorted(cell['cols'])
    track_cols = cols_sorted[::lane_spacing_px]
    if track_cols[-1] != cols_sorted[-1]:
        track_cols.append(cols_sorted[-1])

    waypoints = []
    for i, col in enumerate(track_cols):
        r0, r1 = col_to_range[col]
        top, bottom = (r0, col), (r1, col)  # (row, col) order -- matches every other function
        # alternate direction for boustrophedon
        a, b = (top, bottom) if i % 2 == 0 else (bottom, top)
        if waypoints:
            waypoints.extend(connect_points(unsafe_grid, waypoints[-1], a, warnings))
        waypoints.append(a)
        waypoints.append(b)
    return waypoints


# ---------------------------------------------------------------------
# Route planning: greedy nearest-neighbor cell order + stitching
# ---------------------------------------------------------------------

def order_cells_nearest_neighbor(cells, start_px):
    remaining = dict(cells)
    order = []
    current_pos = start_px
    while remaining:
        best_id, best_dist = None, None
        for cid, cell in remaining.items():
            cx, cy = cell_centroid_px(cell)
            d = math.hypot(cx - current_pos[0], cy - current_pos[1])
            if best_dist is None or d < best_dist:
                best_dist, best_id = d, cid
        order.append(best_id)
        current_pos = cell_centroid_px(remaining[best_id])
        del remaining[best_id]
    return order


def build_full_route(unsafe_grid, cells, resolution, start_px):
    warnings = []
    order = order_cells_nearest_neighbor(cells, start_px)

    full_route = []
    cursor = start_px
    for cid in order:
        cell_wps = cell_guidance_tracks(unsafe_grid, cells[cid], resolution, warnings)
        if not cell_wps:
            continue
        # pick whichever end of this cell's route is closer to connect to
        d_start = math.hypot(cell_wps[0][0] - cursor[0], cell_wps[0][1] - cursor[1])
        d_end = math.hypot(cell_wps[-1][0] - cursor[0], cell_wps[-1][1] - cursor[1])
        if d_end < d_start:
            cell_wps = cell_wps[::-1]

        connector = connect_points(unsafe_grid, cursor, cell_wps[0], warnings)
        full_route.extend(connector)
        full_route.extend(cell_wps)
        cursor = cell_wps[-1]

    return full_route, order, warnings


# ---------------------------------------------------------------------
# ROI visualization -- color each surviving cell distinctly so you can
# eyeball whether the decomposition matches real rooms/sections.
# ---------------------------------------------------------------------

def save_roi_visualization(pgm_path, unsafe_grid, cells, out_path="roi_debug.png"):
    base = np.array(Image.open(pgm_path).convert("RGB"))

    # dim the base image so overlay colors stand out
    base = (base * 0.5).astype(np.uint8)

    n = max(1, len(cells))
    for i, (cid, cell) in enumerate(cells.items()):
        hue = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        color = np.array([r, g, b]) * 255

        for col, (r0, r1) in zip(cell['cols'], cell['row_ranges']):
            base[r0:r1 + 1, col] = (0.4 * base[r0:r1 + 1, col] + 0.6 * color).astype(np.uint8)

    # obstacles/unsafe shown as dark red tint so you can see what got
    # inflated away (useful for spotting sealed-off doorways too)
    obstacle_mask = unsafe_grid
    overlay = base.copy()
    overlay[obstacle_mask] = (0.5 * base[obstacle_mask] + 0.5 * np.array([200, 30, 30])).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)
    print(f"Saved ROI visualization -> {out_path} "
          f"({len(cells)} cells, each a distinct color; red tint = inflated/unsafe area)")

    # also print a text legend: cell id -> approx pixel centroid, so you
    # can cross-reference a color patch in the image back to a cell id
    print("Cell legend (id: centroid row,col):")
    for cid, cell in cells.items():
        cy, cx = cell_centroid_px(cell)
        print(f"  {cid}: ({cy:.0f}, {cx:.0f})")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    meta = load_map_meta(YAML_PATH)
    resolution = meta["resolution"]
    origin_x, origin_y = meta["origin"][0], meta["origin"][1]

    unsafe, height_full = build_unsafe_grid(PGM_PATH, resolution)

    def world_to_px(wx, wy):
        col = int(round((wx - origin_x) / resolution))
        row = int(round(height_full - 1 - (wy - origin_y) / resolution))
        return row, col

    def px_to_world(row, col):
        wx = origin_x + col * resolution
        wy = origin_y + (height_full - 1 - row) * resolution
        return wx, wy

    print("Running BCD decomposition...")
    raw_cells = bcd_decompose(unsafe)
    print(f"Raw cells before filtering: {len(raw_cells)}")

    merged_cells = merge_small_cells(raw_cells, resolution,
                                      min_width_m=MIN_CELL_WIDTH_M,
                                      min_area_m2=MIN_CELL_AREA_M2)
    print(f"Cells after merging furniture-induced fragments: {len(merged_cells)}")

    cells = filter_small_cells(merged_cells, resolution, MIN_CELL_AREA_M2)
    print(f"Cells after filtering < {MIN_CELL_AREA_M2} m^2: {len(cells)}")
    for cid, cell in cells.items():
        area_px = sum((r1 - r0 + 1) for (r0, r1) in cell['row_ranges'])
        print(f"  cell {cid}: {len(cell['cols'])} cols, area {area_px * resolution**2:.2f} m^2")

    if not cells:
        raise RuntimeError("No cells survived filtering -- check MIN_CELL_AREA_M2 "
                            "and CLEANUP_ITER_PX, or inspect the unsafe grid directly.")

    save_roi_visualization(PGM_PATH, unsafe, cells, out_path="roi_debug.png")

    start_px = world_to_px(*SPAWN_WORLD)
    full_route_px, order, warnings = build_full_route(unsafe, cells, resolution, start_px)

    print(f"Cell visiting order: {order}")
    if warnings:
        print(f"⚠ {len(warnings)} stitching warning(s):")
        for w in warnings:
            print(f"  - {w}")

    world_pts = [px_to_world(r, c) for r, c in full_route_px]
    waypoints = []
    for i, (x, y) in enumerate(world_pts):
        if i < len(world_pts) - 1:
            nx, ny = world_pts[i + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = waypoints[-1]["yaw"] if waypoints else 0.0
        waypoints.append({"x": float(x), "y": float(y), "yaw": float(yaw)})

    with open(OUT_PATH, "w") as f:
        yaml.dump({"frame_id": "map", "waypoints": waypoints,
                   "cell_order": order, "cell_count": len(cells)}, f)

    print(f"Generated {len(waypoints)} waypoints across {len(cells)} cells -> {OUT_PATH}")


if __name__ == "__main__":
    main()