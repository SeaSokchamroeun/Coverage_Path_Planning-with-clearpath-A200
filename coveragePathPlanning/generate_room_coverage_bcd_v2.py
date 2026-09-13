#!/usr/bin/env python3
"""
generate_room_coverage_bcd_v2.py

Boustrophedon Cellular Decomposition coverage path generator for the
Clearpath A200 (Husky).

Self-contained: depends only on numpy / scipy / PyYAML / Pillow
(+ matplotlib for the preview). It does NOT import bcd_route_server or
coverage_common, so it cannot inherit their tuning state.

Pipeline
    1. load occupancy map, build free mask (unknown treated as obstacle)
    2. inflate obstacles by robot_radius + safety_margin  (exact EDT)
    3. optionally clip to room_boundary.yaml polygon
    4. keep only the component reachable from the spawn point
    5. BCD slice decomposition -> cells (empty room => exactly 1 cell)
    6. per cell: evenly spaced boustrophedon lanes, serpentine order
    7. connect lanes / cells with line-of-sight or A* on the safe grid
    8. simplify (lane endpoints protected), dedupe, assign yaw
    9. verify coverage by dilating the swept path, write debug + preview

Usage
    python3 generate_room_coverage_bcd_v2.py \
        --map office_mapEmpty.yaml \
        --boundary room_boundary.yaml \
        --out coverage_waypoints_bcd.yaml \
        --spawn 0 0
"""

import argparse
import heapq
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image

try:
    import scipy.ndimage as ndi
except ImportError:
    sys.exit("scipy is required:  pip3 install scipy")


# ----------------------------------------------------------------------
# map loading
# ----------------------------------------------------------------------

def load_map(map_yaml):
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)

    img_rel = meta["image"]
    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(
        os.path.dirname(os.path.abspath(map_yaml)), img_rel)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"map image not found: {img_path}")

    img = np.array(Image.open(img_path).convert("L")).astype(np.float64)

    res = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]
    negate = int(meta.get("negate", 0))
    free_th = float(meta.get("free_thresh", 0.196))
    occ_th = float(meta.get("occupied_thresh", 0.65))

    occ = img / 255.0 if negate else (255.0 - img) / 255.0

    free = occ < free_th
    occupied = occ > occ_th
    unknown = ~free & ~occupied

    return dict(meta=meta, img=img, res=res, origin=origin,
                free=free, occupied=occupied, unknown=unknown,
                height=img.shape[0], width=img.shape[1])


def load_boundary(path):
    """Tolerant reader for room_boundary.yaml. Returns
    (corners_or_None, frame_id, robot_radius_or_None, margin_or_None)."""
    with open(path) as f:
        data = yaml.safe_load(f)

    frame_id = data.get("frame_id", "map")
    radius = data.get("robot_radius")
    margin = data.get("margin")

    corners = None
    for key in ("corners", "polygon", "points", "boundary", "vertices"):
        if key in data and data[key]:
            corners = data[key]
            break

    if corners is not None:
        pts = []
        for c in corners:
            if isinstance(c, dict):
                pts.append((float(c["x"]), float(c["y"])))
            else:
                pts.append((float(c[0]), float(c[1])))
        corners = pts

    return corners, frame_id, radius, margin


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------

class Frame:
    """Pixel <-> world conversion.

    Matches the convention already used across this project:
        wx = origin_x + col * res
        wy = origin_y + (height - 1 - row) * res
    """

    def __init__(self, origin, res, height):
        self.ox, self.oy = origin[0], origin[1]
        self.res = res
        self.h = height

    def to_world(self, row, col):
        return (self.ox + col * self.res,
                self.oy + (self.h - 1 - row) * self.res)

    def to_px(self, wx, wy):
        col = int(round((wx - self.ox) / self.res))
        row = int(round(self.h - 1 - (wy - self.oy) / self.res))
        return row, col


def polygon_mask(corners, frame, shape):
    """Rasterise a world-frame polygon into a boolean pixel mask."""
    from matplotlib.path import Path
    h, w = shape
    px = [frame.to_px(x, y) for x, y in corners]
    poly = Path([(c, r) for r, c in px])
    cc, rr = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.column_stack([cc.ravel(), rr.ravel()])
    inside = poly.contains_points(pts, radius=0.5).reshape(h, w)
    return inside


def line_clear(grid, a, b):
    """True if the straight segment a->b stays on passable cells.
    grid is True where DRIVABLE."""
    r0, c0 = a
    r1, c1 = b
    n = int(max(abs(r1 - r0), abs(c1 - c0)))
    if n == 0:
        return grid[r0, c0]
    for i in range(n + 1):
        t = i / n
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        if not grid[r, c]:
            return False
    return True


def astar(grid, start, goal):
    """8-connected A* over a boolean passable grid. Returns list of
    intermediate points (exclusive of start and goal), or None."""
    h, w = grid.shape
    if not grid[start] or not grid[goal]:
        return None

    def hcost(p):
        dr = abs(p[0] - goal[0])
        dc = abs(p[1] - goal[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    open_heap = [(hcost(start), 0.0, start)]
    came = {}
    gscore = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path[1:]
        if cur in closed:
            continue
        closed.add(cur)
        for dr, dc, cost in nbrs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or not grid[nr, nc]:
                continue
            ng = g + cost
            if ng < gscore.get((nr, nc), float("inf")):
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + hcost((nr, nc)), ng, (nr, nc)))
    return None


def connect(grid, a, b):
    """Intermediate points needed to get from a to b safely."""
    if a == b:
        return []
    if line_clear(grid, a, b):
        return []
    path = astar(grid, a, b)
    if path is None:
        return None
    # thin it out with line-of-sight shortcutting
    out = [a] + path + [b]
    keep = [out[0]]
    i = 0
    while i < len(out) - 1:
        j = len(out) - 1
        while j > i + 1 and not line_clear(grid, out[i], out[j]):
            j -= 1
        keep.append(out[j])
        i = j
    return keep[1:-1]


# ----------------------------------------------------------------------
# BCD decomposition
# ----------------------------------------------------------------------

def column_runs(mask, col):
    """Contiguous True runs in one column -> [(r0, r1), ...]."""
    rows = np.where(mask[:, col])[0]
    if rows.size == 0:
        return []
    splits = np.where(np.diff(rows) > 1)[0]
    return [(int(r[0]), int(r[-1])) for r in np.split(rows, splits + 1)]


def bcd_decompose(mask):
    """Boustrophedon cellular decomposition by vertical slice sweep.

    Returns a list of cells; each cell is {col: (r0, r1)}.
    A room with no interior obstacles yields exactly one cell.
    """
    h, w = mask.shape
    cells = []                 # finished cells
    active = {}                # run_index_in_prev_col -> cell dict
    prev_runs = []

    for col in range(w):
        runs = column_runs(mask, col)

        # overlap adjacency between prev_runs and runs
        prev_to_cur = {i: [] for i in range(len(prev_runs))}
        cur_to_prev = {j: [] for j in range(len(runs))}
        for i, (pr0, pr1) in enumerate(prev_runs):
            for j, (r0, r1) in enumerate(runs):
                if not (r1 < pr0 or r0 > pr1):      # they overlap
                    prev_to_cur[i].append(j)
                    cur_to_prev[j].append(i)

        new_active = {}
        for j, run in enumerate(runs):
            preds = cur_to_prev[j]
            if len(preds) == 1 and len(prev_to_cur[preds[0]]) == 1:
                # simple continuation of one cell
                cell = active.get(preds[0])
                if cell is None:
                    cell = {}
                    cells.append(cell)
                cell[col] = run
                new_active[j] = cell
            else:
                # IN event (no predecessor), or a split / merge:
                # close every predecessor cell, start a fresh one
                cell = {col: run}
                cells.append(cell)
                new_active[j] = cell

        active = new_active
        prev_runs = runs

    return [c for c in cells if c]


# ----------------------------------------------------------------------
# lane generation
# ----------------------------------------------------------------------

def merge_thin_cells(cells, min_cols):
    """Fold away cells that span fewer than `min_cols` slices.

    A single ragged pixel at a wall makes one column split into two runs,
    which is a legitimate BCD critical point but produces a useless
    one-column cell. Those get merged back into the adjacent cell they
    touch, so an obstacle-free room ends up as exactly one cell.
    """
    cells = [dict(c) for c in cells]

    def span(c):
        k = c.keys()
        return min(k), max(k)

    changed = True
    while changed:
        changed = False
        cells.sort(key=lambda c: span(c)[0])
        for i, cell in enumerate(cells):
            lo, hi = span(cell)
            if (hi - lo + 1) >= min_cols:
                continue

            best, best_area = None, -1
            for j, other in enumerate(cells):
                if j == i:
                    continue
                olo, ohi = span(other)
                # column ranges must touch or overlap
                if ohi < lo - 1 or olo > hi + 1:
                    continue
                # and the two must actually connect in the row direction
                # somewhere along the shared / adjacent column
                touches = False
                for col, (cr0, cr1) in cell.items():
                    for probe in (col - 1, col, col + 1):
                        if probe not in other:
                            continue
                        pr0, pr1 = other[probe]
                        if not (cr1 < pr0 or cr0 > pr1):
                            touches = True
                            break
                    if touches:
                        break
                if not touches:
                    continue
                area = sum(r1 - r0 + 1 for r0, r1 in other.values())
                if area > best_area:
                    best, best_area = j, area

            if best is None:
                continue

            target = cells[best]
            for col, (r0, r1) in cell.items():
                if col not in target:
                    target[col] = (r0, r1)
                else:
                    t0, t1 = target[col]
                    if r1 < t0 or r0 > t1:
                        # disjoint runs: keep the taller one rather than
                        # unioning across a real gap
                        if (r1 - r0) > (t1 - t0):
                            target[col] = (r0, r1)
                    else:
                        target[col] = (min(t0, r0), max(t1, r1))
            cells.pop(i)
            changed = True
            break

    return cells


def cell_lanes(cell, spacing_px, inset_px):
    """Evenly spaced boustrophedon lanes across one BCD cell.

    Lanes are placed with linspace over the cell's column extent, so the
    first and last lane sit on the cell's own edges and the real spacing
    is always <= spacing_px. Returns [(a, b), ...] lane endpoint pairs in
    sweep order, already serpentined.
    """
    cols = sorted(cell.keys())
    if not cols:
        return []

    c_lo, c_hi = cols[0], cols[-1]
    span = c_hi - c_lo
    if span <= 0:
        lane_cols = [c_lo]
    else:
        n = int(math.ceil(span / float(spacing_px))) + 1
        n = max(n, 2)
        lane_cols = sorted(set(int(round(v))
                               for v in np.linspace(c_lo, c_hi, n)))

    lanes = []
    for k, col in enumerate(lane_cols):
        if col not in cell:
            # cell is not defined at this exact column; snap to nearest
            col = min(cols, key=lambda c: abs(c - col))
        r0, r1 = cell[col]

        # small inward inset so a lane end never sits on the exact last
        # passable pixel; never past the midpoint
        mid = (r0 + r1) // 2
        a_r = min(r0 + inset_px, mid)
        b_r = max(r1 - inset_px, mid)

        top, bot = (a_r, col), (b_r, col)
        lanes.append((top, bot) if k % 2 == 0 else (bot, top))

    return lanes




# ----------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------

def simplify(path, grid, protected):
    """Line-of-sight shortcutting that never removes a protected point."""
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            # cannot skip over a protected point
            if any(p in protected for p in path[i + 1:j]):
                j -= 1
                continue
            if line_clear(grid, path[i], path[j]):
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


def dedupe(points):
    out = []
    for p in points:
        if not out or p != out[-1]:
            out.append(p)
    return out


def coverage_stats(waypoints, target_mask, grid, frame, res, cover_width):
    """Dilate the driven path by half the coverage width; compare to the
    drivable target area."""
    h, w = target_mask.shape
    swept = np.zeros((h, w), bool)

    px = [frame.to_px(wp["x"], wp["y"]) for wp in waypoints]
    for a, b in zip(px[:-1], px[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
        for i in range(n + 1):
            t = i / n if n else 0.0
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= r < h and 0 <= c < w:
                swept[r, c] = True

    rad_px = max(1, int(round((cover_width / 2.0) / res)))
    dist = ndi.distance_transform_edt(~swept)
    swept = dist <= rad_px

    target = target_mask & grid
    covered = swept & target
    missed = target & ~swept
    pct = 100.0 * covered.sum() / max(1, target.sum())
    return pct, missed, target


def path_length(waypoints):
    return sum(math.dist((a["x"], a["y"]), (b["x"], b["y"]))
               for a, b in zip(waypoints[:-1], waypoints[1:]))


def count_turns(waypoints, thresh_deg=30.0):
    n = 0
    for i in range(1, len(waypoints) - 1):
        a, b, c = waypoints[i - 1], waypoints[i], waypoints[i + 1]
        v1 = math.atan2(b["y"] - a["y"], b["x"] - a["x"])
        v2 = math.atan2(c["y"] - b["y"], c["x"] - b["x"])
        d = abs(math.degrees((v2 - v1 + math.pi) % (2 * math.pi) - math.pi))
        if d > thresh_deg:
            n += 1
    return n


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------

def save_preview(waypoints, corners, out_png, title, frame, shape):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [w["x"] for w in waypoints]
    ys = [w["y"] for w in waypoints]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(xs, ys, "-", lw=0.9, color="0.45", zorder=1)
    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis", s=16, zorder=2)
    plt.colorbar(sc, ax=ax, label="waypoint order (start -> end)")

    if corners:
        cx = [c[0] for c in corners] + [corners[0][0]]
        cy = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(cx, cy, "r-", lw=1.6, label="room boundary")

    ax.plot(xs[0], ys[0], "^", color="blue", ms=14, label="start")
    ax.plot(xs[-1], ys[-1], "s", color="orange", ms=12, label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title} ({len(waypoints)} waypoints)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def save_debug(out_png, target, missed, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.zeros(target.shape + (3,), dtype=np.uint8)
    rgb[...] = 40
    rgb[grid] = (200, 200, 200)
    rgb[target] = (120, 190, 120)
    rgb[missed] = (220, 40, 40)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title("coverage debug -- red = drivable floor never swept")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="office_mapEmpty.yaml")
    ap.add_argument("--boundary", default=None,
                    help="room_boundary.yaml from extract_boundary.py. "
                         "Optional -- omit to use the whole map's free space.")
    ap.add_argument("--out", default="coverage_waypoints_bcd.yaml")
    ap.add_argument("--spawn", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("X", "Y"),
                    help="Where the robot starts, world frame.")

    ap.add_argument("--robot-radius", type=float, default=None,
                    help="Half-footprint used to inflate obstacles. "
                         "Defaults to the value stored in --boundary, "
                         "else 0.40 (A200 half-diagonal-ish).")
    ap.add_argument("--safety-margin", type=float, default=None,
                    help="Extra inflation on top of --robot-radius. "
                         "Defaults to the value stored in --boundary, else 0.10.")

    ap.add_argument("--coverage-width", type=float, default=0.67,
                    help="How wide a strip the robot actually covers, in "
                         "metres. A200 chassis width is 0.67.")
    ap.add_argument("--overlap", type=float, default=0.10,
                    help="Overlap between adjacent lanes, in metres. Lane "
                         "spacing = coverage_width - overlap. Must be > 0 "
                         "for gap-free coverage.")

    ap.add_argument("--endpoint-inset", type=float, default=0.0,
                    help="Pull lane ends inward by this many metres. "
                         "Default 0 -- obstacle inflation already "
                         "guarantees the footprint fits.")
    ap.add_argument("--lane-axis", choices=["auto", "x", "y"], default="auto",
                    help="Direction the lanes run. 'auto' runs them along "
                         "the room's longer axis, which minimises turns.")
    ap.add_argument("--min-cell-area", type=float, default=0.5,
                    help="Discard BCD cells smaller than this many m^2 "
                         "(decomposition noise).")
    ap.add_argument("--preview", default="coverage_preview_bcd.png")
    ap.add_argument("--debug-image", default="coverage_debug.png")
    args = ap.parse_args()

    # ---------------- load ----------------
    try:
        m = load_map(args.map)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    except (KeyError, yaml.YAMLError) as e:
        sys.exit(f"ERROR: {args.map} is not a valid map yaml ({e}).")
    res = m["res"]
    frame = Frame(m["origin"], res, m["height"])
    print(f"Map: {m['width']}x{m['height']} px @ {res} m/px, "
          f"origin {m['origin'][:2]}")

    corners, frame_id, b_radius, b_margin = (None, "map", None, None)
    if args.boundary:
        try:
            corners, frame_id, b_radius, b_margin = load_boundary(args.boundary)
        except (FileNotFoundError, yaml.YAMLError, KeyError, TypeError) as e:
            sys.exit(f"ERROR: could not read {args.boundary} ({e}).")
        if corners:
            print(f"Boundary polygon: {len(corners)} corners, frame '{frame_id}'")
        else:
            print(f"WARNING: no polygon found in {args.boundary}; "
                  "using full map free space.")

    radius = args.robot_radius if args.robot_radius is not None else (
        b_radius if b_radius is not None else 0.40)
    margin = args.safety_margin if args.safety_margin is not None else (
        b_margin if b_margin is not None else 0.10)
    inflation = radius + margin
    print(f"Obstacle inflation: robot_radius {radius} + margin {margin} "
          f"= {inflation:.2f} m")

    spacing = args.coverage_width - args.overlap
    if spacing <= 0:
        sys.exit("ERROR: --overlap must be smaller than --coverage-width.")
    print(f"Lane spacing: coverage_width {args.coverage_width} - overlap "
          f"{args.overlap} = {spacing:.2f} m")

    # ---------------- free / safe grid ----------------
    # unknown counts as obstacle: never plan through unmapped space
    obstacle = ~m["free"]
    dist_px = ndi.distance_transform_edt(~obstacle)
    safe = dist_px * res >= inflation
    print(f"Free: {m['free'].sum()} px ({m['free'].sum()*res*res:.1f} m^2); "
          f"safe after inflation: {safe.sum()} px "
          f"({safe.sum()*res*res:.1f} m^2)")

    target = safe.copy()
    if corners:
        poly = polygon_mask(corners, frame, safe.shape)
        target = safe & poly
        print(f"After clipping to boundary polygon: {target.sum()} px "
              f"({target.sum()*res*res:.1f} m^2)")

    if target.sum() == 0:
        sys.exit("ERROR: nothing drivable after inflation + boundary clip. "
                 "Reduce --robot-radius/--safety-margin or check the boundary.")

    # ---------------- reachable component ----------------
    spawn_px = frame.to_px(*args.spawn)
    lbl, n_lbl = ndi.label(target)
    if not (0 <= spawn_px[0] < lbl.shape[0] and 0 <= spawn_px[1] < lbl.shape[1]):
        sys.exit(f"ERROR: spawn {args.spawn} is outside the map.")

    spawn_lbl = lbl[spawn_px]
    if spawn_lbl == 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        spawn_lbl = int(sizes.argmax())
        rr, cc = np.where(lbl == spawn_lbl)
        d = (rr - spawn_px[0]) ** 2 + (cc - spawn_px[1]) ** 2
        k = int(d.argmin())
        new_spawn = (int(rr[k]), int(cc[k]))
        print(f"WARNING: spawn {args.spawn} is not in drivable space "
              f"(too close to a wall, or unmapped). Snapped to "
              f"{frame.to_world(*new_spawn)}.")
        spawn_px = new_spawn

    target = (lbl == spawn_lbl)
    if n_lbl > 1:
        print(f"Reachability: {n_lbl} components found, kept the one "
              f"containing the spawn ({target.sum()*res*res:.1f} m^2).")

    # ---------------- lane axis ----------------
    rr, cc = np.where(target)
    extent_x = (cc.max() - cc.min()) * res
    extent_y = (rr.max() - rr.min()) * res
    if args.lane_axis == "auto":
        # lanes should run along the LONGER axis -> fewer turns
        lane_axis = "x" if extent_x >= extent_y else "y"
        print(f"Room extent {extent_x:.2f} x {extent_y:.2f} m -> lanes run "
              f"along {lane_axis} (auto, minimises turns)")
    else:
        lane_axis = args.lane_axis

    # The decomposition works on columns. For lanes along x we transpose,
    # run everything identically, then transpose the coordinates back.
    transposed = (lane_axis == "x")

    def T(p):
        return (p[1], p[0])

    work = target.T if transposed else target
    work_safe = (safe.T if transposed else safe)
    start_work = T(spawn_px) if transposed else spawn_px

    # ---------------- decompose ----------------
    spacing_px = max(1, int(round(spacing / res)))

    cells = bcd_decompose(work)
    n_raw_cells = len(cells)
    cells = merge_thin_cells(cells, min_cols=spacing_px)

    min_px = args.min_cell_area / (res * res)
    kept = []
    for c in cells:
        area = sum(r1 - r0 + 1 for r0, r1 in c.values())
        if area >= min_px:
            kept.append(c)
    print(f"BCD: {n_raw_cells} raw cell(s) -> {len(cells)} after merging "
          f"thin edge cells -> {len(kept)} after the "
          f"{args.min_cell_area} m^2 noise filter")
    for i, c in enumerate(kept):
        area = sum(r1 - r0 + 1 for r0, r1 in c.values()) * res * res
        print(f"   cell {i}: {len(c)} slices, {area:.1f} m^2")
    if not kept:
        sys.exit("ERROR: no usable cells.")

    # ---------------- lanes ----------------
    inset_px = max(0, int(round(args.endpoint_inset / res)))

    def flip_all(ls):
        return [(b, a) for a, b in ls]

    def lane_variants(cell):
        """The four sweep orders that keep the serpentine consistent.
        Flipping only the first lane would break every lane after it, so
        direction and entry side must be chosen together."""
        lanes = cell_lanes(cell, spacing_px, inset_px)
        if not lanes:
            return []
        return [lanes, flip_all(lanes),
                lanes[::-1], flip_all(lanes[::-1])]

    protected = set()
    route = [start_work]
    stitch_fail = 0

    # Greedy on the real entry point rather than the cell centroid: a
    # centroid can sit inside an obstacle or on the far side of a cell,
    # which produces long pointless transits between cells.
    remaining = list(kept)
    n_cells_done = 0
    while remaining:
        best = None
        for cell in remaining:
            for variant in lane_variants(cell):
                d = math.dist(route[-1], variant[0][0])
                if best is None or d < best[0]:
                    best = (d, cell, variant)
        if best is None:
            break
        _, cell, lanes = best
        remaining.remove(cell)
        n_cells_done += 1

        for a, b in lanes:
            for pt in (a, b):
                if not work_safe[pt]:
                    continue
                mid = connect(work_safe, route[-1], pt)
                if mid is None:
                    stitch_fail += 1
                    continue
                route.extend(mid)
                route.append(pt)
                protected.add(pt)

    route = dedupe(route)
    if stitch_fail:
        print(f"WARNING: {stitch_fail} lane endpoint(s) unreachable and skipped.")

    n_raw = len(route)
    route = simplify(route, work_safe, protected)
    route = dedupe(route)
    print(f"Path: {n_raw} raw -> {len(route)} after line-of-sight simplify "
          f"({len(protected)} lane endpoints protected)")

    # ---------------- back to world ----------------
    route_px = [T(p) for p in route] if transposed else route
    pts = [frame.to_world(r, c) for r, c in route_px]

    waypoints = []
    for i, (x, y) in enumerate(pts):
        if i < len(pts) - 1:
            nx, ny = pts[i + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = waypoints[-1]["yaw"] if waypoints else 0.0
        waypoints.append({"x": round(float(x), 4),
                          "y": round(float(y), 4),
                          "yaw": round(float(yaw), 6)})

    # Provenance. Every downstream tool (evaluate_coverage_path_a200.py,
    # analyze_trajectory.py, lane_latch_check.py) needs the geometry this
    # path was PLANNED with. Without it they fall back to their own
    # defaults and grade the path against a robot that never drove it --
    # a wider assumed swath does not just mislabel a number, it reports
    # floor as covered that was never touched.
    provenance = {
        "generator": "generate_room_coverage_bcd_v2.py",
        "platform": "a200",
        "algorithm": "bcd",
        "map": os.path.basename(args.map),
        "boundary": os.path.basename(args.boundary) if args.boundary else None,
        "resolution": float(res),
        "robot_radius": float(radius),
        "extra_clearance": float(margin),
        "inflation": float(inflation),
        "coverage_width": float(args.coverage_width),
        "overlap": float(args.overlap),
        "swath_width": float(spacing),      # ACTUAL lane spacing
        "lane_axis": lane_axis,
        "endpoint_inset": float(args.endpoint_inset),
        "spawn": [float(args.spawn[0]), float(args.spawn[1])],
        "n_cells": int(n_cells_done),
    }

    with open(args.out, "w") as f:
        yaml.dump({"frame_id": frame_id,
                   "generated_with": provenance,
                   "waypoints": waypoints},
                  f, sort_keys=False)
    print(f"\nWrote {len(waypoints)} waypoints -> {args.out}")

    # ---------------- verify ----------------
    pct, missed, tgt = coverage_stats(waypoints, target, safe, frame, res,
                                      args.coverage_width)
    length = path_length(waypoints)
    turns = count_turns(waypoints)
    print(f"Coverage : {pct:.1f}% of {tgt.sum()*res*res:.1f} m^2 drivable floor")
    print(f"Path len : {length:.1f} m")
    print(f"Turns    : {turns} (>30 deg)")
    if pct < 95.0:
        print("  WARNING: below 95% -- inspect the debug image "
              "(red = floor never swept).")

    save_debug(args.debug_image, tgt, missed, safe)
    save_preview(waypoints, corners, args.preview,
                 f"BCD coverage ({n_cells_done} cell"
                 f"{'s' if n_cells_done != 1 else ''})", frame, target.shape)
    print(f"Saved {args.debug_image} and {args.preview}")


if __name__ == "__main__":
    main()
