"""
Strict axis-aligned (90-degree-only) replacements for connect_points and
simplify_path. The shared bcd_route_server.connect_points() tries a direct
diagonal shot FIRST (only falls back to an axis-aligned elbow if blocked),
and its own local_astar is 8-connected (allows 45-degree steps) -- both
by design, for other callers that want smoother/shorter paths. This room
script wants strict Manhattan turns instead, so these are separate
functions rather than edits to the shared file used elsewhere.
"""
import heapq
import math
import bcd_route_server as brs


def astar_4connected(unsafe_grid, start_rc, goal_rc):
    """Same as bcd_route_server.astar but with ONLY the 4 axis-aligned
    neighbors -- guarantees every step in the returned path is a pure
    horizontal or vertical move, never diagonal."""
    if unsafe_grid[start_rc] or unsafe_grid[goal_rc]:
        return None
    rows, cols = unsafe_grid.shape

    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
        for dr, dc in neighbors:
            nr, nc = current[0]+dr, current[1]+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if unsafe_grid[nr, nc]:
                continue
            ng = g + 1
            npt = (nr, nc)
            if npt in visited:
                continue
            if ng < g_score.get(npt, float('inf')):
                g_score[npt] = ng
                came_from[npt] = current
                heapq.heappush(open_set, (ng + h(npt, goal_rc), ng, npt))
    return None


def connect_points_manhattan(unsafe_grid, prev, pt, warnings, avoid_dir=None):
    """Elbow-first (axis-aligned single corner), then 4-connected A*
    detour, then the shared diagonal connector ONLY as an absolute last
    resort (with a warning, so a broken route is never silently produced).

    avoid_dir, if given, is the (dr, dc) unit direction the robot just
    traveled to reach `prev`. When both elbows are valid, this discards
    whichever one's first leg is the exact reverse of avoid_dir, so the
    connector never doubles back over its own incoming direction unless
    that's truly the only option."""
    if prev == pt:
        return []
    if prev[0] == pt[0] or prev[1] == pt[1]:
        if brs.segment_clear(unsafe_grid, prev, pt):
            return []

    if pt[0] != prev[0] and pt[1] != prev[1]:
        elbow_a = (pt[0], prev[1])
        elbow_b = (prev[0], pt[1])

        def first_leg_dir(elbow):
            dr, dc = elbow[0]-prev[0], elbow[1]-prev[1]
            if dr != 0:
                return (1 if dr > 0 else -1, 0)
            if dc != 0:
                return (0, 1 if dc > 0 else -1)
            return (0, 0)

        candidates = []
        if (brs.is_safe(unsafe_grid, elbow_a)
                and brs.segment_clear(unsafe_grid, prev, elbow_a)
                and brs.segment_clear(unsafe_grid, elbow_a, pt)):
            candidates.append(elbow_a)
        if (brs.is_safe(unsafe_grid, elbow_b)
                and brs.segment_clear(unsafe_grid, prev, elbow_b)
                and brs.segment_clear(unsafe_grid, elbow_b, pt)):
            candidates.append(elbow_b)

        if avoid_dir is not None and len(candidates) == 2:
            reverse_dir = (-avoid_dir[0], -avoid_dir[1])
            filtered = [c for c in candidates if first_leg_dir(c) != reverse_dir]
            if filtered:
                candidates = filtered

        if candidates:
            return [candidates[0]]

    path = astar_4connected(unsafe_grid, prev, pt)
    if path is not None:
        return path[1:-1] if len(path) > 2 else []

    warnings.append(f"No axis-aligned route found {prev}->{pt}; "
                     f"falling back to the diagonal connector here.")
    return brs.connect_points(unsafe_grid, prev, pt, warnings)


def simplify_path_axis_aligned(path_px, unsafe_grid, essential, max_lookahead=25):
    """Same greedy string-pulling as coverage_common.simplify_path, but a
    jump from i to j is only accepted if that segment is axis-aligned --
    otherwise back off to a smaller j rather than flattening a Manhattan
    corner into a diagonal shortcut."""
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
        while j > i + 1:
            same_row = path_px[i][0] == path_px[j][0]
            same_col = path_px[i][1] == path_px[j][1]
            if (same_row or same_col) and brs.segment_clear(unsafe_grid, path_px[i], path_px[j]):
                break
            j -= 1
        out.append(path_px[j])
        i = j
    return out


def _dir_between(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    if dr != 0:
        return (1 if dr > 0 else -1, 0)
    if dc != 0:
        return (0, 1 if dc > 0 else -1)
    return None


def restitch_manhattan(path_px, unsafe_grid, warnings):
    """Re-run the Manhattan connector between every consecutive pair in an
    already-built path, tracking incoming direction so it never
    introduces a backtrack (see connect_points_manhattan's avoid_dir)."""
    if len(path_px) < 2:
        return path_px
    out = [path_px[0]]
    last_dir = None
    for pt in path_px[1:]:
        added = connect_points_manhattan(unsafe_grid, out[-1], pt, warnings, avoid_dir=last_dir)
        for p in added:
            d = _dir_between(out[-1], p)
            if d is not None:
                last_dir = d
            out.append(p)
        d = _dir_between(out[-1], pt)
        if d is not None:
            last_dir = d
        out.append(pt)
    return out
