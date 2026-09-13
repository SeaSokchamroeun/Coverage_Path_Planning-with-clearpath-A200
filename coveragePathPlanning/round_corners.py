#!/usr/bin/env python3
"""
round_corners.py

Post-processes a coverage_waypoints_*.yaml, replacing sharp lane-end
corners with smooth circular arcs so the controller can flow through
each turn at speed instead of stopping to rotate in place.

WHY: a boustrophedon plan connects lane N's far end to lane N+1's near
end with two waypoints at (nearly) the same cross-axis coordinate. Driving
between them forces a ~180 deg in-place reversal -- which MPPI struggles
to execute (the "stall then goToPose recovery" pattern), and every
recovery curves off the straight line, inflating lateral RMSE. Replacing
the corner with an arc removes the discontinuity at the source.

Each interior waypoint whose incoming and outgoing headings differ by
more than --min-angle is replaced by an arc of radius --radius, tangent
to both segments, sampled every --arc-step metres. The arc is only
accepted if every sampled point stays clear on the map's inflated safe
grid -- otherwise that corner is left as-is (safety first).

Usage:
    python3 round_corners.py \
        --in coverage_waypoints_bcd.yaml \
        --map office_mapEmpty.yaml \
        --out coverage_waypoints_bcd_smooth.yaml \
        --radius 0.4 --arc-step 0.1
"""
import argparse
import math
import yaml
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_safe_grid(map_yaml, robot_radius, margin):
    with open(map_yaml) as f:
        m = yaml.safe_load(f)
    import os
    pgm = os.path.join(os.path.dirname(os.path.abspath(map_yaml)), m["image"])
    img = np.array(Image.open(pgm))
    res = float(m["resolution"])
    ox, oy, _ = m["origin"]
    occ = img < (m.get("occupied_thresh", 0.65) * 255)
    free = img > (m.get("free_thresh", 0.196) * 255)
    # inflate obstacles by robot_radius + margin (same as the generator)
    infl_px = int(round((robot_radius + margin) / res))
    dist = ndi.distance_transform_edt(~occ)
    safe = (dist > infl_px) & free
    h = img.shape[0]

    def to_px(x, y):
        c = int(round((x - ox) / res))
        r = int(round(h - 1 - (y - oy) / res))
        return r, c

    def is_safe(x, y):
        r, c = to_px(x, y)
        if 0 <= r < safe.shape[0] and 0 <= c < safe.shape[1]:
            return bool(safe[r, c])
        return False

    return is_safe


def ang(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def norm(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def round_corner(p_prev, p, p_next, radius, arc_step):
    """Return a list of arc points replacing the sharp corner at p, or
    None if geometry doesn't allow it."""
    v_in = (p[0] - p_prev[0], p[1] - p_prev[1])
    v_out = (p_next[0] - p[0], p_next[1] - p[1])
    len_in = math.hypot(*v_in)
    len_out = math.hypot(*v_out)
    if len_in < 1e-6 or len_out < 1e-6:
        return None

    h_in = math.atan2(v_in[1], v_in[0])
    h_out = math.atan2(v_out[1], v_out[0])
    turn = norm(h_out - h_in)
    if abs(turn) < 1e-3:
        return None

    # distance back from the corner where the arc starts/ends (tangent dist)
    half = abs(turn) / 2.0
    tan_dist = radius / math.tan(math.pi / 2 - half) if half < math.pi/2 else radius
    tan_dist = radius * math.tan(half)
    # cap so we don't eat past the segment endpoints
    tan_dist = min(tan_dist, 0.45 * len_in, 0.45 * len_out)
    if tan_dist < 1e-3:
        return None

    ux_in = v_in[0] / len_in
    uy_in = v_in[1] / len_in
    ux_out = v_out[0] / len_out
    uy_out = v_out[1] / len_out

    start = (p[0] - ux_in * tan_dist, p[1] - uy_in * tan_dist)
    end = (p[0] + ux_out * tan_dist, p[1] + uy_out * tan_dist)

    # arc centre: offset perpendicular from start, toward turn direction
    # bulge the arc toward the INSIDE of the turn (into open room), which is
    # the natural U-turn direction; turn sign already encodes this
    sign = 1.0 if turn > 0 else -1.0
    # actual radius consistent with the capped tan_dist
    r_eff = tan_dist / math.tan(half)
    perp_in = (-uy_in * sign, ux_in * sign)
    centre = (start[0] + perp_in[0] * r_eff, start[1] + perp_in[1] * r_eff)

    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    sweep = norm(a1 - a0)
    arc_len = abs(sweep) * r_eff
    n = max(2, int(round(arc_len / arc_step)))

    pts = []
    for i in range(n + 1):
        t = i / n
        a = a0 + sweep * t
        pts.append((centre[0] + r_eff * math.cos(a),
                    centre[1] + r_eff * math.sin(a)))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.4,
                    help="Nominal U-turn radius in metres.")
    ap.add_argument("--arc-step", type=float, default=0.1,
                    help="Spacing of sampled arc points in metres.")
    ap.add_argument("--min-angle", type=float, default=30.0,
                    help="Only round corners sharper than this (degrees).")
    args = ap.parse_args()

    doc = yaml.safe_load(open(args.inp))
    wps = doc["waypoints"]
    prov = doc.get("generated_with", {})
    rr = float(prov.get("robot_radius", 0.4))
    mg = float(prov.get("extra_clearance", 0.1))
    is_safe = load_safe_grid(args.map, rr, mg)

    pts = [(w["x"], w["y"]) for w in wps]
    min_turn = math.radians(args.min_angle)

    out_pts = [pts[0]]
    rounded = 0
    skipped_unsafe = 0
    for i in range(1, len(pts) - 1):
        p_prev, p, p_next = pts[i - 1], pts[i], pts[i + 1]
        turn = abs(norm(ang(p, p_next) - ang(p_prev, p)))
        if turn < min_turn:
            out_pts.append(p)
            continue
        arc = round_corner(p_prev, p, p_next, args.radius, args.arc_step)
        if arc is None:
            out_pts.append(p)
            continue
        if not all(is_safe(x, y) for x, y in arc):
            skipped_unsafe += 1
            out_pts.append(p)   # keep the sharp corner rather than risk a wall
            continue
        out_pts.extend(arc)
        rounded += 1
    out_pts.append(pts[-1])

    # dedupe consecutive identical points
    deduped = [out_pts[0]]
    for q in out_pts[1:]:
        if math.hypot(q[0] - deduped[-1][0], q[1] - deduped[-1][1]) > 1e-4:
            deduped.append(q)

    # reassign yaw from heading to next point
    new_wps = []
    for i, (x, y) in enumerate(deduped):
        if i < len(deduped) - 1:
            nx, ny = deduped[i + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = new_wps[-1]["yaw"] if new_wps else 0.0
        new_wps.append({"x": round(float(x), 4),
                        "y": round(float(y), 4),
                        "yaw": round(float(yaw), 6)})

    out_doc = {"frame_id": doc.get("frame_id", "map"),
               "generated_with": {**prov, "postprocess": "round_corners.py",
                                   "arc_radius": args.radius},
               "waypoints": new_wps}
    with open(args.out, "w") as f:
        yaml.safe_dump(out_doc, f, sort_keys=False)

    print(f"corners rounded : {rounded}")
    print(f"skipped (unsafe): {skipped_unsafe} (left as sharp corners)")
    print(f"waypoints       : {len(wps)} -> {len(new_wps)}")
    print(f"wrote           : {args.out}")


if __name__ == "__main__":
    main()
