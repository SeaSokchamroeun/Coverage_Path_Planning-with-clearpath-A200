#!/usr/bin/env python3
"""
analyze_trajectory.py

Grades an executed run against the planned path.

Consumes the CSV from record_run.py plus the coverage_waypoints.yaml that was
driven, and answers the questions you cannot answer by watching RViz:

  cross-track error   how far off the planned line the robot actually was.
                      Judged against lane spacing, since that is what decides
                      whether the sweep left gaps.

  backtracking        the robot's projection onto the planned path moving
                      BACKWARDS. This is the metric for "drove half a lane
                      then came back" -- it separates genuine reversal from
                      a robot that merely wandered laterally.

  reversals           sign changes in commanded linear velocity. Distinguishes
                      the controller deliberately backing up from Nav2's
                      recovery behaviours doing it.

  time budget         driving / reversing / rotating in place / stopped.
                      A sweep that spends 40% of its time rotating in place
                      is a controller problem, not a localization problem.

GROUND TRUTH
    When record_run.py was given --truth-topic, the CSV carries the Gazebo
    ground-truth pose alongside AMCL's. This tool then reports THREE numbers
    instead of one, and the difference between them is the whole point:

        AMCL  vs plan   what the robot BELIEVED its tracking error was
        TRUTH vs plan   what the tracking error ACTUALLY was  <- coverage
        AMCL  vs truth  localization error

    A single "distance from the planned path" figure conflates the first two,
    and they call for completely different fixes. If AMCL is 12cm off in x and
    the controller tracks perfectly in AMCL's own frame, the robot is
    physically 12cm off the lane and there is a real coverage hole. If the two
    errors partially cancel, there may not be. You cannot tell without truth.

    Everything degrades gracefully: with no truth columns the tool behaves
    exactly as it did before, with a note saying so.

Usage:
  python3 analyze_trajectory.py --track ~/run5_track.csv \
      --waypoints ../config/coverage_waypoints.yaml \
      --map ../maps/map.yaml \
      --out ../config/run5.png
"""

import argparse
import csv
import math
import os

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Fallback only. The real spacing is MEASURED from the waypoint file, because
# the generator quantises lane spacing to whole map pixels and a 0.26m request
# is emitted as 0.25m at 0.05 m/px. Cross-track error is judged against half
# this number, so getting it wrong by 1cm moves the gap threshold by 5mm.
DEFAULT_LANE_SPACING_M = 0.57  # A200: coverage_width 0.67 - overlap 0.10.
                               # Only a FALLBACK -- generate_room_coverage_bcd_v2.py
                               # writes generated_with.swath_width, which wins.


def load_track(path):
    """Read the recorder CSV. Ground-truth columns are optional: files written
    before record_run.py grew --truth-topic have six columns, and rows
    recorded before the first truth message arrived have them empty."""
    t, x, y, yaw, v, w = [], [], [], [], [], []
    tx, ty, tyaw = [], [], []
    has_truth = False
    with open(path) as f:
        rdr = csv.DictReader(f)
        truth_cols = 'true_x' in (rdr.fieldnames or [])
        for row in rdr:
            t.append(float(row['t']))
            x.append(float(row['x']))
            y.append(float(row['y']))
            yaw.append(float(row['yaw']))
            v.append(float(row['cmd_v']))
            w.append(float(row['cmd_w']))
            if truth_cols and row.get('true_x', '') not in ('', None):
                tx.append(float(row['true_x']))
                ty.append(float(row['true_y']))
                tyaw.append(float(row['true_yaw']))
                has_truth = True
            else:
                tx.append(np.nan)
                ty.append(np.nan)
                tyaw.append(np.nan)
    if not t:
        raise SystemExit(f"{path} contains no samples.")
    truth = (np.array(tx), np.array(ty), np.array(tyaw)) if has_truth else None
    return (np.array(t), np.array(x), np.array(y),
            np.array(yaw), np.array(v), np.array(w), truth)


def load_waypoints(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    wps = data['waypoints']
    pts = np.array([[float(p['x']), float(p['y'])] for p in wps])
    return pts, data.get('generated_with', {})


def segment_arclengths(pts):
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def describe_sweep(pts, min_lane_m=1.0):
    """Work out which axis the lanes run along, and the REALISED lane spacing.

    Both matter and neither can be taken from the provenance block. The
    generator supports --sweep-axis, so orientation is not fixed; and it
    quantises spacing to whole map pixels, so a swath_width of 0.26 is emitted
    as 0.25.

    Lanes are the long segments. Whichever of dx or dy dominates them is the
    lane axis; spacing is then the median gap between distinct waypoint
    coordinates on the OTHER axis.

    Returns (axis, spacing_m, lane_count).
    """
    d = np.diff(pts, axis=0)
    length = np.hypot(d[:, 0], d[:, 1])
    lanes = length > min_lane_m
    if not lanes.any():
        return 'y', DEFAULT_LANE_SPACING_M, 0

    horiz = float((np.abs(d[lanes, 0]) > np.abs(d[lanes, 1])).mean())
    axis = 'x' if horiz > 0.5 else 'y'

    # Spacing is measured perpendicular to the lanes.
    coords = np.unique(np.round(pts[:, 1 if axis == 'x' else 0], 3))
    spacing = (float(np.median(np.diff(coords))) if len(coords) > 2
               else DEFAULT_LANE_SPACING_M)
    return axis, spacing, int(lanes.sum())


def project_to_path(px, py, pts, cum, last_i=0, back=2, fwd=3,
                    reacquire_m=1.0):
    """Nearest point on the planned polyline, searched near the last match.

    A WINDOWED search, not a global one, and the distinction is essential on a
    boustrophedon path. Lanes here sit 0.25m apart, so a robot tracking lane 5
    is within 0.13m of lane 4 and lane 6 as well. A global nearest-segment
    match picks arbitrarily among them, which makes arc length jump between
    distant parts of the route and manufactures backtrack events that never
    happened. Searching a window around the previously matched segment tracks
    the lane the robot is actually on.

    The window is narrow, but NOT as narrow as one segment forward. The
    connectors between lanes are only one lane spacing long -- 0.25m against
    6.6m lanes -- and whenever the robot is laterally offset by more than half
    a lane spacing, the connector is never the nearest segment. With fwd=1 the
    match then cannot step over it, gets stuck on the lane it is on, and
    measures every later lane against that one. Validated against a synthetic
    run with 15cm of injected lateral error: fwd=1 reported 41.5cm and never
    advanced past segment 27 of 30; fwd=3 reports 8.6cm.

    fwd=3 is enough to clear a connector and reach the next lane, and still
    far too small to escape to an arbitrary parallel lane 20m along the route
    -- which is what a wide or global window does, and which matters most at
    the very start where the transit leg from spawn cuts diagonally across the
    whole lane field.

    The fallback searches FORWARD ONLY, never globally. A global fallback
    reported 199.67m of total regression against a 103.9m path, including a
    single 77m "backtrack" -- arithmetically impossible, and the signature of
    the MATCH jumping rather than the robot moving. Forward-only keeps arc
    length monotonic, so a genuine large cross-track error is reported as a
    large error instead of being silently re-attributed to another lane.

    Returns (perpendicular distance, arc length along path, matched index).
    """
    n = len(pts) - 1
    lo = max(0, last_i - back)
    hi = min(n, last_i + fwd + 1)

    def best_in(lo, hi):
        a = pts[lo:hi]
        b = pts[lo + 1:hi + 1]
        ab = b - a
        ab_len2 = (ab ** 2).sum(axis=1)
        ab_len2 = np.where(ab_len2 == 0, 1e-12, ab_len2)
        ap = np.stack([px - a[:, 0], py - a[:, 1]], axis=1)
        tt = np.clip((ap * ab).sum(axis=1) / ab_len2, 0.0, 1.0)
        closest = a + tt[:, None] * ab
        d = np.hypot(closest[:, 0] - px, closest[:, 1] - py)
        k = int(np.argmin(d))
        return d[k], lo + k, tt[k], math.sqrt(ab_len2[k])

    d, i, tt, seglen = best_in(lo, hi)
    if d > reacquire_m:
        d, i, tt, seglen = best_in(last_i, n)
    return d, cum[i] + tt * seglen, i


def project_series(px, py, pts, cum):
    """Run project_to_path over a whole trajectory, carrying the match index
    forward. Returns (cross-track array, arc-length array)."""
    xte = np.empty(len(px))
    s = np.empty(len(px))
    last_i = 0
    for i in range(len(px)):
        xte[i], s[i], last_i = project_to_path(px[i], py[i], pts, cum, last_i)
    return xte, s


def find_backtracks(s, min_regress_m):
    """Spans where progress along the path runs backwards.

    Uses a running maximum: a backtrack starts when s drops more than
    min_regress_m below the furthest point reached, and ends when s climbs
    back to that mark. Lateral wander does not trigger this, because
    wandering sideways leaves s roughly unchanged.
    """
    events = []
    peak = s[0]
    start = None
    for i in range(1, len(s)):
        if s[i] > peak:
            if start is not None:
                events.append((start, i, peak - s[start:i].min()))
                start = None
            peak = s[i]
        elif start is None and peak - s[i] > min_regress_m:
            start = i
    if start is not None:
        events.append((start, len(s) - 1, peak - s[start:].min()))
    return events


def rmse_of(a):
    return float(np.sqrt((a ** 2).mean())) if a.size else float('nan')


def ground_truth_block(truth, x, y, yaw, pts, cum, seg, args):
    """Score the ground-truth trajectory and the AMCL-vs-truth error.

    FRAME ALIGNMENT. The planned path lives in the map frame; ground truth is
    in the Gazebo world frame. The map frame was fixed wherever the robot
    happened to be when SLAM started, so the two need not coincide. Comparing
    truth directly against the plan without correcting would report the frame
    offset as tracking error.

    The correction is the median of (AMCL - truth), which assumes AMCL is
    unbiased on average. That is reasonable -- AMCL is anchored to the map by
    construction, and a constant bias would have to come from the map itself.
    Measured offsets in this project ran between -8.3 and +1.4 cm: small, but
    not zero. The offset is always reported so it stays visible, and
    --truth-offset overrides it with a surveyed value.
    """
    tx, ty, tyaw = truth
    ok = ~np.isnan(tx)

    ex = x[ok] - tx[ok]
    ey = y[ok] - ty[ok]
    if args.truth_offset is not None:
        mx, my = args.truth_offset
    else:
        mx, my = float(np.median(ex)), float(np.median(ey))

    cx, cy = ex - mx, ey - my
    cerr = np.hypot(cx, cy)
    dyaw = np.arctan2(np.sin(yaw[ok] - tyaw[ok]), np.cos(yaw[ok] - tyaw[ok]))

    # Align truth into the map frame, then score it against the plan exactly
    # as the AMCL trajectory is scored. Where truth is missing, fall back to
    # the AMCL pose so the series stays contiguous -- a NaN would break the
    # projection's forward-carrying match index for every later sample.
    fill_x = np.where(ok, tx + mx, x)
    fill_y = np.where(ok, ty + my, y)
    xte_t, s_t = project_series(fill_x, fill_y, pts, cum)

    return {
        'mx': mx, 'my': my,
        'mean_err': float(cerr.mean()), 'max_err': float(cerr.max()),
        'sd_x': float(cx.std()), 'sd_y': float(cy.std()),
        'yaw_rms': float(np.sqrt((dyaw ** 2).mean())),
        'n': int(ok.sum()), 'n_total': int(len(tx)),
        'x': fill_x, 'y': fill_y, 'xte': xte_t, 's': s_t,
        'rmse_true': rmse_of(xte_t[seg]),
    }


def analyse(args):
    t, x, y, yaw, v, w, truth = load_track(args.track)
    pts, prov = load_waypoints(args.waypoints)
    cum = segment_arclengths(pts)
    planned_len = cum[-1]

    axis, measured_lane, n_lanes = describe_sweep(pts)
    lane = args.lane_spacing if args.lane_spacing else measured_lane
    prov_lane = float(prov['swath_width']) if 'swath_width' in prov else None

    xte, s = project_series(x, y, pts, cum)

    driven = float(np.hypot(np.diff(x), np.diff(y)).sum())
    dt = np.diff(t, prepend=t[0])

    sign = np.sign(np.where(np.abs(v) > 0.02, v, 0.0))
    nz = sign[sign != 0]
    reversals = int((np.diff(nz) != 0).sum() // 2) if len(nz) > 1 else 0

    backtracks = find_backtracks(s, args.min_regress)

    # run_coverage.py drives home to HOME_POSE after the last waypoint. That
    # leg runs the length of the room backwards, so path progress collapses
    # from ~100m to ~15m and the detector reads it as one enormous regression.
    # It is the script working as designed, not a tracking failure. Cut at the
    # FIRST sample past the threshold: the homeward regression begins the
    # instant progress peaks, so keying off the last one leaves it inside.
    done = np.flatnonzero(s >= args.complete_frac * planned_len)
    completed = bool(len(done))
    if completed:
        home_i = int(done[0])
        dropped = [b for b in backtracks if b[0] >= home_i]
        backtracks = [b for b in backtracks if b[0] < home_i]
    else:
        # Clamp to the last valid index. A run that never reaches
        # complete_frac of the planned length is exactly the run you most
        # need the report for, and home_i = len(s) crashes the t[home_i]
        # lookup below before anything prints.
        home_i, dropped = len(s) - 1, []
    bt_total = sum(b[2] for b in backtracks)

    # ---- Scoring window --------------------------------------------------
    # Scoped to the coverage path only. The transit from spawn to waypoint 0
    # and the return-home leg are not part of the planned sweep and carry
    # metre-scale lateral error by construction; including them would swamp
    # the average and measure the wrong thing.
    d0 = np.hypot(x - pts[0, 0], y - pts[0, 1])
    on = np.flatnonzero(d0 < args.start_band)
    start_i = int(on[0]) if len(on) else 0
    seg = slice(start_i, home_i)
    xte_cov = xte[seg]
    dt_cov = dt[seg]
    rmse = rmse_of(xte_cov)

    # ---- Motion budget, SCOPED TO THE SCORED WINDOW ----------------------
    # These were previously computed over the WHOLE recording, which is wrong
    # whenever the recorder is left running after the route finishes. A
    # parked robot counts as "stopped", so the figure measured how long the
    # operator took to press Ctrl-C rather than anything about the run.
    #
    # Measured on a real recording: 1850 s captured for a 678 s route, so
    # 63% of the file was a stationary robot. The tool reported 61.5%
    # stopped and 35.6% driving forward; scoped to the route it is 2.8% and
    # 89.6%. Every cross-run comparison of these figures made before this
    # fix is unreliable, because the contamination varies with how long each
    # recording was left open.
    v_cov, w_cov = v[seg], w[seg]
    moving = np.abs(v_cov) > 0.02
    turning = np.abs(w_cov) > 0.05
    t_fwd = float(dt_cov[(v_cov > 0.02)].sum())
    t_rev = float(dt_cov[(v_cov < -0.02)].sum())
    t_spin = float(dt_cov[(~moving) & turning].sum())
    t_idle = float(dt_cov[(~moving) & (~turning)].sum())
    total = max(float(dt_cov.sum()), 1e-9)
    idle_tail = float(t[-1] - t[home_i])

    gt = None
    if truth is not None:
        gt = ground_truth_block(truth, x, y, yaw, pts, cum, seg, args)

    # ======================================================================
    print("=" * 70)
    print(f"track     : {args.track}  ({len(t)} samples, {total:.0f}s)")
    print(f"waypoints : {args.waypoints}  ({len(pts)} points, "
          f"{planned_len:.1f}m planned)")
    if prov:
        print(f"platform  : {prov.get('platform','?')}  "
              f"swath {prov.get('swath_width','?')}m")
    print(f"sweep     : {n_lanes} lanes along {axis.upper()}, spacing "
          f"{lane * 100:.1f}cm  (measured from the waypoints)")
    if prov_lane and abs(prov_lane - measured_lane) > 0.005:
        print(f"            provenance says {prov_lane * 100:.0f}cm; the "
              f"generator quantises to whole map pixels")
    print(f"            lateral error on these lanes is error in "
          f"{'Y' if axis == 'x' else 'X'}")
    if not completed:
        print()
        print(f"  *** RUN INCOMPLETE: progress never reached "
              f"{args.complete_frac * 100:.0f}% of {planned_len:.1f}m.")
        print("      Scoring runs to the last sample. Treat everything below")
        print("      as describing a partial run.")
    print()
    print(f"distance driven      : {driven:.1f} m "
          f"({driven / planned_len:.2f}x planned)")
    print()

    # ---- The pass/fail metric -------------------------------------------
    if gt:
        verdict = 'PASS' if gt['rmse_true'] < args.rmse_target else 'FAIL'
        print("LATERAL RMSE  --  believed vs actual")
        print(f"  AMCL  vs plan      : {rmse * 100:6.1f} cm   "
              f"what the robot BELIEVED")
        print(f"  TRUTH vs plan      : {gt['rmse_true'] * 100:6.1f} cm   "
              f"what ACTUALLY happened  <-- coverage")
        print(f"  target             : <{args.rmse_target * 100:5.0f} cm   "
              f"{verdict}")
    else:
        verdict = 'PASS' if rmse < args.rmse_target else 'FAIL'
        print(f"LATERAL RMSE  ({verdict})")
        print(f"  RMSE               : {rmse * 100:.1f} cm"
              f"   (target < {args.rmse_target * 100:.0f} cm)")
        print("  No ground-truth columns: this is AMCL's BELIEF, not")
        print("  necessarily where the robot was. Re-record with")
        print("  record_run.py --truth-topic to separate the two.")
    print(f"  scored over        : t={t[start_i]:.0f}s..{t[home_i]:.0f}s, "
          f"{xte_cov.size} samples")
    print(f"  excluded           : {start_i} transit + "
          f"{len(t) - home_i - 1} return-home samples")
    print()

    if gt:
        print(f"LOCALIZATION ERROR   (AMCL minus ground truth, "
              f"{gt['n']} paired samples)")
        print(f"  frame offset       : x {gt['mx'] * 100:+.1f} cm, "
              f"y {gt['my'] * 100:+.1f} cm   (removed below)")
        if math.hypot(gt['mx'], gt['my']) > 0.10:
            print("    ^ large. Probably a map-vs-world origin offset rather")
            print("      than error, but worth confirming.")
        print(f"  mean |error|       : {gt['mean_err'] * 100:.1f} cm")
        print(f"  max  |error|       : {gt['max_err'] * 100:.1f} cm")
        print(f"  sd in x            : {gt['sd_x'] * 100:.1f} cm")
        print(f"  sd in y            : {gt['sd_y'] * 100:.1f} cm")
        print(f"  yaw error (rms)    : {math.degrees(gt['yaw_rms']):.2f} deg")
        # Guard the ratio: with both errors near zero it is meaningless, and
        # dividing by an epsilon produces a confident nonsense verdict.
        worst_sd = max(gt['sd_x'], gt['sd_y'])
        ratio = gt['sd_x'] / gt['sd_y'] if gt['sd_y'] > 1e-4 else float('inf')
        print(f"  x/y anisotropy     : "
              f"{ratio:.1f}x" if worst_sd > 0.01 else
              f"  x/y anisotropy     : n/a (both axes under 1cm)")
        if worst_sd > 0.01 and (ratio > 3.0 or ratio < 0.33):
            worse = 'x' if ratio > 1 else 'y'
            print(f"    ^ {worse} is far worse than the other axis. That is the")
            print("      observability signature: check whether the room is")
            print("      longer than twice the lidar range along that axis,")
            print("      and orient lanes so lateral error avoids it.")
        print()

    print("CROSS-TRACK ERROR  (distance from the planned line, AMCL frame)")
    print(f"  mean               : {xte_cov.mean() * 100:.1f} cm")
    print(f"  95th percentile    : {np.percentile(xte_cov, 95) * 100:.1f} cm")
    print(f"  max                : {xte_cov.max() * 100:.1f} cm")
    # Scoped to the same window as the RMSE. Using the whole array here
    # counted the transit and return-home legs, which are off-path by
    # construction and inflated this figure.
    beyond = 100 * dt_cov[xte_cov > lane / 2].sum() / max(dt_cov.sum(), 1e-9)
    print(f"  time beyond {lane / 2 * 100:.0f}cm   : {beyond:.1f}% "
          f"(half lane spacing -- beyond this the swath leaves gaps)")
    # Cross-track error SATURATES at roughly half the lane spacing, and there
    # is no fixing that: once the robot is more than half a spacing off, it is
    # geometrically closer to the neighbouring lane, and the path simply does
    # not contain the information to tell "20cm off lane 5" from "5cm off
    # lane 4". Calibrated against synthetic runs at 25cm spacing: 2cm injected
    # reads 2.0cm, 4cm reads 4.0cm, 8cm reads 6.7cm, 20cm reads 9.8cm.
    #
    # This matters here because the saturation value (~10cm) is also the pass
    # criterion. A badly failing run can look borderline. Say so out loud.
    if beyond > 10.0:
        print()
        print(f"  *** {beyond:.0f}% of the scored run is beyond half a lane "
              f"spacing.")
        print("      Cross-track error saturates near that value -- the robot")
        print("      is closer to a neighbouring lane than its own, and the")
        print("      path cannot distinguish the two. The RMSE above is a")
        print("      LOWER BOUND on the true error, not an estimate of it.")
        if not gt:
            print("      Record with ground truth to get an unambiguous number.")
    print()
    print("BACKTRACKING  (progress along the path running backwards)")
    print(f"  events             : {len(backtracks)}")
    if dropped:
        print(f"  (excluded {len(dropped)} event(s) on the return-home leg "
              f"after t={t[home_i]:.0f}s)")
    print(f"  total regression   : {bt_total:.2f} m")
    if bt_total > planned_len:
        print("    *** Total regression exceeds the planned path length.")
        print("        That is impossible for real motion, and means the path")
        print("        projection is jumping between lanes. Do NOT trust the")
        print("        backtrack or cross-track figures from this run.")
    if backtracks:
        worst = max(backtracks, key=lambda b: b[2])
        print(f"  worst              : {worst[2]:.2f} m at "
              f"t={t[worst[0]]:.0f}s, near "
              f"({x[worst[0]]:.2f}, {y[worst[0]]:.2f})")
        print("  first 10:")
        for i0, i1, mag in backtracks[:10]:
            print(f"    t={t[i0]:6.0f}s  {mag:5.2f}m back  at "
                  f"({x[i0]:6.2f},{y[i0]:6.2f})")
    print()
    print("MOTION  (scored window only -- excludes the parked tail)")
    print(f"  scored over        : {total:.0f}s of {t[-1]:.0f}s recorded")
    if idle_tail > 30.0:
        print(f"  ({idle_tail:.0f}s of stationary recording after the route "
              f"finished, excluded)")
    print(f"  commanded reversals: {reversals}")
    print(f"  driving forward    : {100 * t_fwd / total:4.1f}%")
    print(f"  driving backward   : {100 * t_rev / total:4.1f}%")
    print(f"  rotating in place  : {100 * t_spin / total:4.1f}%")
    print(f"  stopped            : {100 * t_idle / total:4.1f}%")
    print()
    print("HOW TO READ THIS")
    print(f"  Clean run: 95th-pct XTE under {lane / 2 * 100:.0f}cm, backtrack")
    print("  events in single digits, distance driven under ~1.15x planned,")
    print("  rotating-in-place under ~15%.")
    print()
    print("  Many backtracks + high reversal count = the controller is")
    print("  fighting the path, or Nav2 recoveries are firing. Look at the")
    print("  run log for 'Running spin' / 'Failed to make progress' at the")
    print("  timestamps above.")
    print()
    print("  High XTE but FEW backtracks = tracking error, not oscillation.")
    if gt:
        print()
        print("  TRUTH-vs-plan much BETTER than AMCL-vs-plan means the robot")
        print("  drove the lanes well and AMCL misreported it -- a")
        print("  localization problem. The two being CLOSE means the robot")
        print("  really was off the line -- a controller problem. Read them")
        print("  together; neither number means much alone.")
    print("=" * 70)

    if args.out:
        plot(args, t, x, y, v, s, xte, pts, backtracks, lane, planned_len,
             home_i, gt)


def plot(args, t, x, y, v, s, xte, pts, backtracks, lane, planned_len,
         home_i, gt=None):
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.1, 1.1], hspace=0.28)

    ax = fig.add_subplot(gs[0])
    if args.map:
        try:
            draw_map(ax, args.map)
        except Exception as e:
            # Never lose the whole plot over a missing underlay -- the
            # trajectory is the point, the occupancy grid is context.
            print(f"  (map underlay skipped: {e})")
    ax.plot(pts[:, 0], pts[:, 1], '-', color='0.75', lw=1.2,
            zorder=2, label='planned path')

    segs = np.stack([np.column_stack([x[:-1], y[:-1]]),
                     np.column_stack([x[1:], y[1:]])], axis=1)
    lc = LineCollection(segs, cmap='inferno_r', zorder=3, linewidths=1.8)
    lc.set_array(xte[:-1])
    lc.set_clim(0, max(lane, xte.max()))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label='cross-track error (m), AMCL frame',
                 shrink=0.8)

    if gt:
        ax.plot(gt['x'], gt['y'], '-', color='#1f77b4', lw=0.9, alpha=0.8,
                zorder=3.5, label='ground truth (aligned)')

    for i0, _, _ in backtracks:
        ax.plot(x[i0], y[i0], 'o', ms=7, mfc='none', mec='red', mew=1.6,
                zorder=4)
    if backtracks:
        ax.plot([], [], 'o', ms=7, mfc='none', mec='red', mew=1.6,
                label=f'backtrack start ({len(backtracks)})')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Executed trajectory vs planned path')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, s, lw=1.2, label='AMCL')
    if gt:
        ax2.plot(t, gt['s'], lw=1.0, color='#1f77b4', alpha=0.8,
                 label='ground truth')
    if home_i < len(t) - 1:
        ax2.axvspan(t[home_i], t[-1], color='0.85', lw=0,
                    label='return home (excluded)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.axhline(planned_len, ls=':', color='0.6', lw=1)
    for i0, i1, _ in backtracks:
        ax2.axvspan(t[i0], t[i1], color='red', alpha=0.18, lw=0)
    ax2.set_ylabel('progress along\npath (m)')
    ax2.set_title('Progress along the planned path — flat means stopped, '
                  'downward means backtracking', fontsize=10)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ax3.plot(t, v, lw=0.9)
    ax3.axhline(0, color='0.6', lw=0.8)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('cmd_vel\nlinear.x (m/s)')
    ax3.set_title('Commanded forward velocity — excursions below zero are '
                  'reversals', fontsize=10)
    ax3.grid(alpha=0.3)

    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"Plot -> {args.out}")


def draw_map(ax, map_yaml):
    """Occupancy grid underlay, same convention as plot_coverage.py."""
    from PIL import Image
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(map_yaml)),
                                img_path)
    img = np.array(Image.open(img_path).convert('L'))
    res = float(meta['resolution'])
    ox, oy = meta['origin'][0], meta['origin'][1]
    h, w = img.shape
    ax.imshow(img, cmap='gray', origin='upper', zorder=1, alpha=0.55,
              extent=[ox, ox + w * res, oy, oy + h * res])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True, help='CSV from record_run.py')
    ap.add_argument('--waypoints', required=True,
                    help='The coverage_waypoints.yaml that was driven')
    ap.add_argument('--map', default=None, help='map.yaml for the underlay')
    ap.add_argument('--out', default=None, help='Output PNG')
    ap.add_argument('--lane-spacing', type=float, default=None,
                    help='Override. By default the REALISED spacing is '
                         'measured from the waypoint file, which is more '
                         'reliable than generated_with.swath_width because '
                         'the generator quantises to whole map pixels.')
    ap.add_argument('--truth-offset', type=float, nargs=2, default=None,
                    metavar=('X', 'Y'),
                    help='Map-frame minus world-frame offset in metres. '
                         'Defaults to the median of (AMCL - truth).')
    ap.add_argument('--complete-frac', type=float, default=0.95,
                    help='Fraction of planned length that counts as route '
                         'complete. Everything after the first sample past '
                         'this is the drive home, not a tracking failure.')
    ap.add_argument('--rmse-target', type=float, default=0.10,
                    help='Pass criterion for lateral RMSE, in metres. '
                         'Project target is 0.10 (D435 minimum depth 0.28m '
                         'at a 0.4m wall standoff).')
    ap.add_argument('--start-band', type=float, default=0.20,
                    help='RMSE scoring starts when the robot first comes '
                         'within this distance of waypoint 0, excluding the '
                         'transit from spawn.')
    ap.add_argument('--min-regress', type=float, default=0.10,
                    help='Metres of backwards progress before it counts as a '
                         'backtrack. Below ~0.05 this fires on noise.')
    analyse(ap.parse_args())


if __name__ == '__main__':
    main()
