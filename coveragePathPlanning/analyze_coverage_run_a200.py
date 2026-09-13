#!/usr/bin/env python3
"""
analyze_coverage_run_a200.py

Parses run_nav.log -- the console output of run_coverage.py -- and reports
what actually happened during the drive.

FIXED REGEX
    The TurtleBot analyze_coverage_run.py looks for the literal string
        "stalled -- cancelling and replanning around it"
    but run_coverage.py prints
        "stalled (5.0s without 0.05m of progress) -- cancelling and ..."
    The parenthetical sits between "stalled" and "--", so the pattern never
    matched and the tool reported "Stalls detected: 0" on every run, no
    matter how many stalls occurred. Verified against the exact f-string in
    run_coverage.py. This version tolerates the parenthetical and also
    captures the timeout and progress values from it.

    Two more lines run_coverage.py prints were not parsed at all:
      - "waypoint(s) [...] silently skipped during hop retry"
        These ARE lost coverage and belong in the failure count.
      - "Segment N->M complete."
        Needed to compute a success rate rather than just listing attempts.

Usage
    ros2 run ... | tee run_nav.log        # or: python3 run_coverage.py ... 2>&1 | tee run_nav.log
    python3 analyze_coverage_run_a200.py --run-log run_nav.log

    --waypoints coverage_waypoints_bcd.yaml
        Optional. Lets the report say how much FLOOR the unreached
        waypoints represent, not just how many indices were missed.
"""

import argparse
import ast
import math
import os
import re
import sys


RE_CONTINUOUS = re.compile(
    r"Driving continuous path: waypoint (\d+) -> (\d+) \((\d+) waypoints")
RE_SEGMENT_OK = re.compile(r"Segment (\d+)->(\d+) complete\.")

# "stalled (5.0s without 0.05m of progress) -- cancelling and replanning"
# The (?:\(...\))? makes the parenthetical optional so this also matches
# any older log written before the timeout was made configurable.
RE_STALL = re.compile(
    r"stalled\s*(?:\(([\d.]+)s without ([\d.]+)m of progress\))?\s*--\s*"
    r"cancelling and replanning")

RE_DIAGNOSTIC = re.compile(
    r"\[diagnostic\] stalled during (\d+)->(\d+): seg_total=([\d.]+)m, "
    r"best_dist_remaining=(None|[\d.]+)")
RE_REPLAN_HOP = re.compile(
    r"replanning \(goToPose\) to waypoint (\d+) to clear the obstruction")
RE_SILENT_SKIP = re.compile(
    r"waypoint\(s\) (\[.*?\]) silently skipped during hop retry")
RE_SKIP = re.compile(r"could not recover near waypoint (\d+) -- skipping ahead")
RE_FINISHED = re.compile(
    r"Route finished\. (\d+) waypoint\(s\) could not be reached")
RE_UNREACHED = re.compile(r"Unreached waypoint indices: (\[.*\])")
RE_HOME_OK = re.compile(r"Back at \(([-\d.]+), ([-\d.]+)\)\. Done\.")
RE_HOME_FAIL = re.compile(r"Failed to return home")

# Nav2 recovery behaviours. These are the ground truth for "the stack was
# in trouble" -- run_coverage.py's own stall detector can miss a recovery
# that resolved before its timeout fired.
RE_NAV2_RECOVERY = re.compile(
    r"(Running \w*[Ss]pin|Running \w*BackUp|Running \w*Wait|"
    r"Failed to make progress|aborted|Collision Ahead)")


def parse(lines):
    r = dict(attempts=[], segments_ok=[], stalls=[], diagnostics=[],
             hops=[], silent_skips=[], skipped=[], recoveries=0,
             finished=None, unreached=None, home=None)

    for line in lines:
        m = RE_DIAGNOSTIC.search(line)
        if m:
            a, b, tot, best = m.groups()
            r["diagnostics"].append(
                (int(a), int(b), float(tot),
                 None if best == "None" else float(best)))
            continue
        m = RE_CONTINUOUS.search(line)
        if m:
            r["attempts"].append((int(m.group(1)), int(m.group(2)),
                                  int(m.group(3))))
            continue
        m = RE_SEGMENT_OK.search(line)
        if m:
            r["segments_ok"].append((int(m.group(1)), int(m.group(2))))
            continue
        m = RE_STALL.search(line)
        if m:
            r["stalls"].append((float(m.group(1)) if m.group(1) else None,
                                float(m.group(2)) if m.group(2) else None))
            continue
        m = RE_REPLAN_HOP.search(line)
        if m:
            r["hops"].append(int(m.group(1)))
            continue
        m = RE_SILENT_SKIP.search(line)
        if m:
            try:
                r["silent_skips"].extend(ast.literal_eval(m.group(1)))
            except (ValueError, SyntaxError):
                pass
            continue
        m = RE_SKIP.search(line)
        if m:
            r["skipped"].append(int(m.group(1)))
            continue
        m = RE_FINISHED.search(line)
        if m:
            r["finished"] = int(m.group(1))
            continue
        m = RE_UNREACHED.search(line)
        if m:
            r["unreached"] = m.group(1)
            continue
        m = RE_HOME_OK.search(line)
        if m:
            r["home"] = (float(m.group(1)), float(m.group(2)))
            continue
        if RE_HOME_FAIL.search(line):
            r["home"] = False
            continue
        if RE_NAV2_RECOVERY.search(line):
            r["recoveries"] += 1

    return r


def waypoint_gap_metres(indices, wp_path):
    """How much planned travel the unreached waypoints represent."""
    try:
        import yaml
        with open(wp_path) as f:
            wps = yaml.safe_load(f)["waypoints"]
    except Exception as e:
        print(f"  (could not read {wp_path}: {e})")
        return None
    total = 0.0
    for i in indices:
        if 0 < i < len(wps):
            a, b = wps[i - 1], wps[i]
            total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return total, len(wps)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--waypoints", default=None,
                    help="The waypoints file that was driven. Lets the "
                         "report convert missed indices into missed metres.")
    args = ap.parse_args()

    if not os.path.exists(args.run_log):
        sys.exit(f"ERROR: {args.run_log} not found. Capture it with:\n"
                 "  python3 run_coverage.py --waypoints ... 2>&1 | tee run_nav.log")

    with open(args.run_log, errors="replace") as f:
        lines = f.readlines()

    r = parse(lines)

    if not r["attempts"]:
        print("No 'Driving continuous path' lines found.")
        print("This log did not come from run_coverage.py, or the script "
              "died before its first segment. Nav2's own [INFO] lines are "
              "NOT run_coverage.py output -- check for a crash above.")
        return

    n_att = len(r["attempts"])
    n_ok = len(r["segments_ok"])
    print("=" * 70)
    print(f"RUN LOG: {args.run_log}  ({len(lines)} lines)")
    print("=" * 70)
    print()
    print("SEGMENTS")
    print(f"  followPath attempts     : {n_att}")
    print(f"  completed first try     : {n_ok}  "
          f"({100.0 * n_ok / max(1, n_att):.0f}%)")
    print(f"  stalls                  : {len(r['stalls'])}")
    if r["stalls"] and r["stalls"][0][0]:
        to, eps = r["stalls"][0]
        print(f"    (stall = no {eps}m of progress within {to}s)")
    print(f"  goToPose replan hops    : {len(r['hops'])}"
          + (f"  at {r['hops'][:12]}" if r["hops"] else ""))
    print(f"  Nav2 recovery lines     : {r['recoveries']}  "
          "(spin / backup / wait / abort)")

    lost = sorted(set(r["skipped"]) | set(r["silent_skips"]))
    print()
    print("LOST COVERAGE")
    print(f"  unrecoverable skips     : {len(r['skipped'])} "
          f"{r['skipped'][:12] if r['skipped'] else ''}")
    print(f"  silently skipped in hops: {len(r['silent_skips'])} "
          f"{r['silent_skips'][:12] if r['silent_skips'] else ''}")
    print(f"  distinct waypoints lost : {len(lost)}")
    if args.waypoints and lost:
        got = waypoint_gap_metres(lost, args.waypoints)
        if got:
            metres, n_total = got
            print(f"  planned travel missed   : {metres:.1f} m "
                  f"({100.0 * len(lost) / n_total:.1f}% of {n_total} waypoints)")

    if r["diagnostics"]:
        print()
        print("STALL DIAGNOSTICS")
        for a, b, tot, best in r["diagnostics"][:15]:
            note = ""
            if best is not None and tot > 0 and best > tot * 0.9:
                note = "  <-- barely moved before stalling"
            bs = f"{best:.2f}m" if best is not None else "None"
            print(f"  {a}->{b}: segment {tot:.2f}m, best remaining {bs}{note}")
        if len(r["diagnostics"]) > 15:
            print(f"  ... and {len(r['diagnostics']) - 15} more")

    # A hop should always be idx -> idx+1 under the conservative resume.
    # Only a jump that follows a FAILED segment is suspicious: after a
    # SUCCESSFUL one the runner sets idx = window_end, so advancing by the
    # whole FOLLOW_PATH_LOOKAHEAD is normal and must not be flagged.
    ok_starts = {a for a, _ in r["segments_ok"]}
    big = []
    for i in range(1, n_att):
        prev_start = r["attempts"][i - 1][0]
        jump = r["attempts"][i][0] - prev_start
        if prev_start not in ok_starts and jump > 2:
            big.append((prev_start, r["attempts"][i][0]))

    print()
    print("RESULT")
    if r["finished"] is None:
        print("  !! no 'Route finished' line -- the run crashed or the log "
              "is truncated.")
    else:
        print(f"  route finished, {r['finished']} waypoint(s) unreachable")
        if r["unreached"]:
            print(f"  unreached indices: {r['unreached']}")
    if r["home"] is None:
        print("  !! homing result unknown (log truncated?)")
    elif r["home"] is False:
        print("  !! failed to return home -- check Nav2 / costmap state")
    else:
        print(f"  returned home at {r['home']}")

    print()
    print("HOW TO READ THIS")
    if len(r["stalls"]) > n_att * 0.5:
        print(f"  More than half the segments stalled ({len(r['stalls'])} of "
              f"{n_att}). That is the controller failing to track, not an")
        print("  obstacle problem -- the office world has none. Try a longer")
        print("  --stall-timeout, or a smaller FOLLOW_PATH_LOOKAHEAD.")
    elif len(r["stalls"]) == 0:
        print("  No stalls. The path was tracked cleanly end to end.")
    else:
        print(f"  {len(r['stalls'])} stall(s) out of {n_att} segments is "
              "normal-ish.")
        print("  A stall whose 'best remaining' is close to the full segment")
        print("  length means the robot never started moving at all, which")
        print("  points at the costmap not being ready, not at the path.")
    if big:
        print(f"  !! {len(big)} large skip-ahead(s) after a failed segment:")
        print(f"     {big[:5]} -- whole lanes were never driven. Under the")
        print("     conservative idx -> idx+1 resume this cannot happen, so")
        print("     an older run_coverage.py probably executed.")
    print()
    print("  This grades the NAVIGATION STACK. For how well the robot")
    print("  actually tracked the geometry, run analyze_trajectory.py on the")
    print("  CSV from record_run_a200.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
