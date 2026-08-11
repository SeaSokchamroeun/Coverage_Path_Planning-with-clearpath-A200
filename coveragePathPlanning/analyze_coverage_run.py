#!/usr/bin/env python3
"""
analyze_coverage_run.py

Parses the log produced by run_coverage.py (v4 -- segment-at-a-time /
continuous-path-with-replanning version). The old analyzer expected a
per-waypoint '[N/M] segment -> (x, y)' format from an earlier runner
version; v4 doesn't print that at all, which is why it was finding
"No segments parsed". This version matches v4's real print statements:

  Driving continuous path: waypoint N -> M (K waypoints, ...)
  stalled -- cancelling and replanning around it
  replanning (goToPose) to waypoint N to clear the obstruction...
  could not recover near waypoint N -- skipping ahead
  Route finished. N waypoint(s) could not be reached ...
  Unreached waypoint indices: [...]
  Back at (x, y). Done.
"""

import argparse
import re

RE_CONTINUOUS = re.compile(
    r"Driving continuous path: waypoint (\d+) -> (\d+) \((\d+) waypoints")
RE_STALL = re.compile(r"stalled -- cancelling and replanning around it")
RE_REPLAN_HOP = re.compile(
    r"replanning \(goToPose\) to waypoint (\d+) to clear the obstruction")
RE_SKIP = re.compile(r"could not recover near waypoint (\d+) -- skipping ahead")
RE_FINISHED = re.compile(
    r"Route finished\. (\d+) waypoint\(s\) could not be reached")
RE_UNREACHED = re.compile(r"Unreached waypoint indices: (\[.*\])")
RE_HOME_OK = re.compile(r"Back at \(([-\d.]+), ([-\d.]+)\)\. Done\.")
RE_HOME_FAIL = re.compile(r"Failed to return home")
RE_DIAGNOSTIC = re.compile(
    r"\[diagnostic\] stalled during (\d+)->(\d+): seg_total=([\d.]+)m, "
    r"best_dist_remaining=(None|[\d.]+)m?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-log", required=True)
    args = ap.parse_args()

    with open(args.run_log) as f:
        lines = f.readlines()

    continuous_attempts = []
    stalls = 0
    replan_hops = []
    skipped = []
    finished_failed_count = None
    unreached_indices = None
    home_ok = None
    diagnostics = []

    for line in lines:
        m = RE_DIAGNOSTIC.search(line)
        if m:
            idx_from, idx_to, seg_total, best_dist_raw = m.groups()
            best_dist = None if best_dist_raw == "None" else float(best_dist_raw)
            diagnostics.append((int(idx_from), int(idx_to), float(seg_total), best_dist))
            continue
        m = RE_CONTINUOUS.search(line)
        if m:
            continuous_attempts.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        if RE_STALL.search(line):
            stalls += 1
            continue
        m = RE_REPLAN_HOP.search(line)
        if m:
            replan_hops.append(int(m.group(1)))
            continue
        m = RE_SKIP.search(line)
        if m:
            skipped.append(int(m.group(1)))
            continue
        m = RE_FINISHED.search(line)
        if m:
            finished_failed_count = int(m.group(1))
            continue
        m = RE_UNREACHED.search(line)
        if m:
            unreached_indices = m.group(1)
            continue
        m = RE_HOME_OK.search(line)
        if m:
            home_ok = (float(m.group(1)), float(m.group(2)))
            continue
        if RE_HOME_FAIL.search(line):
            home_ok = False

    if not continuous_attempts:
        print("No 'Driving continuous path' lines found -- is this really a "
              "v4 run_coverage.py log? (Note: leading '[INFO] ... Executing "
              "path / Navigating to goal' lines with no matching script "
              "prints are NOT from run_coverage.py -- they're leftover "
              "output from a different/earlier process. Check for those "
              "before this point in the file.)")
        return

    print(f"Continuous-path attempts: {len(continuous_attempts)}")
    for i, (start, end, count) in enumerate(continuous_attempts):
        print(f"  attempt {i}: waypoint {start} -> {end} ({count} waypoints in this call)")

    print(f"\nStalls detected: {stalls}")
    print(f"goToPose replanning hops attempted: {len(replan_hops)} "
          f"{replan_hops if replan_hops else ''}")
    print(f"Waypoints skipped (unrecoverable after MAX_HOPS): {len(skipped)} "
          f"{skipped if skipped else ''}")

    if diagnostics:
        print(f"\n[diagnostic] entries found: {len(diagnostics)} "
              f"(this run is using the patched resume logic)")
        for idx_from, idx_to, seg_total, best_dist in diagnostics:
            bd_str = f"{best_dist:.2f}m" if best_dist is not None else "None"
            print(f"  stalled {idx_from}->{idx_to}: seg_total={seg_total:.2f}m, "
                  f"best_dist_remaining={bd_str}")
    else:
        print("\nNo [diagnostic] lines found in this log -- this run was NOT "
              "executed with the patched run_coverage.py (the one with the "
              "conservative idx+1 resume + diagnostic prints). Check that the "
              "file actually running is the patched version before trusting "
              "the hop targets below.")

    # Flag any hop that skips more than a couple of waypoints -- with the
    # patched script this should never happen (always idx -> idx+1), so if
    # it does, that's strong evidence the OLD unpatched script actually ran.
    if len(continuous_attempts) >= 2:
        large_skips = []
        for i in range(1, len(continuous_attempts)):
            prev_start = continuous_attempts[i - 1][0]
            this_start = continuous_attempts[i][0]
            if this_start - prev_start > 2:
                large_skips.append((prev_start, this_start))
        if large_skips:
            print(f"\nWARNING: {len(large_skips)} large skip-ahead hop(s) detected: "
                  f"{large_skips}")
            print("  A hop this large means many waypoints (whole lanes) were "
                  "never driven via followPath. If you're expecting the patched "
                  "conservative-resume behavior (idx -> idx+1 only), this is a "
                  "strong sign the OLD run_coverage.py is what actually executed.")

    print()
    if finished_failed_count is None:
        print("WARNING: never saw a 'Route finished' line -- run may have "
              "crashed or is incomplete.")
    else:
        print(f"Route finished. {finished_failed_count} waypoint(s) unreachable "
              f"even with replanning fallback.")
        if unreached_indices:
            print(f"Unreached waypoint indices: {unreached_indices}")

    print()
    if home_ok is None:
        print("WARNING: no 'Back at ...' or 'Failed to return home' line found "
              "-- homing result unknown (log may be truncated).")
    elif home_ok is False:
        print("Robot FAILED to return home -- check Nav2 / costmap state.")
    else:
        print(f"Robot returned home successfully at {home_ok}.")


if __name__ == "__main__":
    main()