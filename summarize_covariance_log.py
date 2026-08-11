#!/usr/bin/env python3
"""
summarize_covariance_log.py

Turns one or more log_localization_covariance.py CSVs into a
report-ready comparison table -- the same "measured comparison rather
than a judgment call" standard the report already applies to coverage
and clearance (Table 4.2/4.3 style).

Single run:
    python3 summarize_covariance_log.py --run baseline:baseline.csv

Multi-arm before/after comparison (this is the one you want for the
report -- baseline AMCL vs tuned AMCL vs SLAM Toolbox, side by side):
    python3 summarize_covariance_log.py \\
        --run baseline:baseline.csv \\
        --run tuned:tuned.csv \\
        --run slam_toolbox:slam_toolbox.csv

Optional: split stats into "near known trouble spots" vs "elsewhere" by
passing the coordinates of the stall points you already have from the
clearance analysis (Section 4.4.2) or the pixel-scale defect (Section
4.5.4). Any point within --radius meters of a logged sample counts as
"near":
    ... --stall-point 10.454,3.2 --stall-point 6.1,1.8 --radius 0.5

With no --stall-point given, only whole-run stats are reported.
"""

import argparse
import csv
import math


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "x": float(r["x"]), "y": float(r["y"]),
                "trace_pos": float(r["trace_pos"]),
                "uncertainty_radius_m": float(r["uncertainty_radius_m"]),
                "flag": r["flag"],
            })
    return rows


def stats_for(rows):
    if not rows:
        return None
    traces = [r["trace_pos"] for r in rows]
    radii = [r["uncertainty_radius_m"] for r in rows]
    n_flagged = sum(1 for r in rows if r["flag"])
    return {
        "n": len(rows),
        "mean_trace": sum(traces) / len(traces),
        "max_trace": max(traces),
        "mean_radius_m": sum(radii) / len(radii),
        "max_radius_m": max(radii),
        "n_flagged": n_flagged,
        "flagged_pct": 100.0 * n_flagged / len(rows),
    }


def near_stall_points(rows, stall_points, radius):
    if not stall_points:
        return rows, []
    near, far = [], []
    for r in rows:
        d = min(math.hypot(r["x"] - sx, r["y"] - sy) for sx, sy in stall_points)
        (near if d <= radius else far).append(r)
    return near, far


def fmt_row(label, s):
    if s is None:
        return f"| {label} | -- | -- | -- | -- | -- |"
    return (f"| {label} | {s['n']} | {s['mean_trace']:.4f} | {s['max_trace']:.4f} "
            f"| {s['mean_radius_m']:.3f} | {s['n_flagged']} ({s['flagged_pct']:.1f}%) |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                     help="label:path.csv, repeatable")
    ap.add_argument("--stall-point", action="append", default=[],
                     help="x,y in map frame, repeatable")
    ap.add_argument("--radius", type=float, default=0.5,
                     help="meters -- distance from a stall point counted as 'near'")
    args = ap.parse_args()

    stall_points = []
    for sp in args.stall_point:
        x, y = sp.split(",")
        stall_points.append((float(x), float(y)))

    runs = []
    for spec in args.run:
        label, path = spec.split(":", 1)
        runs.append((label, load_csv(path)))

    print("## Whole-run localization confidence\n")
    print("| Run | samples | mean trace_pos (m^2) | max trace_pos (m^2) "
          "| mean uncertainty radius (m) | confidence-drop events |")
    print("|---|---|---|---|---|---|")
    for label, rows in runs:
        print(fmt_row(label, stats_for(rows)))

    if stall_points:
        print(f"\n## Split by proximity to known trouble spots "
              f"(within {args.radius}m of {len(stall_points)} point(s))\n")
        print("### Near trouble spots\n")
        print("| Run | samples | mean trace_pos (m^2) | max trace_pos (m^2) "
              "| mean uncertainty radius (m) | confidence-drop events |")
        print("|---|---|---|---|---|---|")
        for label, rows in runs:
            near, _ = near_stall_points(rows, stall_points, args.radius)
            print(fmt_row(label, stats_for(near)))

        print("\n### Elsewhere\n")
        print("| Run | samples | mean trace_pos (m^2) | max trace_pos (m^2) "
              "| mean uncertainty radius (m) | confidence-drop events |")
        print("|---|---|---|---|---|---|")
        for label, rows in runs:
            _, far = near_stall_points(rows, stall_points, args.radius)
            print(fmt_row(label, stats_for(far)))

        print("\nIf 'near trouble spots' shows a materially higher mean/max "
              "trace_pos and more confidence-drop events than 'elsewhere' for "
              "the baseline run, and that gap shrinks or disappears in the "
              "tuned/SLAM Toolbox runs, that's the quantified before/after "
              "evidence for the report.")


if __name__ == "__main__":
    main()