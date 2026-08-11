#!/usr/bin/env python3
"""
plot_coverage_preview.py

Regenerate a coverage_preview_<label>.png from an EXISTING waypoints
yaml, without re-running the generator that produced it. Works for any
algorithm -- boustrophedon, BSA, or whatever comes next (STC, spiral,
...) -- since it only needs the waypoints file, the boundary file, and
a label, all via coverage_common.save_preview_plot.

Useful when: you already generated coverage_waypoints_bsa.yaml earlier
and just want the picture again, or you're comparing an old vs new run
of the same algorithm without regenerating the route.

Usage:
    python3 plot_coverage_preview.py \\
        --waypoints coverage_waypoints_bsa.yaml \\
        --boundary room_boundary.yaml \\
        --label BSA \\
        --out coverage_preview_bsa.png
"""

import argparse
import yaml

from coverage_common import load_boundary, save_preview_plot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoints", required=True)
    ap.add_argument("--boundary", default="room_boundary.yaml")
    ap.add_argument("--label", required=True,
                    help="Shown in the plot title, e.g. 'BCD', 'BSA', 'STC'")
    ap.add_argument("--out", default=None,
                    help="Defaults to coverage_preview_<label lowercased>.png")
    args = ap.parse_args()

    with open(args.waypoints) as f:
        waypoints = yaml.safe_load(f)["waypoints"]
    corners, _ = load_boundary(args.boundary)

    out_path = args.out or f"coverage_preview_{args.label.lower()}.png"
    save_preview_plot(waypoints, corners, out_path, args.label)
    print(f"Saved {out_path} ({len(waypoints)} waypoints, label='{args.label}')")


if __name__ == "__main__":
    main()
