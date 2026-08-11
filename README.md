# Husky A200 — Automated Boundary & Coverage Path Planning

Autonomous indoor inspection pipeline for a Clearpath Husky A200: build a 2D occupancy map via SLAM,
extract a room's drivable boundary from that map, generate a full-coverage floor path (Boustrophedon/BCD
or Backtracking Spiral/BSA), drive it through Nav2, and verify the result against ground truth — not against
how the plot looks.

Built against a six-phase task guide (Hardware Bringup → SLAM → Boundary Extraction → Coverage Path
Planning → Nav2 Execution → Visualization/Validation). See **Status** below for where each phase stands.

## Status snapshot

| Phase | Deliverable | Status |
|---|---|---|
| 1 — Hardware Bringup | Stable 2D LaserScan from the 3D Velodyne | PARTIAL |
| 2 — SLAM & Mapping | Saved occupancy grid + localization | PARTIAL |
| 3 — Route Server & Boundary | Node: `/map` → boundary waypoints | PARTIAL |
| 4 — Coverage Path Planning | Waypoint array, full floor coverage | PARTIAL |
| 5 — Nav2 Execution & Tracking | Autonomous execution + `/inspection_path` | PARTIAL |
| 6 — Visualization & Validation | RViz Path display + MCAP logging | NOT MET |

The planning side (boundary extraction, coverage generation, verification) is the most mature part of the
project — validated on two rooms with measured, ground-truth numbers. The execution side works reliably
in simulation. The main structural gap across every phase is that the pipeline is still a set of offline Python
scripts + a Python Action Client, not yet the live ROS 2 nodes (Route Server Action Server, path-tracking
node, MCAP logging) the task guide specifies. See **Known issues & open work**.

## Repository layout

```
clearpath/
├── robot.yaml, robot.urdf.xacro, robot.srdf(.xacro)   # Husky A200 platform description
├── setup.bash                                         # source before any ros2 launch/run
├── office_map.pgm / office_map.yaml                   # SLAM-built occupancy grid of the office
├── localization_custom.launch.py / .yaml               # AMCL localization (primary)
├── slam_toolbox_localization_launch.py / .yaml         # alt: SLAM Toolbox localization mode
├── nav2_custom.launch.py / .yaml                        # Nav2 bringup
├── log_localization_covariance.py                       # capture AMCL covariance over time
├── summarize_covariance_log.py                           # summarize a covariance log
├── world/                                                # Gazebo world file(s)
├── platform/, sensors/, manipulators/, route/            # ClearPath-generated platform config
└── coveragePathPlanning/                                 # boundary extraction + CPP + eval (below)
```

## `coveragePathPlanning/` — pipeline scripts

| Script | Role |
|---|---|
| `extract_boundary.py` | Flood-fills free space from a seed point inside the room, erodes it by robot radius + margin, and writes `room_boundary.yaml`. Won't leak through doorways (bounding-box clipped). |
| `bcd_route_server.py` | Tier-2 core: whole-map boustrophedon cellular decomposition (BCD), plus the shared low-level primitives (A*, `connect_points`, `is_safe`, `segment_clear`) every other script imports. Can also run standalone as a **multi-room** generator — edit the constants at the top of the file (`PGM_PATH`, `SPAWN_WORLD`, etc.), no CLI flags. |
| `generate_room_coverage_bcd.py` | **Primary generator.** Boustrophedon (lawnmower) coverage for a *single* room: two-layer lane sweep, multi-interval column handling, clearance repair, coverage verification. Writes `coverage_waypoints.yaml`. |
| `generate_room_coverage_bsa.py` | Backtracking Spiral Algorithm — comparison candidate. Same map/boundary/robot params/clearance repair as the BCD generator (only the guidance-track step differs). Never touches `coverage_waypoints.yaml`; writes `coverage_waypoints_bsa.yaml`. |
| `coverage_common.py` | Algorithm-agnostic shared layer: boundary I/O, path simplify/smooth/clearance-repair, coverage verification, debug + preview image writers. Both generators import from here so a fix in one never breaks the other. |
| `manhattan_utils.py` | Strict axis-aligned (4-connected, 90°-only) A* and path-simplify variants, for stitches that must not cut diagonally. |
| `plot_coverage.py` | Static sanity-check plot: boundary + waypoints → PNG, before trusting anything in sim. |
| `plot_coverage_preview.py` | Regenerate a labeled preview PNG from an *existing* waypoints file, without rerunning the generator. |
| `check_defect_corner.py` | Focused clearance check at a known problem coordinate; compares candidates side by side to tell a real map-scale pinch apart from an algorithm artifact. |
| `run_coverage.py` | Nav2 execution driver. Drives the route in short segments (`followPath`, falling back to `goToPose` on a stall), returns home at the end, logs every skipped waypoint. |
| `analyze_coverage_run.py` | Parses a `run_coverage.py` log and summarizes continuous-path attempts, stalls, replanning hops, skipped waypoints, and final outcome. |
| `evaluate_coverage_path.py` | Standardized metrics for any `coverage_waypoints.yaml`: coverage %, path length, waypoint/turn count, sharp-turn %, overlap %, estimated time. Supports `--compare` across multiple candidates. |

## Quick start — full pipeline for one room

```bash
cd ~/clearpath/coveragePathPlanning

# 1. Boundary
python3 extract_boundary.py office_map.yaml \
  --seed 7.5 5.95 --bbox 0.6 0.5 14.4 11.4 \
  --robot-radius 0.4 --margin 0.1 \
  --out room_boundary.yaml --debug-image boundary_debug.png

# 2. Coverage path (Boustrophedon/BCD baseline)
python3 generate_room_coverage_bcd.py \
  --map office_map.yaml --boundary room_boundary.yaml \
  --out coverage_waypoints.yaml --spawn 1.303262 1.481998

#    ...or the BSA candidate, side by side:
python3 generate_room_coverage_bsa.py \
  --map office_map.yaml --boundary room_boundary.yaml \
  --spawn 1.303262 1.481998 --out coverage_waypoints_bsa.yaml

# 3. Preview before trusting it in sim
python3 plot_coverage.py --boundary room_boundary.yaml \
  --waypoints coverage_waypoints.yaml --out coverage_preview.png
```

Then bring up the stack (each in its own terminal, `source ~/clearpath/setup.bash` first):

```bash
# Gazebo
ros2 launch clearpath_gz simulation.launch.py world:=office \
  x:=1.303262 y:=1.481998 yaw:=1.5707963267948966

# Localization
ros2 launch ~/clearpath/localization_custom.launch.py \
  map:=$HOME/clearpath/coveragePathPlanning/office_map.yaml use_sim_time:=true

# RViz
ros2 launch clearpath_viz view_navigation.launch.py namespace:=a200_1103

# Nav2
ros2 launch ~/clearpath/nav2_custom.launch.py use_sim_time:=true
```

Verify `controller_server` and `planner_server` both report `active [3]` (run the lifecycle checks one at a
time or as a loop — pasting several on one line interleaves their output and looks like a false failure), then
run a single-goal sanity check before committing to the full sweep. Once Nav2 is confirmed healthy:

```bash
cd ~/clearpath/coveragePathPlanning
python3 run_coverage.py 2>&1 | tee run_coverage_log.txt
python3 analyze_coverage_run.py --run-log run_coverage_log.txt
python3 evaluate_coverage_path.py --map office_map.yaml --boundary room_boundary.yaml \
  --waypoints coverage_waypoints.yaml
```

The full annotated version of this sequence (with expected output at each step and troubleshooting notes)
lives in `run_sequence.txt`.

## File formats

**`room_boundary.yaml`** — polygon corners in map frame, output by `extract_boundary.py`:
```yaml
corners:
- [1.454, 10.538]
- [1.454, 1.338]
- [13.554, 1.338]
- [13.554, 10.538]
frame_id: map
```

**`coverage_waypoints.yaml`** — ordered drive-through poses, the common schema every generator writes
and every downstream script (`run_coverage.py`, `evaluate_coverage_path.py`, `plot_coverage.py`) reads:
```yaml
frame_id: map
waypoints:
- {x: 1.304, y: 1.488, yaw: 1.5487}
- {x: 1.454, y: 8.288, yaw: -1.5708}
```

**`office_map.yaml`** — standard ROS 2 `map_server`/SLAM Toolbox map metadata (`image`, `resolution`,
`origin`, `negate`, `occupied_thresh`, `free_thresh`) alongside the paired `office_map.pgm` occupancy grid.

## BCD vs. BSA — algorithm comparison

Both generators were run through an identical, fairness-controlled pipeline (same map, boundary, spawn
pose, robot width/margin, and execution driver) on Room 2:

| Metric | Boustrophedon (BCD) | BSA |
|---|---|---|
| Coverage | 99.8% | 99.8% |
| Waypoints | 69 | 34 |
| Turns | 67 | 29 |
| Real execution runtime | ≈28.6 min | ≈17.1 min |
| Real failure rate | 2/69 (2.9%) | 1/34 (2.9%) |

BCD has one hard failure point: a genuine map-scale pinch at world ≈(10.45, 8.5) where clearance bottoms
out at 0.000 m — confirmed independent of any generator parameter via a 4-configuration stress test. BSA's
coarse-cell placement lets it stand off that same pinch, but introduces its own single weak point instead: a
98.7° turn at waypoint 14 where `goToPose` took 141 s to give up. **BSA is the current recommendation** for
Room 2 — it matches BCD on coverage and reliability while being ~40% faster and structurally avoiding the
BCD defect corner — with that one turn-geometry issue as the remaining open item before finalizing it.

## Known issues & open work

- **BSA waypoint 14** — 98.7° turn, 141 s stuck `goToPose`. Worth a targeted fix (wider corner-cut or a
  brief reorient-in-place) before finalizing BSA as the Room 2 planner.
- **BCD defect corner** — real map pinch at ≈(10.45, 8.5), 0.000 m clearance. Decision needed: accept,
  special-case the one waypoint, or physically relocate the obstacle.
- **Route Server is not a live node.** Boundary extraction and coverage generation are offline scripts, not a
  ROS 2 Action Server subscribing to `/map` as the task guide specifies (Phase 3).
- **No `/tf` → `/inspection_path` tracking node.** Only the pre-planned route is published once for RViz —
  the actual driven trajectory isn't tracked or published (Phase 5.3), which also blocks the RViz Path
  display and MCAP bag logging (Phase 6).
- **3D point cloud isn't a second local-costmap layer yet** — obstacle checks currently rely on the 2D scan
  slice only.
- **Not yet scaled past two rooms.** Multi-room decomposition exists and was validated separately
  (19 clean cells across all 4 rooms, 0 stitching warnings, via `bcd_route_server.py`), but isn't wired into
  the per-room coverage generator yet.

## Requirements

- ROS 2 Humble Hawksbill or Jazzy Jalisco, Nav2, `clearpath_gz`, `clearpath_viz`
- Python 3, with: `numpy`, `opencv-python`, `pyyaml`, `matplotlib`, `scipy`, `Pillow`,
  `nav2_simple_commander`
