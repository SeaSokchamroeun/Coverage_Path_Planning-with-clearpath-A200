# Coverage Path Planning — Clearpath Husky A200 (Nav2 / Jazzy)

**Branch: `smooth-coverage-working`**

A full-coverage (boustrophedon) path-planning pipeline for the Clearpath Husky
A200 on ROS 2 Jazzy + Nav2. The robot sweeps an entire room in a tidy
back-and-forth pattern, follows the planned path precisely, and returns home.

**Verified result:** 6.8 ± 1.0 cm lateral RMSE across 10 runs, 100% coverage,
zero backtracking.

---

## Pipeline at a glance

```
1  Map        SLAM Toolbox drive        → office_mapEmpty.pgm + .yaml
2  Boundary   extract_boundary.py       → room_boundary.yaml
3  Coverage   generate_room_coverage_bcd_v2.py → coverage_waypoints_bcd.yaml
4  Smooth     round_corners.py          → coverage_waypoints_smooth.yaml
5  Preview    plot_smooth.py            → coverage_preview_smooth.png
6  Run        run_coverage.py + record_run_a200.py → run_nav.log + run_track.csv
7  Analyse    analyze_coverage_run_a200.py + analyze_trajectory.py → metrics + png
```

Every stage is an independent script consuming the previous stage's file, so
the pipeline is easy to re-run and adapt to a new room or world.

---

## Prerequisites

```bash
# ROS 2 Jazzy + Clearpath packages installed and sourced:
source /opt/ros/jazzy/setup.bash && source ~/clearpath/setup.bash

# Python deps for the planning/analysis scripts:
pip install numpy scipy pyyaml pillow matplotlib

# Confirm the robot namespace (used throughout as a200_1103):
ros2 topic list | grep -m1 a200
```

All commands below run from `~/clearpath/coveragePathPlanning`. Replace
`a200_1103` with your robot's namespace and `office_mapEmpty` with your map name.

---

## Stage 1 — Build and save the map

Only needed once per world. If you already have a saved map, skip to Stage 2.

**Launch simulation, SLAM, and teleop in three terminals:**

```bash
# Terminal 1 — simulated world (spawn at 0,0,0)
ros2 launch clearpath_gz simulation.launch.py world:=office x:=0.0 y:=0.0 yaw:=0.0

# Terminal 2 — SLAM Toolbox in online mapping mode
ros2 launch clearpath_nav2_demos slam.launch.py \
  setup_path:=$HOME/clearpath/ use_sim_time:=true

# Terminal 3 — keyboard teleop; drive the whole room slowly
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/a200_1103/cmd_vel
```

Drive the full perimeter and criss-cross the interior until RViz shows a clean,
closed room. Then **save the map** (keep SLAM running):

```bash
ros2 run nav2_map_server map_saver_cli \
  -t /a200_1103/map \
  -f ~/clearpath/coveragePathPlanning/office_mapEmpty
```

This writes `office_mapEmpty.pgm` (image) and `office_mapEmpty.yaml` (metadata).
The `-f` path is **without** extension.

---

## Stage 2 — Extract the room boundary

Isolates one room from the map into a clean polygon, inset from the walls by the
robot's footprint so no planned path can hit a wall.

```bash
python3 extract_boundary.py office_mapEmpty.yaml \
  --seed -0.6 1.0 \
  --bbox -6.9 -3.8 5.7 5.9 \
  --robot-radius 0.71 --margin 0.15 \
  --out room_boundary.yaml \
  --debug-image boundary_debug.png
```

| Parameter | Meaning |
|---|---|
| `office_mapEmpty.yaml` | The map to read (positional). |
| `--seed X Y` | A point **inside** the target room; flood-fill starts here. For office_mapEmpty use its centre `-0.6 1.0`. **Change this per room.** |
| `--bbox XMIN YMIN XMAX YMAX` | Clip box just inside the outer walls, so the fill doesn't leak through doorways. |
| `--robot-radius` | Footprint circumscribing radius. **0.71 m** for the A200 (from its 0.55×0.45 m footprint half-extents) — NOT the bare chassis value, or the robot wedges on turns. |
| `--margin` | Extra safety inset (0.15 m). |
| `--out` | Output polygon → Stage 3. |
| `--debug-image` | Preview PNG — **always check it** shows a clean rectangle with no doorway leak. |

> **Note:** For a single clean rectangular room like `office_mapEmpty`, the
> boundary is optional — Stage 3 can plan directly on the map. The boundary
> matters when a map has multiple rooms or doorway gaps.

---

## Stage 3 — Generate the coverage path

Boustrophedon cellular decomposition → evenly spaced parallel lanes.

```bash
python3 generate_room_coverage_bcd_v2.py \
  --map office_mapEmpty.yaml \
  --boundary room_boundary.yaml \
  --out coverage_waypoints_bcd.yaml \
  --spawn 0 0 \
  --robot-radius 0.71 --safety-margin 0.15
```

(Omit `--boundary` to plan on the whole map — works for a single clean room.)

| Parameter | Meaning |
|---|---|
| `--map` | Occupancy grid to plan over. |
| `--boundary` | Room polygon from Stage 2 (optional). |
| `--out` | Output waypoints. |
| `--spawn X Y` | Robot start pose; path is ordered to begin near here. **Match your sim spawn.** |
| `--robot-radius` | Obstacle-inflation radius (0.71 m; match the boundary). |
| `--safety-margin` | Extra inflation (0.15 m). Total inflation = radius + margin = 0.86 m. |
| `--coverage-width` | Cleaned strip width (default 0.67 m = A200 chassis). |
| `--overlap` | Overlap between lanes (default 0.10 m). Lane spacing = coverage_width − overlap. |
| `--lane-axis` | `auto` / `x` / `y`. `auto` runs lanes along the longer axis to minimise turns. |

Expect: `Obstacle inflation: ... = 0.86 m`, `~99–100% coverage`, ~34 waypoints
(x-axis lanes). Auto-saves `coverage_preview_bcd.png` and `coverage_debug.png`.

**Optional — y-axis lanes** (more turns, useful for comparison):
```bash
python3 generate_room_coverage_bcd_v2.py --map office_mapEmpty.yaml \
  --out coverage_waypoints_bcd_y.yaml --spawn 0 0 \
  --robot-radius 0.71 --safety-margin 0.15 --lane-axis y
```

---

## Stage 4 — Smooth the corners (key step)

Replaces the sharp ~180° point-turns at each lane end with rounded arcs the
controller can drive through at speed. This is what eliminates stalls and
backtracking and roughly halves worst-case tracking error.

```bash
python3 round_corners.py \
  --in coverage_waypoints_bcd.yaml \
  --map office_mapEmpty.yaml \
  --out coverage_waypoints_smooth.yaml \
  --radius 0.4 --arc-step 0.1
```

| Parameter | Meaning |
|---|---|
| `--in` | Raw waypoints from Stage 3. |
| `--map` | Map, used to collision-check every arc (unsafe arcs are left sharp). |
| `--out` | Smoothed waypoints. |
| `--radius` | U-turn radius (0.4 m). Larger = gentler but needs more space; must be < wall clearance. |
| `--arc-step` | Spacing of arc points (0.1 m). Smaller = smoother. |
| `--min-angle` | Only round corners sharper than this (default 30°). |

Expect: `corners rounded: 31`, `skipped (unsafe): 0`, `34 -> 158 waypoints`.

> **If many corners are skipped as unsafe:** the lane ends sit too close to the
> walls. Regenerate Stage 3 with a larger `--safety-margin` (e.g. 0.2) so the
> arcs have room, or reduce `--radius`.

---

## Stage 5 — Preview the smoothed path

```bash
python3 plot_smooth.py \
  --waypoints coverage_waypoints_smooth.yaml \
  --map office_mapEmpty.yaml \
  --out coverage_preview_smooth.png
```

`plot_smooth.py` draws the path over the real map with equal axis scaling, so
the U-turn arcs render undistorted. Confirm a clean lawnmower pattern with
rounded ends, no wall clips, before driving.

---

## Stage 6 — Configure the stack and run

**Launch order** (each config file is read once at launch — relaunch to apply changes):

```bash
# 1. Simulation (spawn must match Stage 3 --spawn)
ros2 launch clearpath_gz simulation.launch.py world:=office x:=0.0 y:=0.0 yaw:=0.0

# 2. RViz
ros2 launch clearpath_viz view_navigation.launch.py namespace:=a200_1103

# 3. Localisation (AMCL)
ros2 launch ~/clearpath/localization_custom.launch.py \
  map:=$HOME/clearpath/coveragePathPlanning/office_mapEmpty.yaml use_sim_time:=true

# 4. Nav2
ros2 launch ~/clearpath/nav2_custom.launch.py use_sim_time:=true
```

**Verify config is live** before trusting a run:
```bash
ros2 topic info /a200_1103/map                                          # Publisher count: 1
ros2 lifecycle get /a200_1103/amcl                                      # active [3]
ros2 param get /a200_1103/global_costmap/global_costmap rolling_window  # False
```

**Set the initial pose** (identity orientation for yaw=0 spawn):
```bash
ros2 topic pub --once /a200_1103/initialpose \
  geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"

# Confirm a stable map->odom transform before driving:
ros2 run tf2_ros tf2_echo map odom --ros-args \
  -r /tf:=/a200_1103/tf -r /tf_static:=/a200_1103/tf_static
```

**Record + drive** (two terminals — start the recorder FIRST):
```bash
# Terminal A — recorder (note the namespaced /tf remap). Wait for "first sample".
python3 record_run_a200.py --namespace a200_1103 --out run_track_smooth.csv \
  --ros-args -r /tf:=/a200_1103/tf -r /tf_static:=/a200_1103/tf_static

# Terminal B — drive the SMOOTHED path
python3 run_coverage.py --waypoints coverage_waypoints_smooth.yaml \
  --stall-timeout 15 2>&1 | tee run_nav_smooth.log
```

| Parameter | Meaning |
|---|---|
| `run_coverage.py --waypoints` | Path to drive (use the smoothed file). |
| `run_coverage.py --stall-timeout` | Seconds without progress before a segment is re-planned (15). |
| `record_run_a200.py --out` | Trajectory CSV. |
| `record_run_a200.py --truth-topic` | Optional Gazebo ground-truth topic, to separate localisation from tracking error. |

---

## Stage 7 — Analyse the run

```bash
# Navigation-stack health: stalls, replans, lost coverage
python3 analyze_coverage_run_a200.py \
  --run-log run_nav_smooth.log \
  --waypoints coverage_waypoints_smooth.yaml

# Tracking quality: RMSE, cross-track error, backtracking, smoothness
python3 analyze_trajectory.py \
  --track run_track_smooth.csv \
  --waypoints coverage_waypoints_smooth.yaml \
  --map office_mapEmpty.yaml \
  --out run_track_smooth.png
```

**Metrics to check:**

| Metric | Target |
|---|---|
| Coverage | 100% waypoints reached, returned home |
| Lateral RMSE | < 10 cm |
| Cross-track 95th percentile | < 27 cm (half lane spacing) |
| Backtracking events | single digits (smoothed path → 0) |
| Distance driven / planned | < ~1.15× |

---

## Reusing in a new room or world

Only these change per environment:

| Value | Where |
|---|---|
| `world:=` , `x:= y:= yaw:=` | Stage 1 & 6 launch — the new world and spawn |
| map name | rebuild in Stage 1, or reuse |
| `--seed X Y` | Stage 2 — a point inside the new room |
| `--bbox` | Stage 2 — the new room's extent |
| `--spawn X Y` | Stage 3 — match the sim spawn |
| `--robot-radius` / `--margin` | only if the robot footprint changes (keep 0.71 / 0.15 for a standard A200) |

---

## Common pitfalls

- **Editing a config changes nothing until you relaunch that node** — ROS reads params once at startup.
- **Recorder started after the drive** → empty CSV. Always start it first and wait for "first sample".
- **Spawn mismatch** between Gazebo, `--spawn`, and the initial pose → AMCL starts in the wrong place and the run drifts.
- **Wrong world for the map** → the robot fights phantom or missing obstacles.
- **Rolling global costmap** in a large room → far goals rejected as "outside bounds". Keep it static (`rolling_window: false`).

---

## Key scripts

| Script | Role |
|---|---|
| `extract_boundary.py` | Flood-fills one room into a clean inset polygon. |
| `generate_room_coverage_bcd_v2.py` | Boustrophedon decomposition → lane waypoints. |
| `round_corners.py` | Rounds sharp lane-end turns into smooth arcs. |
| `plot_smooth.py` | Previews a waypoints file over the map (undistorted). |
| `run_coverage.py` | Drives the path via stall-aware followPath/goToPose. |
| `record_run_a200.py` | Records the true executed trajectory to CSV. |
| `analyze_coverage_run_a200.py` | Grades nav-stack health. |
| `analyze_trajectory.py` | Grades tracking accuracy and smoothness. |
