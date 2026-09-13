# Smoothed Full-Coverage Path Planning — Clearpath Husky A200 (Nav2, Jazzy)

**Branch status:** working milestone. Full-room boustrophedon coverage with
smooth rounded U-turns, executed through Nav2 with MPPI.

**Measured result (empty office room, single run):**

| Metric | Value | Target |
|---|---|---|
| Coverage | 100% (0 unreached waypoints) | — |
| Lateral RMSE | 6.0 cm | < 10 cm |
| Cross-track error (mean) | 4.4 cm | — |
| Cross-track error (95th pct) | 10.7 cm | — |
| Backtracking events | 0 | single digits |
| Commanded reversals | 3 | low |
| Driving forward | 94% of run time | — |

> Note: the trajectory analyser reports RMSE as a **lower bound** when lane
> spacing is small (cross-track error saturates against the neighbouring lane).
> For a rigorous number, re-record a run with `--truth-topic` (Gazebo ground
> truth) so true pose is compared against ground truth, not AMCL's own belief.

---

## Pipeline overview

```
map (.pgm/.yaml)
   │  extract_boundary.py
   ▼
room_boundary.yaml
   │  generate_room_coverage_bcd_v2.py
   ▼
coverage_waypoints_bcd.yaml          (sharp point-turns at lane ends)
   │  round_corners.py
   ▼
coverage_waypoints_smooth.yaml       (rounded U-turn arcs)
   │  run_coverage.py  (+ record_run_a200.py)
   ▼
run_nav.log + run_track.csv
   │  analyze_coverage_run_a200.py + analyze_trajectory.py
   ▼
metrics + run_track.png
```

---

## What made it work

This milestone is the result of fixing several independent issues. Recorded
here so the reasoning isn't lost:

1. **Footprint clearance.** The coverage generator was inflating obstacles by
   0.4 m (bare chassis) while Nav2's real footprint is 0.55 × 0.45 m
   (circumscribing radius 0.71 m). Lanes sat closer to walls than the robot
   could turn in. Fixed by generating with `--robot-radius 0.71 --safety-margin
   0.15`.

2. **Global costmap rolling window.** `global_costmap` had
   `rolling_window: true` with a 20 × 20 m window. In a room wider than the
   window, goals on the far side were rejected as "outside bounds". Fixed by
   setting `rolling_window: false` so the global costmap sizes to the whole map.

3. **AMCL config.** `laser_max_range` was left at the 100 m default (real
   Hokuyo range ~20 m); `initial_pose` pointed at an old room's spawn;
   `recovery_alpha_fast/slow` were left enabled from an A/B test. Reverted to
   0.0 / 0.0, `laser_max_range: 20.0`, initial pose at the true (0,0,0) spawn.

4. **followPath windowing.** `run_coverage.py` bundled multiple waypoints per
   `followPath` call, which meant a cold-started MPPI optimiser was handed the
   route's hardest maneuver (a ~180° lane-end reversal) on the first tick.
   Setting `FOLLOW_PATH_LOOKAHEAD = 1` drives one segment per call so hard
   segments fail fast to the `goToPose` global-planner fallback.

5. **MPPI rotation search.** `iteration_count: 1` gave the optimiser only one
   pass; raising to 3 let it solve more lane-end rotations. `wz_std: 0.4` widens
   rotational sampling.

6. **Smooth U-turns (the big precision win).** The remaining stalls and the
   RMSE ceiling both came from the sharp point-turns the boustrophedon plan
   demanded at every lane end. `round_corners.py` replaces each with a
   collision-checked circular arc, so the controller flows through turns at
   speed. This eliminated backtracking entirely and cut RMSE from ~9 cm to
   6 cm.

---

## Quick start (reproduce a run)

Assumes a saved map for your world and a sourced ROS 2 Jazzy environment.

```bash
# 1. Boundary (one point inside the room; bbox clips doorways)
python3 extract_boundary.py office_mapEmpty.yaml \
  --seed 0 0 --robot-radius 0.71 --margin 0.15 \
  --out room_boundary.yaml --debug-image boundary_debug.png

# 2. Coverage path
python3 generate_room_coverage_bcd_v2.py \
  --map office_mapEmpty.yaml --boundary room_boundary.yaml \
  --out coverage_waypoints_bcd.yaml --spawn 0 0 \
  --robot-radius 0.71 --safety-margin 0.15

# 3. Smooth the corners
python3 round_corners.py \
  --in coverage_waypoints_bcd.yaml --map office_mapEmpty.yaml \
  --out coverage_waypoints_smooth.yaml --radius 0.4 --arc-step 0.1

# 4. Bring up sim + localization + Nav2 (your custom launch files), set the
#    initial pose, confirm the map->odom transform is stable.

# 5. Record (terminal A, wait for "first sample") then drive (terminal B)
python3 record_run_a200.py --namespace a200_1103 --out run_track.csv \
  --ros-args -r /tf:=/a200_1103/tf -r /tf_static:=/a200_1103/tf_static

python3 run_coverage.py --waypoints coverage_waypoints_smooth.yaml \
  --stall-timeout 15 2>&1 | tee run_nav.log

# 6. Analyse
python3 analyze_coverage_run_a200.py --run-log run_nav.log \
  --waypoints coverage_waypoints_smooth.yaml
python3 analyze_trajectory.py --track run_track.csv \
  --waypoints coverage_waypoints_smooth.yaml \
  --map office_mapEmpty.yaml --out run_track.png
```

For the full per-parameter explanation and how to adapt this to any room or
world, see the pipeline guide PDF.

---

## Key files

| File | Role |
|---|---|
| `extract_boundary.py` | Isolate one room from a map into an inset polygon |
| `generate_room_coverage_bcd_v2.py` | BCD coverage-path generation |
| `round_corners.py` | Post-process sharp corners into smooth U-turn arcs |
| `run_coverage.py` | Drive the path through Nav2 (segment-at-a-time, goToPose fallback) |
| `record_run_a200.py` | Record the executed trajectory to CSV |
| `analyze_coverage_run_a200.py` | Grade nav-stack health (stalls, replans, lost coverage) |
| `analyze_trajectory.py` | Grade tracking quality (RMSE, cross-track, backtracking) |
| `nav2_custom.yaml` | Nav2 params (MPPI-tuned, static global costmap) |
| `localization_custom.yaml` | AMCL params |

---

## Known remaining item

The long cross-room straight segments still occasionally stall on their first
`followPath` attempt and recover via `goToPose` (no coverage lost, adds ~15 s
each). The lane-end **turns** are fully solved by the smoothing; the straights
are the last thing left if further speed-up is wanted. Candidate levers:
further MPPI tuning, or higher `FOLLOW_PATH_LOOKAHEAD` now that the reversals
are gone.

## Next planned work (separate branch)

Integrate the Spatio-Temporal Voxel Layer (STVL) for 3D obstacle sensing from
the VLP-16, ported from the ROS 2 Humble configuration.
