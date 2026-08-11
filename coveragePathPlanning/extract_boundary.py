#!/usr/bin/env python3
"""
extract_boundary.py

Automatically derive a room's floor-space boundary polygon from a ROS 2
map_server-style occupancy grid (.pgm + .yaml), instead of hand-measuring
wall coordinates into a hardcoded ROOM_CORNERS list.

Approach:
  1. Load the map yaml (image path, resolution, origin, negate,
     occupied_thresh, free_thresh) and the pgm image.
  2. Convert pixel values -> occupancy probability using the same
     convention as ROS map_server / SLAM Toolbox map_saver.
  3. Flood-fill the free-space mask starting from a SEED point you give
     in world (map-frame) coordinates -- a point you know is inside the
     target room (e.g. the Husky's start pose, or any point confirmed
     inside the room in RViz). This isolates just that room's free
     space, so the boundary doesn't leak through open doorways into
     hallways or other rooms.
  4. Erode the isolated region inward by (robot_radius + safety_margin)
     so the resulting polygon is already a safe drivable boundary, same
     as the manual "shrunk ~0.5m inward" comment in the existing script.
  5. Find the outer contour of the eroded region, simplify it to a
     manageable number of vertices, and convert back to world (map
     frame) coordinates.
  6. Write room_boundary.yaml: {frame_id, corners: [[x, y], ...]}

generate_coverage_path.py should then load ROOM_CORNERS from this file
instead of hardcoding it -- see the snippet in the docstring at the
bottom of this file.

Usage:
    python3 extract_boundary.py office_map.yaml --seed 7.5 -5.95 \
        --robot-radius 0.4 --margin 0.1 --out room_boundary.yaml

    --seed X Y     : world-frame point known to be inside the target room
    --robot-radius : Husky footprint radius in meters (default 0.4, per
                     the original task guide's stated approx. radius)
    --margin       : additional safety margin in meters (default 0.1)
    --out          : output yaml path (default room_boundary.yaml)
    --debug-image  : optional path to save a visualization png showing
                     the free mask, flood-filled region, eroded region,
                     and final polygon overlaid -- useful to sanity
                     check before trusting the output.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml


def load_map(yaml_path):
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    map_dir = os.path.dirname(os.path.abspath(yaml_path))
    image_path = meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(map_dir, image_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load map image at {image_path}")

    resolution = float(meta["resolution"])
    origin = meta["origin"]  # [x, y, yaw]
    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.25))

    return img, resolution, origin, negate, occupied_thresh, free_thresh


def occupancy_free_mask(img, negate, occupied_thresh, free_thresh):
    v = img.astype(np.float32) / 255.0
    occ_prob = v if negate else (1.0 - v)
    free_mask = np.where(occ_prob < free_thresh, 255, 0).astype(np.uint8)
    return free_mask


def world_to_pixel(x, y, resolution, origin, img_height):
    ox, oy = origin[0], origin[1]
    px = int(round((x - ox) / resolution))
    py = int(round(img_height - 1 - (y - oy) / resolution))
    return px, py


def pixel_to_world(px, py, resolution, origin, img_height):
    ox, oy = origin[0], origin[1]
    x = ox + px * resolution
    y = oy + (img_height - 1 - py) * resolution
    return x, y


def clip_to_bbox(free_mask, resolution, origin, img_height, bbox):
    h, w = free_mask.shape
    xmin, ymin, xmax, ymax = bbox
    p0 = world_to_pixel(xmin, ymin, resolution, origin, img_height)
    p1 = world_to_pixel(xmax, ymax, resolution, origin, img_height)
    px_min, px_max = sorted([p0[0], p1[0]])
    py_min, py_max = sorted([p0[1], p1[1]])
    px_min = max(0, px_min)
    py_min = max(0, py_min)
    px_max = min(w - 1, px_max)
    py_max = min(h - 1, py_max)

    clipped = np.zeros_like(free_mask)
    clipped[py_min:py_max + 1, px_min:px_max + 1] = free_mask[py_min:py_max + 1, px_min:px_max + 1]
    return clipped


def flood_fill_room(free_mask, seed_px):
    h, w = free_mask.shape
    if not (0 <= seed_px[0] < w and 0 <= seed_px[1] < h):
        raise ValueError(f"Seed pixel {seed_px} is outside the map image bounds ({w}x{h})")
    if free_mask[seed_px[1], seed_px[0]] == 0:
        raise ValueError(
            "Seed point lands on an occupied/unknown cell in the map. "
            "Pick a world coordinate that's clearly inside the free floor area."
        )

    filled = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask_copy = free_mask.copy()
    cv2.floodFill(
        mask_copy, filled, seedPoint=seed_px, newVal=128,
        loDiff=0, upDiff=0, flags=4
    )
    room_mask = np.where(mask_copy == 128, 255, 0).astype(np.uint8)
    return room_mask


def erode_room(room_mask, resolution, robot_radius, margin):
    shrink_m = robot_radius + margin
    shrink_px = max(1, int(round(shrink_m / resolution)))
    kernel_size = 2 * shrink_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(room_mask, kernel, iterations=1)
    return eroded


def extract_polygon(eroded_mask, epsilon_ratio=0.005):
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "No contour found after erosion -- the room may be too small "
            "for the given robot_radius + margin, or the seed/flood-fill "
            "produced an empty region."
        )
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    return simplified.reshape(-1, 2)


def save_debug_image(path, free_mask, room_mask, eroded_mask, polygon_px):
    vis = cv2.cvtColor(free_mask, cv2.COLOR_GRAY2BGR)
    vis[room_mask == 255] = (200, 150, 0)
    vis[eroded_mask == 255] = (0, 200, 0)
    pts = polygon_px.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(path, vis)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_yaml", help="Path to the map yaml (e.g. office_map.yaml)")
    ap.add_argument("--seed", nargs=2, type=float, required=True, metavar=("X", "Y"),
                     help="World-frame point known to be inside the target room")
    ap.add_argument("--robot-radius", type=float, default=0.4,
                     help="Husky footprint radius in meters (default 0.4)")
    ap.add_argument("--margin", type=float, default=0.1,
                     help="Additional safety margin in meters (default 0.1)")
    ap.add_argument("--out", default="room_boundary.yaml", help="Output yaml path")
    ap.add_argument("--debug-image", default=None, help="Optional path to save a visualization png")
    ap.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                     help="World-frame bounding box to clip to BEFORE flood-fill. Use this when internal "
                          "walls have doorway gaps and a plain flood-fill would leak into adjacent rooms.")
    args = ap.parse_args()

    img, resolution, origin, negate, occ_thresh, free_thresh = load_map(args.map_yaml)
    h = img.shape[0]

    free_mask = occupancy_free_mask(img, negate, occ_thresh, free_thresh)
    if args.bbox:
        free_mask = clip_to_bbox(free_mask, resolution, origin, h, args.bbox)
    seed_px = world_to_pixel(args.seed[0], args.seed[1], resolution, origin, h)
    room_mask = flood_fill_room(free_mask, seed_px)
    eroded = erode_room(room_mask, resolution, args.robot_radius, args.margin)
    polygon_px = extract_polygon(eroded)

    corners = [list(pixel_to_world(int(px), int(py), resolution, origin, h)) for px, py in polygon_px]
    if corners[0] != corners[-1]:
        corners.append(corners[0])

    with open(args.out, "w") as f:
        yaml.dump({"frame_id": "map", "corners": corners}, f)

    print(f"Extracted {len(corners)} corners -> {args.out}")
    for c in corners:
        print(f"  ({c[0]:.2f}, {c[1]:.2f})")

    if args.debug_image:
        save_debug_image(args.debug_image, free_mask, room_mask, eroded, polygon_px)
        print(f"Debug visualization saved -> {args.debug_image}")


if __name__ == "__main__":
    main()