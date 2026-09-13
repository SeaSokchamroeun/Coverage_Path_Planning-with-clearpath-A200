#!/usr/bin/env python3
"""
extract_boundary.py

Single-room boundary extraction from a SLAM occupancy grid (.pgm/.yaml).

Pipeline:
  1. Load the occupancy grid (ROS map_server format: .yaml pointing at a .pgm).
  2. Threshold into a binary free-space mask.
  3. Clip the free-space mask to a user-supplied bounding box (world coords) so
     flood-fill physically cannot leak through a doorway into a corridor/other room.
  4. Flood-fill from a seed point (world coords, must be inside the target room)
     to isolate that room's connected free-space region.
  5. Erode the region inward by (robot_radius + margin) so the boundary is a
     safe drivable envelope, not the raw wall line.
  6. cv2.findContours + polygon simplification (cv2.approxPolyDP) to get the
     corner list.
  7. Write corners (world coords, closed ring) to a YAML file, and optionally
     save a debug image (free space / filled region / eroded region / final
     polygon) matching what the rest of the pipeline (plot_coverage.py etc.)
     expects to consume.

Usage:
  python3 extract_boundary.py office_map.yaml --seed 7.5 5.95 \
      --bbox 0.6 0.5 14.4 11.4 --robot-radius 0.4 --margin 0.1 \
      --out room_boundary_topright.yaml --debug-image boundary_debug_topright.png
"""

import argparse
import os
import sys

import numpy as np
import yaml
import cv2
from PIL import Image


def load_map(map_yaml_path):
    """Load a ROS map_server-style map: .yaml metadata + .pgm image.

    Returns:
        img: 2D uint8 numpy array, row 0 = TOP of image as stored on disk
        resolution: meters/pixel
        origin: (x, y, yaw) world coords of the BOTTOM-LEFT pixel (ROS convention)
        negate, occupied_thresh, free_thresh: thresholding params from the yaml
    """
    with open(map_yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    map_dir = os.path.dirname(os.path.abspath(map_yaml_path))
    image_path = meta["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(map_dir, image_path)

    pil_img = Image.open(image_path).convert("L")
    img = np.array(pil_img)

    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    return img, resolution, origin, negate, occupied_thresh, free_thresh


def pixel_to_world(px, py, img_height, resolution, origin):
    """ROS map convention: origin is the world coord of the pixel at the
    BOTTOM-LEFT of the image, image row 0 is the TOP."""
    wx = origin[0] + (px + 0.5) * resolution
    wy = origin[1] + (img_height - py - 0.5) * resolution
    return wx, wy


def world_to_pixel(wx, wy, img_height, resolution, origin):
    px = (wx - origin[0]) / resolution
    py = img_height - (wy - origin[1]) / resolution
    return int(round(px)), int(round(py))


def build_free_space_mask(img, negate, occupied_thresh, free_thresh):
    """Return a binary mask (255 = free/drivable, 0 = occupied or unknown)."""
    norm = img.astype(np.float32) / 255.0
    if negate:
        occ_prob = norm
    else:
        occ_prob = 1.0 - norm

    free_mask = np.zeros(img.shape, dtype=np.uint8)
    free_mask[occ_prob <= free_thresh] = 255
    # Anything at/above occupied_thresh, or in the unknown band between the
    # two thresholds, is treated as NOT free (conservative — matches Nav2's
    # own costmap interpretation of an unmarked map).
    return free_mask


def clip_to_bbox(mask, bbox_world, img_height, resolution, origin):
    """Zero out everything outside the world-space bounding box."""
    xmin, ymin, xmax, ymax = bbox_world
    clipped = np.zeros_like(mask)

    px0, py0 = world_to_pixel(xmin, ymax, img_height, resolution, origin)  # top-left in pixel space
    px1, py1 = world_to_pixel(xmax, ymin, img_height, resolution, origin)  # bottom-right in pixel space

    px0, px1 = sorted((max(px0, 0), min(px1, mask.shape[1])))
    py0, py1 = sorted((max(py0, 0), min(py1, mask.shape[0])))

    clipped[py0:py1, px0:px1] = mask[py0:py1, px0:px1]
    return clipped


def flood_fill_room(mask, seed_px):
    """Flood-fill the connected free-space component containing seed_px.
    mask is already bbox-clipped, so the fill physically cannot escape
    the box (e.g. through a doorway)."""
    sx, sy = seed_px
    h, w = mask.shape
    if not (0 <= sx < w and 0 <= sy < h) or mask[sy, sx] == 0:
        raise ValueError(
            f"Seed point falls outside the free-space/bbox-clipped mask at pixel ({sx},{sy}). "
            "Check --seed is actually inside the target room and inside --bbox."
        )

    filled = np.zeros((h + 2, w + 2), dtype=np.uint8)
    work = mask.copy()
    cv2.floodFill(work, filled, (sx, sy), 128)

    room_mask = np.zeros_like(mask)
    room_mask[work == 128] = 255
    return room_mask


def erode_room(room_mask, resolution, robot_radius, margin):
    """Erode the room's free-space mask inward by (robot_radius + margin),
    converted from meters to pixels."""
    shrink_m = robot_radius + margin
    shrink_px = max(1, int(round(shrink_m / resolution)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink_px + 1, 2 * shrink_px + 1))
    eroded = cv2.erode(room_mask, kernel)
    return eroded, shrink_px


def extract_corners(eroded_mask, epsilon_frac=0.01):
    """cv2.findContours on the eroded mask -> largest contour -> polygon
    simplification -> ordered corner list (pixel coords, closed ring)."""
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in the eroded room mask — room may have fully closed up "
                            "(robot_radius + margin too large for this room's dimensions).")

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    epsilon = epsilon_frac * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)

    corners_px = [tuple(pt[0]) for pt in approx]
    corners_px.append(corners_px[0])  # close the ring
    return corners_px


def save_debug_image(free_mask, room_mask, eroded_mask, corners_px, out_path):
    """Grayscale free space, tinted filled room, green eroded region, red
    outlined final polygon — matches the format plot_coverage.py expects."""
    h, w = free_mask.shape
    debug = cv2.cvtColor(free_mask, cv2.COLOR_GRAY2BGR)

    tint = debug.copy()
    tint[room_mask > 0] = (60, 60, 200)  # room region tinted (BGR)
    debug = cv2.addWeighted(debug, 0.6, tint, 0.4, 0)

    debug[eroded_mask > 0] = (0, 200, 0)  # eroded safe region in green

    pts = np.array(corners_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imwrite(out_path, debug)


def main():
    ap = argparse.ArgumentParser(description="Single-room boundary extraction via flood-fill + bbox clip.")
    ap.add_argument("map_yaml", help="Path to the ROS map_server .yaml (pairs with a .pgm).")
    ap.add_argument("--seed", type=float, nargs=2, required=True, metavar=("X", "Y"),
                     help="World-coordinate seed point inside the target room.")
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                     help="World-coordinate bounding box the flood-fill is clipped to.")
    ap.add_argument("--robot-radius", type=float, default=0.4, help="Robot radius in meters (default 0.4).")
    ap.add_argument("--margin", type=float, default=0.1, help="Extra safety margin in meters (default 0.1).")
    ap.add_argument("--out", default="room_boundary.yaml", help="Output YAML path for the corner list.")
    ap.add_argument("--debug-image", default=None, help="Optional path to save a debug PNG.")
    ap.add_argument("--epsilon-frac", type=float, default=0.01,
                     help="Polygon simplification strength as a fraction of contour perimeter (default 0.01).")
    args = ap.parse_args()

    img, resolution, origin, negate, occ_th, free_th = load_map(args.map_yaml)
    h, w = img.shape

    free_mask = build_free_space_mask(img, negate, occ_th, free_th)
    clipped_mask = clip_to_bbox(free_mask, args.bbox, h, resolution, origin)

    seed_px = world_to_pixel(args.seed[0], args.seed[1], h, resolution, origin)
    room_mask = flood_fill_room(clipped_mask, seed_px)

    eroded_mask, shrink_px = erode_room(room_mask, resolution, args.robot_radius, args.margin)

    corners_px = extract_corners(eroded_mask, epsilon_frac=args.epsilon_frac)
    corners_world = [pixel_to_world(px, py, h, resolution, origin) for (px, py) in corners_px]

    with open(args.out, "w") as f:
        yaml.safe_dump({
            "frame_id": "map",
            "robot_radius": args.robot_radius,
            "margin": args.margin,
            "corners": [[round(float(x), 2), round(float(y), 2)] for (x, y) in corners_world],
        }, f, default_flow_style=False, sort_keys=False)

    print(f"Extracted {len(corners_world) - 1} corners -> {args.out}")
    for x, y in corners_world:
        print(f"  ({x:.2f}, {y:.2f})")

    if args.debug_image:
        save_debug_image(free_mask, room_mask, eroded_mask, corners_px, args.debug_image)
        print(f"Debug image -> {args.debug_image}")


if __name__ == "__main__":
    main()
