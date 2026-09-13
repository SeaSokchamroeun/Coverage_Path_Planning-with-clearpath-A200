#!/usr/bin/env python3
"""
record_run_a200.py

Records where the A200 ACTUALLY went during a coverage run, so path-following
can be measured instead of described. Writes the CSV that
analyze_trajectory.py consumes.

Samples at 20 Hz:
    map -> <base frame>    where AMCL BELIEVES the robot is
    ground-truth odometry  where it ACTUALLY is (optional)
    cmd_vel                what the controller commanded

WHAT CHANGED FROM THE TURTLEBOT VERSION
    1. NAMESPACE. Everything on the A200 lives under /a200_XXXX/. The
       TurtleBot version hardcodes bare /cmd_vel and the frame
       'base_footprint', so on a Husky it subscribes to a topic nobody
       publishes and looks up a frame that does not exist -- it would
       record a file of nothing. Pass --namespace; topics and frames are
       derived from it.

    2. FRAME AUTODETECTION. Clearpath prefixes TF frames with the
       namespace (a200_1103/base_link), but this varies with setup and
       ROS version. Rather than guess, this probes the TF tree at startup
       for the first frame that resolves against 'map' and reports which
       one it picked. Override with --base-frame.

    3. GROUND TRUTH IS OPTIONAL AND OFF BY DEFAULT. The TurtleBot version
       aborts without it. The clearpath_gz office world has no
       ground-truth pose plugin, so demanding it would block every run.
       Pass --truth-topic to enable it; see the note at the bottom of this
       docstring for how to publish one.

    4. CMD_VEL TOPIC. Clearpath routes commands through
       /<ns>/platform/cmd_vel after the twist mux, while Nav2 publishes to
       /<ns>/cmd_vel. Sampling the wrong one hides anything the mux or
       safety stop did. Default is the post-mux topic; --cmd-topic to
       change. The message type (Twist vs TwistStamped) is detected at
       runtime.

Usage
    python3 record_run_a200.py --namespace a200_1103 --out run_track.csv

    # start this FIRST, then run_coverage.py in another terminal
    # Ctrl-C here when the route finishes

CSV columns
    t, x, y, yaw, cmd_v, cmd_w, true_x, true_y, true_yaw
    (the true_* columns are empty when ground truth is not enabled)

ENABLING GROUND TRUTH (optional)
    Gazebo publishes model poses on /world/office/dynamic_pose/info.
    Bridge it, then point --truth-topic at the result:
        ros2 run ros_gz_bridge parameter_bridge \\
          /world/office/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V
    The bridge does NOT survive a Gazebo restart. Without ground truth you
    can still measure tracking against the plan; you just cannot separate
    localization error from controller error.
"""

import argparse
import csv
import math
import statistics
import sys

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RunRecorder(Node):
    def __init__(self, args):
        super().__init__('run_recorder')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ns = args.namespace.strip('/')
        self.out_path = args.out
        self.map_frame = args.map_frame
        self.base_frame = args.base_frame        # may be None -> autodetect
        self.ns = ns

        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)

        self.rows = []
        self.v = 0.0
        self.w = 0.0
        self.miss = 0
        self.t0 = None
        self.ticks = 0
        self.rate_hz = args.rate
        self.heartbeat_ticks = int(args.heartbeat * args.rate)
        self.frame_deadline = int(args.frame_timeout * args.rate)

        cmd_topic = args.cmd_topic or f"/{ns}/platform/cmd_vel"
        self._subscribe_cmd(cmd_topic)

        self.truth_topic = args.truth_topic
        self.truth = None
        self.truth_n = 0
        self.truth_last_n = 0
        if self.truth_topic:
            qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Odometry, self.truth_topic,
                                     self.on_truth, qos)
            print(f"  ground truth : {self.truth_topic} (waiting)")
        else:
            print("  ground truth : disabled (no --truth-topic). "
                  "Localization error cannot be separated from tracking error.")

        self.create_timer(1.0 / args.rate, self.tick)
        print(f"\nRecording at {args.rate} Hz -> {self.out_path}")
        print("Start run_coverage.py now. Ctrl-C here when it finishes.\n")

    # ---------- setup helpers ----------

    def _subscribe_cmd(self, topic):
        # Subscribe to EXACTLY ONE type. ROS 2 forbids two subscriptions to
        # the same topic with incompatible types -- the second throws
        # "invalid allocator" and crashes the node. Discovery can lag at
        # startup, so when the type is not yet visible we default to
        # TwistStamped, which is what Clearpath's platform/cmd_vel always is
        # (enable_stamped_cmd_vel is true on the A200).
        found = dict(self.get_topic_names_and_types()).get(topic, [])
        if 'geometry_msgs/msg/Twist' in found and \
                'geometry_msgs/msg/TwistStamped' not in found:
            self.create_subscription(Twist, topic, self.on_twist, 10)
            kind = "Twist"
        else:
            self.create_subscription(TwistStamped, topic, self.on_stamped, 10)
            kind = ("TwistStamped" if found
                    else "TwistStamped (assumed; not advertised yet)")
        print(f"  cmd_vel      : {topic} ({kind})")

    def _autodetect_base(self):
        """Find the robot base frame by trying the usual Clearpath names.

        Guessing wrong is silent: lookup_transform just keeps failing and
        the CSV stays empty until Ctrl-C. Probing costs one call each.
        """
        candidates = [
            f"{self.ns}/base_link", f"{self.ns}/base_footprint",
            "base_link", "base_footprint",
        ]
        for frame in candidates:
            try:
                self.buf.lookup_transform(self.map_frame, frame,
                                          rclpy.time.Time(),
                                          timeout=Duration(seconds=0.2))
                return frame
            except Exception:
                continue
        return None

    # ---------- callbacks ----------

    def on_twist(self, msg):
        self.v, self.w = msg.linear.x, msg.angular.z

    def on_stamped(self, msg):
        self.v, self.w = msg.twist.linear.x, msg.twist.angular.z

    def on_truth(self, msg):
        p = msg.pose.pose
        self.truth = (p.position.x, p.position.y, yaw_of(p.orientation))
        self.truth_n += 1
        if self.truth_n == 1:
            print(f"  ground truth OK, first at "
                  f"({p.position.x:+.3f}, {p.position.y:+.3f})")

    # ---------- main loop ----------

    def tick(self):
        self.ticks += 1

        if self.base_frame is None:
            self.base_frame = self._autodetect_base()
            if self.base_frame is None:
                if self.ticks == self.frame_deadline:
                    self._frame_failure()
                return
            print(f"  base frame   : {self.base_frame} (autodetected)")

        try:
            t = self.buf.lookup_transform(self.map_frame, self.base_frame,
                                          rclpy.time.Time(),
                                          timeout=Duration(seconds=0.1))
        except Exception:
            self.miss += 1
            if self.miss in (20, 200, 1000):
                print(f"  [{self.miss} misses] {self.map_frame} -> "
                      f"{self.base_frame} unavailable. Is AMCL/Nav2 up and "
                      "has the initial pose been set?")
            return

        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = stamp
            print(f"  first sample at sim t={stamp:.2f}")

        tr = t.transform.translation
        tx, ty, tyaw = self.truth if self.truth else ('', '', '')
        self.rows.append((stamp - self.t0, tr.x, tr.y,
                          yaw_of(t.transform.rotation), self.v, self.w,
                          tx, ty, tyaw))

        if self.heartbeat_ticks and self.ticks % self.heartbeat_ticks == 0:
            self._heartbeat(tr)

    def _frame_failure(self):
        print()
        print("=" * 68)
        print("ABORTING: could not resolve the robot's base frame.")
        print()
        print(f"Tried, against '{self.map_frame}':")
        print(f"  {self.ns}/base_link, {self.ns}/base_footprint, "
              "base_link, base_footprint")
        print()
        print("Check what actually exists:")
        print("  ros2 run tf2_tools view_frames")
        print(f"  ros2 topic echo /{self.ns}/tf_static --once")
        print()
        print("Then pass it explicitly:  --base-frame <name>")
        print()
        print("If NO map frame exists at all, AMCL is not running or the")
        print("initial pose has not been set in RViz -- set it first.")
        print("=" * 68)
        rclpy.shutdown()

    def _heartbeat(self, tr):
        """Proof of life. A recorder quietly dropping a column looks
        identical to a working one until the run ends."""
        mins = self.rows[-1][0] / 60.0 if self.rows else 0.0
        line = (f"  [{mins:5.1f} min] {len(self.rows)} samples, "
                f"at ({tr.x:+.2f}, {tr.y:+.2f})")
        if self.truth_topic:
            new = self.truth_n - self.truth_last_n
            self.truth_last_n = self.truth_n
            line += ("  *** GROUND TRUTH STOPPED -- bridge died? ***"
                     if new == 0 else f", truth {self.truth_n}")
        print(line)

    # ---------- output ----------

    def save(self):
        if not self.rows:
            print("\nNo samples recorded -- nothing written.")
            print("Nav2 was probably not active, or the initial pose was "
                  "never set.")
            return

        with open(self.out_path, 'w', newline='') as f:
            wtr = csv.writer(f)
            wtr.writerow(['t', 'x', 'y', 'yaw', 'cmd_v', 'cmd_w',
                          'true_x', 'true_y', 'true_yaw'])
            wtr.writerows(self.rows)

        dur = self.rows[-1][0]
        dist = sum(math.hypot(self.rows[i][1] - self.rows[i - 1][1],
                              self.rows[i][2] - self.rows[i - 1][2])
                   for i in range(1, len(self.rows)))
        print(f"\nWrote {len(self.rows)} samples ({dur:.1f}s sim, "
              f"{dist:.2f} m driven) -> {self.out_path}")

        xs = [r[1] for r in self.rows]
        ys = [r[2] for r in self.rows]
        print(f"  extent: x {min(xs):+.2f}..{max(xs):+.2f}   "
              f"y {min(ys):+.2f}..{max(ys):+.2f}")
        print("  (office room is x -7.0..5.8, y -3.9..6.0 -- if the extent "
              "is much smaller, the run stopped early)")

        if self.truth_topic:
            self._report_localization_error()

        print(f"\nNext:")
        print(f"  python3 analyze_trajectory.py --track {self.out_path} \\")
        print(f"      --waypoints coverage_waypoints_bcd.yaml \\")
        print(f"      --map office_mapEmpty.yaml --out run_track.png")

    def _report_localization_error(self):
        paired = [r for r in self.rows if r[6] != '']
        if not paired:
            print("\nNo ground-truth samples landed in the file -- the bridge "
                  "was not publishing.")
            return

        ex = [r[1] - r[6] for r in paired]
        ey = [r[2] - r[7] for r in paired]
        mx, my = statistics.median(ex), statistics.median(ey)
        cx = [a - mx for a in ex]
        cy = [b - my for b in ey]
        cerr = [math.hypot(a, b) for a, b in zip(cx, cy)]

        print(f"\nAMCL POSE vs GROUND TRUTH  ({len(paired)} paired samples)")
        print(f"  constant offset : x {mx * 100:+.1f} cm, y {my * 100:+.1f} cm")
        if math.hypot(mx, my) > 0.10:
            print("    ^ large. Probably a map-vs-world origin offset rather")
            print("      than error; the figures below have it removed.")
        print(f"  mean |error|    : {sum(cerr) / len(cerr) * 100:.1f} cm")
        print(f"  max  |error|    : {max(cerr) * 100:.1f} cm")
        print(f"  sd in x         : {statistics.pstdev(cx) * 100:.1f} cm")
        print(f"  sd in y         : {statistics.pstdev(cy) * 100:.1f} cm")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--namespace', default='a200_1103',
                    help='Robot namespace. Check with: ros2 topic list | '
                         'grep -m1 a200')
    ap.add_argument('--out', default='run_track.csv')
    ap.add_argument('--rate', type=float, default=20.0,
                    help='Sample rate. 20 Hz matches the controller rate; '
                         'lower and you alias out the oscillation you are '
                         'trying to measure.')
    ap.add_argument('--cmd-topic', default=None,
                    help='Default /<ns>/platform/cmd_vel (POST twist-mux). '
                         '/<ns>/cmd_vel is what Nav2 asked for, not what the '
                         'wheels got.')
    ap.add_argument('--map-frame', default='map')
    ap.add_argument('--base-frame', default=None,
                    help='Skip autodetection and use this frame.')
    ap.add_argument('--frame-timeout', type=float, default=20.0,
                    help='Seconds to keep probing for the base frame.')
    ap.add_argument('--truth-topic', default=None,
                    help='nav_msgs/Odometry with the Gazebo ground-truth '
                         'pose. Off by default.')
    ap.add_argument('--heartbeat', type=float, default=60.0,
                    help='Seconds between progress lines. 0 disables.')
    args, _ = ap.parse_known_args()

    rclpy.init(args=sys.argv)
    print(f"Namespace: {args.namespace}")
    node = RunRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
