#!/usr/bin/env python3
"""
slam_toolbox_localization_launch.py

Drop-in alternative to localization_custom_launch.py (which brings up
map_server + AMCL). This brings up SLAM Toolbox's own localization-mode
launch file instead, with our params overlaid -- same pattern this
project already uses for AMCL (include the vendor/package launch file,
override params_file, don't reimplement its internals). Never run this
alongside localization_custom_launch.py: only one node may own the
map->odom transform, same lesson as the duplicate Route Server bug
(Section 4.7 of the report).

PREREQUISITE: see the comment block at the top of
slam_toolbox_localization.yaml -- this needs a serialized .posegraph
map, not the plain office_map.pgm/.yaml AMCL uses. Verify both files
exist before your first comparison run.

Run this AFTER simulation.launch.py, IN PLACE OF localization_custom_launch.py:
  ros2 launch <pkg> slam_toolbox_localization_launch.py
Then run_coverage.py and log_localization_covariance.py exactly as with
the AMCL arm -- only the pose topic name changes (see --topic in the
logger); Nav2 and run_coverage.py are unaffected either way since both
localization nodes provide the same map->odom TF contract.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_share_directory

NAMESPACE = 'a200_1103'
HOME      = os.path.expanduser('~')
PARAMS    = os.path.join(HOME, 'clearpath', 'slam_toolbox_localization.yaml')

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
]

def launch_setup(context, *args, **kwargs):
    pkg_slam_toolbox = get_package_share_directory('slam_toolbox')
    localization = GroupAction([
        PushRosNamespace(NAMESPACE),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_slam_toolbox, 'launch', 'localization_launch.py'])
            ),
            launch_arguments=[
                ('use_sim_time',     LaunchConfiguration('use_sim_time')),
                ('slam_params_file', PARAMS),
            ]
        ),
    ])
    return [localization]

def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld