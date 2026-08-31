#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, GroupAction, RegisterEventHandler
from launch.conditions import IfCondition
from launch_ros.event_handlers import OnStateTransition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, PushRosNamespace
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

NAMESPACE = 'a200_1103'
HOME      = os.path.expanduser('~')
PARAMS    = os.path.join(HOME, 'clearpath', 'slam_toolbox_localization.yaml')

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
    DeclareLaunchArgument('autostart', default_value='true'),
]

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart    = LaunchConfiguration('autostart')

    # /tf and /tf_static don't automatically follow PushRosNamespace the
    # way ordinary topics do -- without an explicit remap, the node
    # broadcasts transforms to the global /tf topic instead of the
    # namespaced /a200_1103/tf that RViz and everything else in this
    # project listens on. This is exactly why RViz showed "Frame [map]
    # does not exist" despite the node being confirmed active: it WAS
    # broadcasting map->odom, just to the wrong topic. nav2_bringup's own
    # localization_launch.py (the AMCL one) already does this remap on
    # every node it launches -- this file just never had it copied over.
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    slam_toolbox_node = LifecycleNode(
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        namespace='',
        output='screen',
        parameters=[PARAMS, {'use_sim_time': use_sim_time}],
        remappings=remappings,
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(autostart),
    )

    # slam_toolbox does NOT self-activate after configure -- it was
    # assumed to, based on official slam_toolbox launch files only
    # showing a configure event, but that assumption was wrong: the node
    # sat in 'inactive' indefinitely (confirmed via RViz showing zero TF
    # from any link, and the node never publishing /tf or /map). It needs
    # an explicit ACTIVATE transition, fired only once configure has
    # actually finished -- waiting for the node to reach 'inactive'
    # (configure's target state) rather than firing both events
    # immediately, which could send ACTIVATE before CONFIGURE completes
    # and have it silently ignored.
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ),
        condition=IfCondition(autostart),
    )

    configure_to_active = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_toolbox_node,
            goal_state='inactive',
            entities=[activate_event],
        )
    )

    localization = GroupAction([
        PushRosNamespace(NAMESPACE),
        slam_toolbox_node,
        configure_event,
        configure_to_active,
    ])

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(localization)
    return ld