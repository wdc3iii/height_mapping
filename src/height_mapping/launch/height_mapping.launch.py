#!/usr/bin/env python3

from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.actions import IncludeLaunchDescription, LogInfo, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Declare launch argument for use_sim_time (default: false)
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time when playing back bag files'
    )

    # Get the launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Include the fast_lio_vel mapping launch file
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fast_lio_vel'),
                'launch',
                'mapping.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Height mapping node with parameters from YAML file
    config_file = PathJoinSubstitution([
        FindPackageShare('height_mapping'),
        'config',
        'hieght_mapping.yaml'
    ])

    height_mapping_node = Node(
        package='height_mapping',
        executable='height_mapping_node',
        name='height_map_node',  # Must match the node name in YAML file
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        # TODO: Debugging
        arguments=['--ros-args', '--log-level', 'debug']
    )

    # Log info about what's being launched
    log_info = LogInfo(
        msg="Launching FAST-LIO and Height Mapping pipeline..."
    )

    return LaunchDescription([
        use_sim_time_arg,
        log_info,
        fast_lio_launch,
        height_mapping_node
    ])