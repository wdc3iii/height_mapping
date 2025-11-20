#!/usr/bin/env python3

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.conditions import IfCondition
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
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Launch RViz for visualization'
    )

    # Get the launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_use = LaunchConfiguration('rviz')

    log_rviz = LogInfo(msg=['rviz flag: ', rviz_use])

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
            'use_sim_time': use_sim_time,
            'rviz': 'false'
        }.items()
    )

    # Height mapping node with parameters from YAML file
    config_file = PathJoinSubstitution([
        FindPackageShare('height_mapping'),
        'config',
        'height_mapping.yaml'
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
    )

    top_rviz_config = PathJoinSubstitution([
        FindPackageShare('height_mapping'),
        'rviz',
        'heightmap.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', top_rviz_config],
        # condition=IfCondition(rviz_use),
    )

    # Log info about what's being launched
    log_info = LogInfo(
        msg="Launching FAST-LIO and Height Mapping pipeline..."
    )

    static_tf_body_pelvis = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_body_pelvis",
        arguments=[
            # x y z roll pitch yaw
            "0.01911", "0.0", "0.47580",   # x y z
            "0.0", "0.0401426", "0.0",     # roll pitch yaw (rad)
            "body",
            "pelvis",
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        rviz_arg,
        log_info,
        log_rviz,
        static_tf_body_pelvis,
        fast_lio_launch,
        height_mapping_node,
        rviz_node
    ])