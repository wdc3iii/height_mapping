#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
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
            'use_sim_time': 'true'
        }.items()
    )
    
    # Height mapping node with parameters
    height_mapping_node = Node(
        package='height_mapping',
        executable='height_mapping_node',
        name='height_mapping_node',
        parameters=[{
            'use_sim_time': True,
            'map_frame': 'odom',
            'base_frame': 'body', 
            'topic_cloud': '/cloud_registered',
            'livox_frame': '',
            'resolution': 0.2,
            'big_width': 100,
            'big_height': 100,
            'max_height': 2.0,
            'z_min': -20.0,
            'z_max': 20.0,
            'drop_thresh': 0.07,
            'min_support': 2,
            'shift_thresh_m': 0.5,
            'sub_width': 200,
            'sub_height': 200,
            'sub_resolution': 0.05,
            'publish_rate_hz': 10.0,
            'voxel_downsample': False,
            'voxel_leaf': 0.05,
            'transform_cloud': True
        }],
        output='screen',
        respawn=True,
        respawn_delay=2.0
    )

    # Log info about what's being launched
    log_info = LogInfo(
        msg="Launching FAST-LIO and Height Mapping pipeline..."
    )

    return LaunchDescription([
        log_info,
        fast_lio_launch,
        height_mapping_node
    ])