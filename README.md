# Height Mapping Repository

This repository performs height mapping on a Livox MID360 Lidar.
It contains a mild fork of FAST-LIO, which includes publishing a frame transform such that the odom frame is 'z-up' -- useful for height mapping.
Pointclouds transformed into the odom frame from FAST-LIO are then minimized, with some outlier detection and filling of unobserved squares.

A large heightmap is maintained on one thread; this heightmap remains axis aligned with the odom frame, and ingests points from lidar as fast as possible.
On a separate thread, the larger heightmap is queried from an arbitary SE(2) pose.
Maintaining the larger map helps avoid frequent map shifts, as well as having to deal with orientation changes during map shifting.

## Installation
The `docker` branch of this repository maintains a docker image which can be used to run the heightmapping software in ROS2 humble. 

### Heightmap only
To run only the heightmap, this code can be cloned (be sure to run `git submodule update --init --recursive` to pull the submoduled FAST-LIO fork); create an environment variable `$HEIGHT_MAPPING_ROOT=<path-to-repo>`.
Additionally, the installed livox drivers need to be updated with the proper ip addresses for the lidar unit in the file `TODO`. For the 'nice' G1 in Amber lab, this is:

```TODO```

Which comes pre-set on the docker image. If using a different lidar, the IP addresses of the host and the lidar need to be set appropriately. 
Then, build docker image. To install terminal aliases, run `source scripts/setup_aliases.bash`. Using the installed terminal aliases, run `livox`, which performs:

```source /opt/ros/humble/setup.bash && source /ws_livox/install/setup.bash```

Then, in `~/repos/height_mapping` run `colcon build --symlink-install` to build both the FAST-LIO fork and `height_mapping`. Finally, run `source install/setup.bash` from `$HEIGHT_MAPPING_ROOT` to source the install, or use the alias `height_mapping`. 

### Integration with downstream project
To integrate with a downstream project, simply submodule the main repository (this has not been tested yet...). Note dependencies on the [Livox SDK2](https://github.com/Livox-SDK/Livox-SDK2) and [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2). 
Additionally, note that the config file `TODO` for the Livox SDK2 needs to be updated with proper host and lidar IP addresses (see section above for details). 

## Running
Currently, to run the heightmap, the Lidar must be launched first, then the height mapping launch file (these will be combined soon). To launch the lidar, run

```ros2 launch livox_ros_driver2 ...```

And then to run the height mapping code:

```ros2 launch height_mapping height_mapping.launch.py --use_sim_time:=<bool> --rviz:=<bool>```

Where `use_sim_time` should be included, as true, if running the height mapping on a pre-recorded bag file (which should be played with `--clock`), and `--rviz:=<true` should be included to launch an rviz session to visualize the output of the height mapping. 

The launch file launches a height mapping node with parameters in `$HEIGHT_MAPPING_ROOT/src/height_mapping/config/height_mapping.yaml`, and FAST-LIO gets launch with parameters in `$HEIGHT_MAPPING_ROOT/src/fastlio_vel/config/mid360_g1.yaml`.

