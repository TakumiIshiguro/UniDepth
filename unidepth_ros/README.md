# unidepth_ros

ROS Noetic integration for the local Python 3.8-compatible UniDepthV2 checkout.
It estimates metric depth from a camera image, publishes depth images and robot-
frame obstacle point clouds, and optionally saves visualization artifacts under
the repository-level `runs/` directory.

## Run

Place the ViT-S checkpoint as described in `../weights/README.md`, build the
catkin workspace, then run:

```bash
roslaunch unidepth_ros unidepth.launch
```

Configuration, including input/output topics and camera projection parameters,
is in `config/unidepth.yaml`. The launch file supports optional model path,
device, and inference-rate overrides.

Published topics by default:

- `/unidepth/depth`: metric depth (`sensor_msgs/Image`, `32FC1`)
- `/unidepth/depth_color`: colorized depth (`sensor_msgs/Image`, `bgr8`)
- `/unidepth/obstacle_points`: lateral-bin nearest points (`PointCloud2`)
- `/unidepth/obstacle_points_all`: all height-filtered points (`PointCloud2`)
