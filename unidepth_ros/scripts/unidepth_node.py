#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

from unidepth_ros.config import (
    load_config,
    repository_root,
    resolve_repository_path,
)
from unidepth_ros.depth_estimator import (
    DepthResultWriter,
    UniDepthV2Small,
    build_obstacle_occupancy_grid,
    colorize_depth,
    extract_nearest_obstacle_points,
    filter_metric_point_cloud,
    make_obstacle_occupancy_panel,
    render_obstacle_occupancy_map,
    transform_point_cloud,
)
from unidepth_ros.image_subscriber import LatestImageSubscriber


def _apply_ros_overrides(config):
    depth = config["depth"]
    runtime = config["runtime"]

    model_path = str(rospy.get_param("~model_path_override", "")).strip()
    if model_path:
        depth["model_path"] = model_path
    device = str(rospy.get_param("~device_override", "")).strip()
    if device:
        depth["device"] = device
    rate = float(rospy.get_param("~inference_rate_override", 0.0))
    if rate > 0.0:
        runtime["inference_rate"] = rate


def _copy_header(output_message: Image, input_message: Image) -> Image:
    output_message.header = input_message.header
    return output_message


def _obstacle_points_message(points, frame_id, stamp):
    header = Header(stamp=stamp, frame_id=frame_id)
    xyz_points = [
        (float(forward), float(left), 0.0)
        for forward, left in points
    ]
    return point_cloud2.create_cloud_xyz32(header, xyz_points)


def main():
    rospy.init_node("unidepth")
    config_file = rospy.get_param("~config_file", None)
    config = load_config(config_file)
    _apply_ros_overrides(config)

    depth_config = config["depth"]
    runtime = config["runtime"]
    projection_config = config["projection"]
    saving = config["saving"]
    topics = config["topics"]
    repository_path = repository_root()
    model_path = resolve_repository_path(depth_config["model_path"])
    required_files = (
        "config.json",
        "model.safetensors",
    )
    missing = [
        filename
        for filename in required_files
        if not os.path.isfile(os.path.join(model_path, filename))
    ]
    if missing:
        raise FileNotFoundError(
            "UniDepthV2 ViT-S files were not found in "
            f"{model_path}: {', '.join(missing)}. See weights/README.md."
        )
    if not os.path.isdir(os.path.join(repository_path, "unidepth")):
        raise FileNotFoundError(
            "UniDepth source package was not found in {}. See "
            "weights/README.md.".format(repository_path)
        )

    estimator = UniDepthV2Small(
        depth_config,
        repository_path,
        model_path,
    )
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    bridge = CvBridge()
    subscriber = LatestImageSubscriber(topics["image_topic"])
    depth_publisher = rospy.Publisher(
        topics["depth_topic"],
        Image,
        queue_size=1,
    )
    obstacle_points_publisher = rospy.Publisher(
        topics["obstacle_points_topic"],
        PointCloud2,
        queue_size=1,
    )
    all_obstacle_points_publisher = rospy.Publisher(
        topics["all_obstacle_points_topic"],
        PointCloud2,
        queue_size=1,
    )
    color_publisher = None
    if runtime["publish_color"]:
        color_publisher = rospy.Publisher(
            topics["depth_color_topic"],
            Image,
            queue_size=1,
        )

    result_writer = None
    if saving["enabled"]:
        output_dir = resolve_repository_path(saving["output_dir"])
        result_writer = DepthResultWriter(
            output_dir=output_dir,
            save_raw_depth=saving["save_raw_depth"],
            save_color=saving["save_color"],
            save_obstacle_mask=saving["save_obstacle_mask"],
            save_obstacle_points_2d=saving["save_obstacle_points_2d"],
            save_occupancy_grid=saving["save_occupancy_grid"],
            save_map=saving["save_map"],
            save_panel=saving["save_panel"],
        )
        rospy.loginfo(
            "depth results will be saved to %s",
            result_writer.session_dir,
        )

    rate_hz = float(runtime["inference_rate"])
    rate = rospy.Rate(rate_hz)
    rospy.loginfo(
        "UniDepthV2 ViT-S loaded from %s on %s "
        "(parameters=%d, resolution_level=%d, depth_scale=%.3f, "
        "camera_model=%s, horizontal_fov=%.3f rad, rate=%.2f Hz, "
        "transform=%s->%s)",
        model_path,
        estimator.device,
        estimator.parameter_count,
        depth_config["resolution_level"],
        depth_config["depth_scale"],
        depth_config["camera_model"],
        depth_config["horizontal_fov_rad"],
        rate_hz,
        projection_config["camera_optical_frame"],
        projection_config["robot_frame"],
    )

    while not rospy.is_shutdown():
        image_message = subscriber.take_latest()
        if image_message is None:
            rate.sleep()
            continue

        try:
            rgb = bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding="rgb8",
            )
            prediction = estimator.predict(PILImage.fromarray(rgb))
            depth_meters = prediction.depth_meters
            depth_message = bridge.cv2_to_imgmsg(
                depth_meters,
                encoding="32FC1",
            )
            depth_publisher.publish(
                _copy_header(depth_message, image_message)
            )

            color = None
            if color_publisher is not None or (
                result_writer is not None
                and (saving["save_color"] or saving["save_panel"])
            ):
                color = colorize_depth(
                    depth_meters,
                    runtime["color_min_percentile"],
                    runtime["color_max_percentile"],
                )
            if color_publisher is not None:
                color_message = bridge.cv2_to_imgmsg(
                    color,
                    encoding="bgr8",
                )
                color_publisher.publish(
                    _copy_header(color_message, image_message)
                )

            camera_points, valid_depth_mask = filter_metric_point_cloud(
                prediction.points_camera,
                depth_meters,
                minimum_depth_m=projection_config["minimum_depth_m"],
                maximum_depth_m=projection_config["maximum_depth_m"],
            )
            intrinsics = prediction.intrinsics
            rospy.loginfo_once(
                "UniDepth input camera parameters: fx=%.2f fy=%.2f "
                "cx=%.2f cy=%.2f",
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            )
            transform = tf_buffer.lookup_transform(
                projection_config["robot_frame"],
                projection_config["camera_optical_frame"],
                image_message.header.stamp,
                rospy.Duration(projection_config["tf_timeout_seconds"]),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            robot_points = transform_point_cloud(
                camera_points,
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w),
            )
            (
                _care_obstacle_mask,
                care_obstacle_points_2d,
                obstacle_mask,
                obstacle_points_2d,
            ) = extract_nearest_obstacle_points(
                robot_points,
                valid_depth_mask,
                minimum_forward_m=projection_config[
                    "minimum_forward_m"
                ],
                maximum_forward_m=projection_config[
                    "maximum_forward_m"
                ],
                minimum_height_m=projection_config[
                    "minimum_height_m"
                ],
                maximum_height_m=projection_config[
                    "maximum_height_m"
                ],
                map_width_m=projection_config["obstacle_width_m"],
                lateral_bins=projection_config["lateral_bins"],
                border_margin_ratio=projection_config[
                    "border_margin_ratio"
                ],
                return_all_points=True,
            )
            obstacle_points_publisher.publish(
                _obstacle_points_message(
                    care_obstacle_points_2d,
                    projection_config["robot_frame"],
                    image_message.header.stamp,
                )
            )
            all_obstacle_points_publisher.publish(
                _obstacle_points_message(
                    obstacle_points_2d,
                    projection_config["robot_frame"],
                    image_message.header.stamp,
                )
            )

            if result_writer is not None:
                occupancy_grid = None
                map_image = None
                panel = None
                if (
                    saving["save_occupancy_grid"]
                    or saving["save_map"]
                    or saving["save_panel"]
                ):
                    occupancy_grid = build_obstacle_occupancy_grid(
                        obstacle_points_2d,
                        map_width_m=projection_config["map_width_m"],
                        map_height_m=projection_config["map_height_m"],
                        grid_size=projection_config["canvas_size"],
                        obstacle_radius_cells=projection_config[
                            "obstacle_radius_cells"
                        ],
                    )
                if saving["save_map"] or saving["save_panel"]:
                    map_image = render_obstacle_occupancy_map(
                        occupancy_grid,
                        map_width_m=projection_config["map_width_m"],
                        map_height_m=projection_config["map_height_m"],
                    )
                if saving["save_panel"]:
                    panel = make_obstacle_occupancy_panel(
                        rgb,
                        color,
                        obstacle_mask,
                        map_image,
                    )
                result_writer.save(
                    depth_meters,
                    color,
                    obstacle_mask=obstacle_mask,
                    obstacle_points_2d=obstacle_points_2d,
                    occupancy_grid=occupancy_grid,
                    map_image=map_image,
                    panel=panel,
                    stamp=image_message.header.stamp.to_sec(),
                )
        except (
            CvBridgeError,
            OSError,
            RuntimeError,
            ValueError,
            tf2_ros.TransformException,
        ) as error:
            rospy.logwarn_throttle(
                5.0,
                f"failed to estimate metric depth: {error}",
            )
        rate.sleep()


if __name__ == "__main__":
    main()
