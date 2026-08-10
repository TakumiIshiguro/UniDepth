import os
from typing import Any, Dict

import yaml


def package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def repository_root() -> str:
    return os.path.abspath(os.path.join(package_root(), ".."))


def resolve_repository_path(path: str) -> str:
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(repository_root(), expanded))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(config_file: str = None) -> Dict[str, Any]:
    if config_file is None:
        config_file = os.path.join(package_root(), "config", "unidepth.yaml")
    config_file = os.path.abspath(os.path.expanduser(config_file))
    data = load_yaml(config_file)
    config = {
        "depth": dict(data.get("depth", {})),
        "runtime": dict(data.get("runtime", {})),
        "projection": dict(data.get("projection", {})),
        "saving": dict(data.get("saving", {})),
        "topics": dict(data.get("topics", {})),
    }
    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    depth = config["depth"]
    runtime = config["runtime"]
    projection = config["projection"]
    saving = config["saving"]
    topics = config["topics"]

    required_depth_keys = (
        "model_path",
        "resolution_level",
        "depth_scale",
        "device",
    )
    missing = [key for key in required_depth_keys if key not in depth]
    if missing:
        raise ValueError(f"depth config is missing keys: {', '.join(missing)}")
    if not str(depth["model_path"]).strip():
        raise ValueError("depth.model_path must not be empty")
    resolution_level = int(depth["resolution_level"])
    if not 0 <= resolution_level < 10:
        raise ValueError("depth.resolution_level must be in [0, 10)")
    depth_scale = float(depth["depth_scale"])
    if depth_scale <= 0.0:
        raise ValueError("depth.depth_scale must be positive")
    camera_model = str(depth.get("camera_model", "estimated_pinhole")).strip()
    if camera_model not in ("estimated_pinhole", "gazebo_equidistant"):
        raise ValueError(
            "depth.camera_model must be 'estimated_pinhole' or "
            "'gazebo_equidistant'"
        )
    horizontal_fov_rad = float(depth.get("horizontal_fov_rad", 2.62))
    if not 0.0 < horizontal_fov_rad < 3.141592653589793:
        raise ValueError("depth.horizontal_fov_rad must be in (0, pi)")
    depth.update(
        {
            "model_path": str(depth["model_path"]).strip(),
            "resolution_level": resolution_level,
            "depth_scale": depth_scale,
            "device": str(depth["device"]).strip(),
            "camera_model": camera_model,
            "horizontal_fov_rad": horizontal_fov_rad,
        }
    )

    inference_rate = float(runtime.get("inference_rate", 0.0))
    if inference_rate <= 0.0:
        raise ValueError("runtime.inference_rate must be positive")
    color_min = float(runtime.get("color_min_percentile", 0.0))
    color_max = float(runtime.get("color_max_percentile", 100.0))
    if not 0.0 <= color_min < color_max <= 100.0:
        raise ValueError(
            "runtime color percentiles must satisfy 0 <= min < max <= 100"
        )
    runtime.update(
        {
            "inference_rate": inference_rate,
            "publish_color": bool(runtime.get("publish_color", True)),
            "color_min_percentile": color_min,
            "color_max_percentile": color_max,
        }
    )

    defaults = {
        "minimum_depth_m": 0.1,
        "maximum_depth_m": 20.0,
        "camera_optical_frame": "camera_rgb_optical_frame",
        "robot_frame": "base_footprint",
        "tf_timeout_seconds": 0.1,
        "minimum_forward_m": 0.05,
        "maximum_forward_m": 3.0,
        "minimum_height_m": 0.1,
        "maximum_height_m": 1.8,
        "lateral_bins": 180,
        "border_margin_ratio": 0.02,
        "obstacle_width_m": 6.0,
        "map_width_m": 1.0,
        "map_height_m": 3.0,
        "canvas_size": 512,
        "obstacle_radius_cells": 2,
    }
    float_keys = (
        "minimum_depth_m", "maximum_depth_m", "tf_timeout_seconds",
        "minimum_forward_m", "maximum_forward_m", "minimum_height_m",
        "maximum_height_m", "border_margin_ratio", "obstacle_width_m",
        "map_width_m", "map_height_m",
    )
    for key in float_keys:
        projection[key] = float(projection.get(key, defaults[key]))
    projection["camera_optical_frame"] = str(
        projection.get("camera_optical_frame", defaults["camera_optical_frame"])
    ).strip()
    projection["robot_frame"] = str(
        projection.get("robot_frame", defaults["robot_frame"])
    ).strip()
    projection["lateral_bins"] = int(
        projection.get("lateral_bins", projection.get("azimuth_bins", 180))
    )
    projection["canvas_size"] = int(
        projection.get("canvas_size", defaults["canvas_size"])
    )
    projection["obstacle_radius_cells"] = int(
        projection.get(
            "obstacle_radius_cells", defaults["obstacle_radius_cells"]
        )
    )

    if projection["minimum_depth_m"] < 0.0:
        raise ValueError("projection.minimum_depth_m must be non-negative")
    if projection["maximum_depth_m"] <= projection["minimum_depth_m"]:
        raise ValueError("projection.maximum_depth_m must exceed minimum_depth_m")
    if not projection["camera_optical_frame"] or not projection["robot_frame"]:
        raise ValueError("projection frame names must not be empty")
    if projection["tf_timeout_seconds"] <= 0.0:
        raise ValueError("projection.tf_timeout_seconds must be positive")
    if projection["minimum_forward_m"] < 0.0:
        raise ValueError("projection.minimum_forward_m must be non-negative")
    if projection["maximum_forward_m"] <= projection["minimum_forward_m"]:
        raise ValueError(
            "projection.maximum_forward_m must exceed minimum_forward_m"
        )
    if projection["maximum_height_m"] <= projection["minimum_height_m"]:
        raise ValueError("projection.maximum_height_m must exceed minimum_height_m")
    if projection["lateral_bins"] <= 0:
        raise ValueError("projection.lateral_bins must be positive")
    if not 0.0 <= projection["border_margin_ratio"] < 0.5:
        raise ValueError("projection.border_margin_ratio must be in [0, 0.5)")
    for key in ("obstacle_width_m", "map_width_m", "map_height_m"):
        if projection[key] <= 0.0:
            raise ValueError(f"projection.{key} must be positive")
    if projection["maximum_forward_m"] > projection["map_height_m"]:
        raise ValueError(
            "projection.maximum_forward_m must not exceed map_height_m"
        )
    if projection["canvas_size"] < 128:
        raise ValueError("projection.canvas_size must be at least 128")
    if projection["obstacle_radius_cells"] < 0:
        raise ValueError("projection.obstacle_radius_cells must be non-negative")

    saving_defaults = {
        "enabled": False,
        "output_dir": "runs/depth_estimation",
        "save_raw_depth": True,
        "save_color": True,
        "save_obstacle_mask": True,
        "save_obstacle_points_2d": True,
        "save_occupancy_grid": True,
        "save_map": True,
        "save_panel": True,
    }
    for key, default in saving_defaults.items():
        if key == "output_dir":
            saving[key] = str(saving.get(key, default)).strip()
        else:
            saving[key] = bool(saving.get(key, default))
    if saving["enabled"] and not saving["output_dir"]:
        raise ValueError("saving.output_dir must not be empty when saving is enabled")
    output_flags = [key for key in saving_defaults if key.startswith("save_")]
    if saving["enabled"] and not any(saving[key] for key in output_flags):
        raise ValueError("saving must enable at least one output format")

    required_topics = (
        "image_topic",
        "depth_topic",
        "obstacle_points_topic",
        "all_obstacle_points_topic",
    )
    if runtime["publish_color"]:
        required_topics += ("depth_color_topic",)
    missing = [key for key in required_topics if not topics.get(key)]
    if missing:
        raise ValueError(f"topics config is missing keys: {', '.join(missing)}")
