import os

import pytest

from unidepth_ros.config import (
    load_config,
    package_root,
    repository_root,
    resolve_repository_path,
)


def test_default_config_uses_local_vits_checkpoint_and_unidepth_topics():
    config = load_config()

    assert config["depth"]["model_path"] == "weights/unidepth_v2_vits14_hf"
    assert config["depth"]["resolution_level"] == 0
    assert config["depth"]["depth_scale"] == pytest.approx(0.6)
    assert config["depth"]["camera_model"] == "gazebo_equidistant"
    assert config["projection"]["maximum_forward_m"] == pytest.approx(1.0)
    assert config["topics"]["depth_topic"] == "/unidepth/depth"
    assert config["topics"]["obstacle_points_topic"] == (
        "/unidepth/obstacle_points"
    )


def test_repository_relative_paths_resolve_outside_nested_ros_package():
    assert repository_root() == os.path.dirname(package_root())
    assert resolve_repository_path("weights/example") == os.path.join(
        repository_root(), "weights", "example"
    )


def test_resolution_level_is_validated(tmp_path):
    config_file = tmp_path / "unidepth.yaml"
    config_file.write_text(
        "depth:\n"
        "  model_path: weights/model\n"
        "  resolution_level: 10\n"
        "  depth_scale: 1.0\n"
        "  device: cpu\n"
        "runtime:\n"
        "  inference_rate: 1.0\n"
        "projection:\n"
        "  maximum_forward_m: 1.0\n"
        "  map_height_m: 1.0\n"
        "topics:\n"
        "  image_topic: /image\n"
        "  depth_topic: /depth\n"
        "  depth_color_topic: /depth_color\n"
        "  obstacle_points_topic: /points\n"
        "  all_obstacle_points_topic: /points_all\n"
    )

    with pytest.raises(ValueError, match="resolution_level"):
        load_config(str(config_file))
