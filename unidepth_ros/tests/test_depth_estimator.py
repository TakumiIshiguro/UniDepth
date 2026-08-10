import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from unidepth_ros.depth_estimator import (
    DepthResultWriter,
    UniDepthV2Small,
    build_obstacle_occupancy_grid,
    colorize_depth,
    extract_nearest_obstacle_points,
    filter_metric_point_cloud,
    gazebo_equidistant_camera_parameters,
    make_obstacle_occupancy_panel,
    render_obstacle_occupancy_map,
    transform_point_cloud,
)


class FakeUniDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_rgb = None
        self.resolution_level = None
        self.last_camera = None

    def infer(self, rgb, camera=None):
        self.last_rgb = rgb
        self.last_camera = camera
        height, width = rgb.shape[-2:]
        values = torch.linspace(
            0.5,
            4.0,
            height * width,
            device=self.scale.device,
        ).reshape(1, 1, height, width) * self.scale
        x = torch.zeros_like(values)
        y = torch.zeros_like(values)
        points = torch.cat((x, y, values), dim=1)
        intrinsics = torch.tensor(
            [[[100.0, 0.0, width / 2.0], [0.0, 101.0, height / 2.0], [0.0, 0.0, 1.0]]],
            device=self.scale.device,
        )
        return {
            "depth": values,
            "points": points,
            "intrinsics": intrinsics,
        }


def depth_config():
    return {
        "resolution_level": 0,
        "depth_scale": 1.0,
        "device": "cpu",
    }


def test_unidepth_estimator_returns_depth_points_and_intrinsics():
    model = FakeUniDepthModel()
    estimator = UniDepthV2Small(
        depth_config(),
        repository_path="unused",
        model_path="unused",
        model=model,
    )

    prediction = estimator.predict(Image.new("RGB", (64, 48)))

    assert prediction.depth_meters.shape == (48, 64)
    assert prediction.depth_meters.dtype == np.float32
    assert prediction.points_camera.shape == (48, 64, 3)
    assert prediction.points_camera.dtype == np.float32
    assert prediction.intrinsics.shape == (3, 3)
    assert prediction.intrinsics.dtype == np.float32
    assert prediction.points_camera[..., 2] == pytest.approx(
        prediction.depth_meters
    )
    assert model.last_rgb.shape == (3, 48, 64)
    assert model.last_rgb.dtype == torch.uint8
    assert model.resolution_level == 0
    assert estimator.parameter_count == 1


def test_unidepth_estimator_applies_metric_scale_to_depth_and_points():
    unscaled = UniDepthV2Small(
        depth_config(),
        repository_path="unused",
        model_path="unused",
        model=FakeUniDepthModel(),
    )
    scaled_config = depth_config()
    scaled_config["depth_scale"] = 0.8
    scaled = UniDepthV2Small(
        scaled_config,
        repository_path="unused",
        model_path="unused",
        model=FakeUniDepthModel(),
    )
    image = Image.new("RGB", (64, 48))
    unscaled_prediction = unscaled.predict(image)
    scaled_prediction = scaled.predict(image)

    assert scaled_prediction.depth_meters == pytest.approx(
        unscaled_prediction.depth_meters * 0.8
    )
    assert scaled_prediction.points_camera == pytest.approx(
        unscaled_prediction.points_camera * 0.8
    )


class FakeFisheyeCamera:
    def __init__(self, parameters):
        self.params = parameters


def test_unidepth_receives_gazebo_equidistant_camera_parameters():
    config = depth_config()
    config.update(
        {
            "camera_model": "gazebo_equidistant",
            "horizontal_fov_rad": 2.62,
        }
    )
    model = FakeUniDepthModel()
    estimator = UniDepthV2Small(
        config,
        repository_path="unused",
        model_path="unused",
        model=model,
        camera_class=FakeFisheyeCamera,
    )

    prediction = estimator.predict(Image.new("RGB", (640, 480)))

    expected_focal_length = 640.0 / 2.62
    assert isinstance(model.last_camera, FakeFisheyeCamera)
    assert model.last_camera.params.shape == (1, 16)
    assert model.last_camera.params[0, :4].numpy() == pytest.approx(
        [expected_focal_length, expected_focal_length, 319.5, 239.5]
    )
    assert prediction.intrinsics == pytest.approx(
        np.asarray(
            [
                [expected_focal_length, 0.0, 319.5],
                [0.0, expected_focal_length, 239.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


def test_gazebo_equidistant_parameters_validate_inputs():
    with pytest.raises(ValueError, match="image dimensions"):
        gazebo_equidistant_camera_parameters(0, 480, 2.62)
    with pytest.raises(ValueError, match="horizontal_fov_rad"):
        gazebo_equidistant_camera_parameters(640, 480, np.pi)


def test_unidepth_resolution_level_is_validated():
    config = depth_config()
    config["resolution_level"] = 10

    with pytest.raises(ValueError, match="resolution_level"):
        UniDepthV2Small(
            config,
            repository_path="unused",
            model_path="unused",
            model=FakeUniDepthModel(),
        )


def test_colorize_depth_returns_bgr_uint8():
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)

    color = colorize_depth(depth)

    assert color.shape == (4, 6, 3)
    assert color.dtype == np.uint8
    assert not np.array_equal(color[0, 0], color[-1, -1])


def test_colorize_masks_non_finite_pixels():
    depth = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float32)

    color = colorize_depth(depth, 0.0, 100.0)

    assert np.array_equal(color[1, 0], np.zeros(3, dtype=np.uint8))


def test_colorize_rejects_array_without_finite_depth():
    with pytest.raises(ValueError, match="finite value"):
        colorize_depth(
            np.full((2, 2), np.nan, dtype=np.float32)
        )


def test_result_writer_saves_raw_depth_color_and_metadata(tmp_path):
    writer = DepthResultWriter(
        output_dir=str(tmp_path),
        session_name="test_session",
    )
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)
    color = colorize_depth(depth, 0.0, 100.0)

    paths = writer.save(depth, color, stamp=123.25)

    loaded_depth = np.load(paths["raw_depth"], allow_pickle=False)
    assert np.array_equal(loaded_depth, depth)
    assert paths["raw_depth"].endswith("depth_meters/000000.npy")
    assert paths["color"].endswith("color/000000.png")
    assert cv2.imread(paths["color"]).shape == (4, 6, 3)
    metadata = (tmp_path / "test_session" / "metadata.csv").read_text()
    assert "index,stamp,height,width" in metadata
    assert "0,123.250000000,4,6" in metadata


def test_result_writer_uses_incrementing_filenames(tmp_path):
    writer = DepthResultWriter(
        output_dir=str(tmp_path),
        save_color=False,
        session_name="test_session",
    )
    depth = np.ones((2, 3), dtype=np.float32)

    first = writer.save(depth)
    second = writer.save(depth)

    assert first["raw_depth"].endswith("000000.npy")
    assert second["raw_depth"].endswith("000001.npy")
    assert first["color"] == ""


def test_unidepth_point_cloud_is_filtered_without_reprojection():
    depth = np.asarray([[0.05, 2.0], [12.0, 3.0]], dtype=np.float32)
    points = np.asarray(
        [
            [[-1.0, -2.0, 0.05], [4.0, 5.0, 2.0]],
            [[6.0, 7.0, 12.0], [8.0, 9.0, 3.0]],
        ],
        dtype=np.float32,
    )

    filtered, valid = filter_metric_point_cloud(
        points,
        depth,
        minimum_depth_m=0.1,
        maximum_depth_m=10.0,
    )

    assert np.array_equal(valid, [[False, True], [False, True]])
    assert np.array_equal(filtered[0, 1], points[0, 1])
    assert np.array_equal(filtered[1, 1], points[1, 1])
    assert np.array_equal(filtered[0, 0], np.zeros(3))
    assert np.array_equal(filtered[1, 0], np.zeros(3))


def test_result_writer_saves_obstacle_map_outputs(tmp_path):
    writer = DepthResultWriter(
        output_dir=str(tmp_path),
        save_raw_depth=False,
        save_color=False,
        save_obstacle_mask=True,
        save_obstacle_points_2d=True,
        save_occupancy_grid=True,
        save_map=True,
        save_panel=True,
        session_name="obstacle_map_session",
    )
    depth = np.ones((4, 6), dtype=np.float32)
    obstacle_mask = np.full((4, 6), 255, dtype=np.uint8)
    obstacle_points = np.ones((6, 2), dtype=np.float32)
    grid = np.full((128, 128), -1, dtype=np.int8)
    map_image = np.zeros((128, 128, 3), dtype=np.uint8)
    panel = np.zeros((162, 512, 3), dtype=np.uint8)

    paths = writer.save(
        depth,
        obstacle_mask=obstacle_mask,
        obstacle_points_2d=obstacle_points,
        occupancy_grid=grid,
        map_image=map_image,
        panel=panel,
    )

    assert np.array_equal(
        np.load(paths["obstacle_points_2d"], allow_pickle=False),
        obstacle_points,
    )
    assert np.array_equal(
        np.load(paths["occupancy_grid"], allow_pickle=False),
        grid,
    )
    assert paths["obstacle_mask"].endswith(
        "obstacle_mask/000000.png"
    )
    assert paths["map"].endswith("000000_map.png")
    assert paths["visualization"].endswith(
        "000000_visualization.png"
    )


def test_point_cloud_is_transformed_to_robot_frame():
    camera_points = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    half_angle = np.pi / 4.0

    robot_points = transform_point_cloud(
        camera_points,
        translation_xyz=(1.0, 2.0, 3.0),
        quaternion_xyzw=(0.0, 0.0, np.sin(half_angle), np.cos(half_angle)),
    )

    assert robot_points[0, 0] == pytest.approx((1.0, 3.0, 3.0))


def test_nearest_obstacles_are_selected_per_lateral_bin():
    robot_points = np.asarray(
        [
            [
                (2.0, -0.5, 0.5),
                (1.0, -0.4, 0.5),
                (0.7, 0.5, 0.05),
            ],
            [
                (1.5, 0.5, 0.6),
                (0.8, 0.6, 2.0),
                (4.0, 1.5, 0.6),
            ],
        ],
        dtype=np.float32,
    )

    mask, obstacles = extract_nearest_obstacle_points(
        robot_points,
        np.ones((2, 3), dtype=bool),
        minimum_forward_m=0.05,
        maximum_forward_m=3.0,
        minimum_height_m=0.1,
        maximum_height_m=1.8,
        map_width_m=4.0,
        lateral_bins=4,
        border_margin_ratio=0.0,
    )

    assert np.count_nonzero(mask) == 2
    assert obstacles.shape == (2, 2)
    assert any(np.allclose(point, (1.0, -0.4)) for point in obstacles)
    assert any(np.allclose(point, (1.5, 0.5)) for point in obstacles)


def test_lateral_bins_keep_only_nearest_point_on_each_side_wall():
    robot_points = np.asarray(
        [
            [(1.0, -1.0, 0.5), (2.0, -1.0, 0.5), (3.0, -1.0, 0.5)],
            [(1.0, 1.0, 0.5), (2.0, 1.0, 0.5), (3.0, 1.0, 0.5)],
        ],
        dtype=np.float32,
    )

    mask, obstacles, all_mask, all_obstacles = extract_nearest_obstacle_points(
        robot_points,
        np.ones((2, 3), dtype=bool),
        minimum_forward_m=0.05,
        maximum_forward_m=4.0,
        minimum_height_m=0.1,
        maximum_height_m=1.8,
        map_width_m=4.0,
        lateral_bins=36,
        border_margin_ratio=0.0,
        return_all_points=True,
    )

    assert np.count_nonzero(mask) == 2
    assert obstacles.shape == (2, 2)
    assert np.count_nonzero(all_mask) == 6
    assert all_obstacles.shape == (6, 2)
    right_wall = obstacles[obstacles[:, 1] < 0.0]
    left_wall = obstacles[obstacles[:, 1] > 0.0]
    assert np.ptp(right_wall[:, 1]) == pytest.approx(0.0)
    assert np.ptp(left_wall[:, 1]) == pytest.approx(0.0)
    assert right_wall[0, 0] == pytest.approx(1.0)
    assert left_wall[0, 0] == pytest.approx(1.0)


def test_obstacle_map_renders_robot_coordinate_axes():
    grid = np.full((128, 128), -1, dtype=np.int8)

    image = render_obstacle_occupancy_map(
        grid,
        map_width_m=6.0,
        map_height_m=3.0,
    )

    assert image.shape == (128, 128, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image[64, 64], np.array([30, 100, 220]))
    assert not np.array_equal(image[102, 10], np.array([170, 170, 170]))


def test_obstacle_map_rejects_non_positive_metric_extent():
    grid = np.full((128, 128), -1, dtype=np.int8)

    with pytest.raises(ValueError, match="map dimensions"):
        render_obstacle_occupancy_map(grid, map_width_m=0.0)


def test_robot_obstacles_build_occupancy_grid_and_panel():
    obstacle_points = np.asarray(
        ((1.0, 1.0), (2.0, -1.0)),
        dtype=np.float32,
    )
    grid = build_obstacle_occupancy_grid(
        obstacle_points,
        map_width_m=6.0,
        map_height_m=3.0,
        grid_size=128,
        obstacle_radius_cells=1,
    )
    map_image = render_obstacle_occupancy_map(
        grid,
        map_width_m=6.0,
        map_height_m=3.0,
    )
    panel = make_obstacle_occupancy_panel(
        source_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        depth_color_bgr=colorize_depth(
            np.ones((24, 32), dtype=np.float32)
        ),
        obstacle_mask=np.full((24, 32), 255, dtype=np.uint8),
        map_bgr=map_image,
        panel_size=128,
    )

    assert grid.shape == (64, 127)
    assert grid.dtype == np.int8
    assert set(np.unique(grid)).issubset({-1, 0, 100})
    assert np.count_nonzero(grid == 100) > 0
    assert np.count_nonzero(grid == 0) > 0
    assert panel.shape == (162, 512, 3)


def test_two_by_one_metre_grid_uses_equal_scale_on_both_axes():
    grid = build_obstacle_occupancy_grid(
        np.empty((0, 2), dtype=np.float32),
        map_width_m=2.0,
        map_height_m=1.0,
        grid_size=512,
    )

    horizontal_pixels_per_metre = (grid.shape[1] - 1) / 2.0
    forward_pixels_per_metre = grid.shape[0] - 1
    assert grid.shape == (256, 511)
    assert horizontal_pixels_per_metre == forward_pixels_per_metre
