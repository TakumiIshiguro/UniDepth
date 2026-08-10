import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image

from unidepth_ros.device import resolve_device


class DepthResultWriter:
    def __init__(
        self,
        output_dir: str,
        save_raw_depth: bool = True,
        save_color: bool = True,
        save_obstacle_mask: bool = False,
        save_obstacle_points_2d: bool = False,
        save_occupancy_grid: bool = False,
        save_map: bool = False,
        save_panel: bool = False,
        session_name: str = None,
    ):
        self.save_raw_depth = bool(save_raw_depth)
        self.save_color = bool(save_color)
        self.save_obstacle_mask = bool(save_obstacle_mask)
        self.save_obstacle_points_2d = bool(save_obstacle_points_2d)
        self.save_occupancy_grid = bool(save_occupancy_grid)
        self.save_map = bool(save_map)
        self.save_panel = bool(save_panel)
        if not any(
            (
                self.save_raw_depth,
                self.save_color,
                self.save_obstacle_mask,
                self.save_obstacle_points_2d,
                self.save_occupancy_grid,
                self.save_map,
                self.save_panel,
            )
        ):
            raise ValueError(
                "at least one result format must be enabled"
            )
        if session_name is None:
            session_name = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        self.session_dir = os.path.abspath(
            os.path.join(str(output_dir), str(session_name))
        )
        self.raw_dir = os.path.join(self.session_dir, "depth_meters")
        self.color_dir = os.path.join(self.session_dir, "color")
        self.obstacle_mask_dir = os.path.join(
            self.session_dir,
            "obstacle_mask",
        )
        self.obstacle_points_dir = os.path.join(
            self.session_dir,
            "obstacle_points_2d",
        )
        self.occupancy_dir = os.path.join(
            self.session_dir,
            "occupancy_grid",
        )
        self.map_dir = os.path.join(self.session_dir, "map")
        self.panel_dir = os.path.join(self.session_dir, "visualization")
        os.makedirs(self.session_dir, exist_ok=False)
        if self.save_raw_depth:
            os.makedirs(self.raw_dir)
        if self.save_color:
            os.makedirs(self.color_dir)
        if self.save_obstacle_mask:
            os.makedirs(self.obstacle_mask_dir)
        if self.save_obstacle_points_2d:
            os.makedirs(self.obstacle_points_dir)
        if self.save_occupancy_grid:
            os.makedirs(self.occupancy_dir)
        if self.save_map:
            os.makedirs(self.map_dir)
        if self.save_panel:
            os.makedirs(self.panel_dir)

        self.metadata_path = os.path.join(
            self.session_dir,
            "metadata.csv",
        )
        with open(self.metadata_path, "w", newline="") as stream:
            csv.writer(stream).writerow(
                (
                    "index",
                    "stamp",
                    "height",
                    "width",
                    "minimum",
                    "maximum",
                    "raw_depth_path",
                    "color_path",
                    "obstacle_mask_path",
                    "obstacle_points_2d_path",
                    "occupancy_grid_path",
                    "map_path",
                    "visualization_path",
                )
            )
        self._next_index = 0

    def save(
        self,
        depth_meters: np.ndarray,
        color: np.ndarray = None,
        obstacle_mask: np.ndarray = None,
        obstacle_points_2d: np.ndarray = None,
        occupancy_grid: np.ndarray = None,
        map_image: np.ndarray = None,
        panel: np.ndarray = None,
        stamp: float = None,
    ) -> Dict[str, str]:
        depth_meters = np.asarray(depth_meters, dtype=np.float32)
        if depth_meters.ndim != 2:
            raise ValueError(
                "depth_meters must be a two-dimensional array"
            )
        if not np.isfinite(depth_meters).all():
            raise ValueError("depth_meters must contain only finite values")
        if np.any(depth_meters < 0.0):
            raise ValueError("depth_meters must be non-negative")
        if self.save_color:
            if color is None:
                raise ValueError("color is required when save_color is enabled")
            color = np.asarray(color)
            if (
                color.shape
                != (depth_meters.shape[0], depth_meters.shape[1], 3)
                or color.dtype != np.uint8
            ):
                raise ValueError(
                    "color must be a uint8 BGR array matching depth size"
                )
        if self.save_obstacle_mask:
            obstacle_mask = np.asarray(obstacle_mask)
            if (
                obstacle_mask.shape != depth_meters.shape
                or obstacle_mask.dtype != np.uint8
            ):
                raise ValueError(
                    "obstacle_mask must be a uint8 array matching depth size"
                )
        if self.save_obstacle_points_2d:
            obstacle_points_2d = np.asarray(
                obstacle_points_2d,
                dtype=np.float32,
            )
            if (
                obstacle_points_2d.ndim != 2
                or obstacle_points_2d.shape[1] != 2
                or not np.isfinite(obstacle_points_2d).all()
            ):
                raise ValueError(
                    "obstacle_points_2d must be finite with shape (N, 2)"
                )
        if self.save_occupancy_grid:
            occupancy_grid = np.asarray(occupancy_grid)
            if (
                occupancy_grid.ndim != 2
                or occupancy_grid.dtype != np.int8
                or not np.isin(
                    occupancy_grid,
                    (-1, 0, 100),
                ).all()
            ):
                raise ValueError(
                    "occupancy_grid must be int8 with values -1, 0, 100"
                )
        for name, enabled, image in (
            ("map_image", self.save_map, map_image),
            ("panel", self.save_panel, panel),
        ):
            if enabled and (
                image is None
                or np.asarray(image).ndim != 3
                or np.asarray(image).shape[2] != 3
                or np.asarray(image).dtype != np.uint8
            ):
                raise ValueError(f"{name} must be a uint8 BGR image")

        index = self._next_index
        stem = f"{index:06d}"
        raw_relative_path = ""
        color_relative_path = ""
        obstacle_mask_relative_path = ""
        obstacle_points_relative_path = ""
        occupancy_relative_path = ""
        map_relative_path = ""
        panel_relative_path = ""
        if self.save_raw_depth:
            raw_relative_path = os.path.join(
                "depth_meters",
                f"{stem}.npy",
            )
            np.save(
                os.path.join(self.session_dir, raw_relative_path),
                depth_meters,
                allow_pickle=False,
            )
        if self.save_color:
            color_relative_path = os.path.join("color", f"{stem}.png")
            if not cv2.imwrite(
                os.path.join(self.session_dir, color_relative_path),
                color,
            ):
                raise OSError("failed to save depth color image")
        if self.save_obstacle_mask:
            obstacle_mask_relative_path = os.path.join(
                "obstacle_mask",
                f"{stem}.png",
            )
            if not cv2.imwrite(
                os.path.join(
                    self.session_dir,
                    obstacle_mask_relative_path,
                ),
                obstacle_mask,
            ):
                raise OSError("failed to save obstacle mask")
        if self.save_obstacle_points_2d:
            obstacle_points_relative_path = os.path.join(
                "obstacle_points_2d",
                f"{stem}.npy",
            )
            np.save(
                os.path.join(
                    self.session_dir,
                    obstacle_points_relative_path,
                ),
                obstacle_points_2d,
                allow_pickle=False,
            )
        if self.save_occupancy_grid:
            occupancy_relative_path = os.path.join(
                "occupancy_grid",
                f"{stem}.npy",
            )
            np.save(
                os.path.join(
                    self.session_dir,
                    occupancy_relative_path,
                ),
                occupancy_grid,
                allow_pickle=False,
            )
        for enabled, directory, suffix, image in (
            (self.save_map, "map", "map", map_image),
            (
                self.save_panel,
                "visualization",
                "visualization",
                panel,
            ),
        ):
            if not enabled:
                continue
            relative_path = os.path.join(
                directory,
                f"{stem}_{suffix}.png",
            )
            if not cv2.imwrite(
                os.path.join(self.session_dir, relative_path),
                image,
            ):
                raise OSError(f"failed to save {suffix} image")
            if suffix == "map":
                map_relative_path = relative_path
            else:
                panel_relative_path = relative_path

        with open(self.metadata_path, "a", newline="") as stream:
            csv.writer(stream).writerow(
                (
                    index,
                    "" if stamp is None else f"{float(stamp):.9f}",
                    depth_meters.shape[0],
                    depth_meters.shape[1],
                    f"{float(depth_meters.min()):.9g}",
                    f"{float(depth_meters.max()):.9g}",
                    raw_relative_path,
                    color_relative_path,
                    obstacle_mask_relative_path,
                    obstacle_points_relative_path,
                    occupancy_relative_path,
                    map_relative_path,
                    panel_relative_path,
                )
            )
        self._next_index += 1
        return {
            "raw_depth": (
                os.path.join(self.session_dir, raw_relative_path)
                if raw_relative_path
                else ""
            ),
            "color": (
                os.path.join(self.session_dir, color_relative_path)
                if color_relative_path
                else ""
            ),
            "obstacle_mask": (
                os.path.join(
                    self.session_dir,
                    obstacle_mask_relative_path,
                )
                if obstacle_mask_relative_path
                else ""
            ),
            "obstacle_points_2d": (
                os.path.join(
                    self.session_dir,
                    obstacle_points_relative_path,
                )
                if obstacle_points_relative_path
                else ""
            ),
            "occupancy_grid": (
                os.path.join(
                    self.session_dir,
                    occupancy_relative_path,
                )
                if occupancy_relative_path
                else ""
            ),
            "map": (
                os.path.join(self.session_dir, map_relative_path)
                if map_relative_path
                else ""
            ),
            "visualization": (
                os.path.join(self.session_dir, panel_relative_path)
                if panel_relative_path
                else ""
            ),
        }


def _load_unidepth_model(repository_path: str, model_path: str):
    repository_path = os.path.abspath(os.path.expanduser(repository_path))
    package_path = os.path.join(repository_path, "unidepth")
    if not os.path.isdir(package_path):
        raise FileNotFoundError(
            "UniDepth source package was not found at {}. Clone the "
            "python38-compat branch before starting the depth node.".format(
                repository_path
            )
        )
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    try:
        from unidepth.models import UniDepthV2
    except ImportError as error:
        raise ImportError(
            "UniDepthV2 could not be imported from {}. Install the Python "
            "3.8 inference dependencies described in UniDepth/PYTHON38.md."
            .format(repository_path)
        ) from error
    return UniDepthV2.from_pretrained(model_path)


def _load_unidepth_fisheye_camera(repository_path: str):
    repository_path = os.path.abspath(repository_path)
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    try:
        from unidepth.utils.camera import Fisheye624
    except ImportError as error:
        raise ImportError(
            "UniDepth Fisheye624 could not be imported from {}".format(
                repository_path
            )
        ) from error
    return Fisheye624


def gazebo_equidistant_camera_parameters(
    image_width: int,
    image_height: int,
    horizontal_fov_rad: float,
):
    """Build UniDepth Fisheye624 parameters for Gazebo's equidistant lens."""
    image_width = int(image_width)
    image_height = int(image_height)
    horizontal_fov_rad = float(horizontal_fov_rad)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < horizontal_fov_rad < np.pi:
        raise ValueError("horizontal_fov_rad must be in (0, pi)")

    # Gazebo's scale_to_hfov=true scales the equidistant projection r=f*theta
    # so that the requested horizontal field of view spans the image width.
    focal_length = image_width / horizontal_fov_rad
    centre_x = (image_width - 1.0) / 2.0
    centre_y = (image_height - 1.0) / 2.0
    parameters = np.zeros(16, dtype=np.float32)
    parameters[:4] = (
        focal_length,
        focal_length,
        centre_x,
        centre_y,
    )
    intrinsics = np.asarray(
        (
            (focal_length, 0.0, centre_x),
            (0.0, focal_length, centre_y),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return parameters, intrinsics


def colorize_depth(
    depth: np.ndarray,
    minimum_percentile: float = 2.0,
    maximum_percentile: float = 98.0,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    if not 0.0 <= minimum_percentile < maximum_percentile <= 100.0:
        raise ValueError(
            "percentiles must satisfy 0 <= minimum < maximum <= 100"
        )

    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("depth does not contain a finite value")
    finite_values = depth[finite]
    lower, upper = np.percentile(
        finite_values,
        [minimum_percentile, maximum_percentile],
    )
    scale = float(upper - lower)
    if scale <= np.finfo(np.float32).eps:
        normalized = np.zeros(depth.shape, dtype=np.float32)
    else:
        normalized = np.clip((depth - lower) / scale, 0.0, 1.0)
    normalized[~finite] = 0.0
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~finite] = 0
    return color


def filter_metric_point_cloud(
    point_cloud: np.ndarray,
    depth: np.ndarray,
    minimum_depth_m: float = 0.1,
    maximum_depth_m: float = 10.0,
):
    """Apply metric range limits to an already reconstructed point cloud."""
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
        raise ValueError("point_cloud must have shape (height, width, 3)")
    if depth.shape != point_cloud.shape[:2]:
        raise ValueError("depth must match the point cloud image dimensions")
    if not np.isfinite(point_cloud).all() or not np.isfinite(depth).all():
        raise ValueError("depth and point_cloud must contain only finite values")
    if minimum_depth_m < 0.0:
        raise ValueError("minimum_depth_m must be non-negative")
    if maximum_depth_m <= minimum_depth_m:
        raise ValueError("maximum_depth_m must exceed minimum_depth_m")

    valid = (
        (depth >= minimum_depth_m)
        & (depth <= maximum_depth_m)
        & (point_cloud[..., 2] > 0.0)
    )
    filtered = np.ascontiguousarray(point_cloud.copy(), dtype=np.float32)
    filtered[~valid] = 0.0
    return filtered, valid


def transform_point_cloud(
    point_cloud: np.ndarray,
    translation_xyz,
    quaternion_xyzw,
) -> np.ndarray:
    """Transform a dense point cloud with a target-from-source pose."""
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
        raise ValueError("point_cloud must have shape (height, width, 3)")
    if not np.isfinite(point_cloud).all():
        raise ValueError("point_cloud must contain only finite values")
    translation = np.asarray(translation_xyz, dtype=np.float32)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float32)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation_xyz must contain three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("quaternion_xyzw must have non-zero norm")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float32,
    )
    transformed = np.einsum(
        "ij,...j->...i",
        rotation,
        point_cloud,
    )
    return np.ascontiguousarray(transformed + translation, dtype=np.float32)


def extract_nearest_obstacle_points(
    robot_point_cloud: np.ndarray,
    valid_depth_mask: np.ndarray,
    minimum_forward_m: float,
    maximum_forward_m: float,
    minimum_height_m: float,
    maximum_height_m: float,
    map_width_m: float,
    lateral_bins: int,
    border_margin_ratio: float = 0.02,
    return_all_points: bool = False,
):
    """Select the minimum-forward robot-frame point in each lateral bin.

    Robot coordinates are x forward, y left, z up. Returned obstacle points
    have shape (N, 2) and use the same (forward, left) order. When requested,
    the filtered mask and all filtered points are appended to the return tuple
    for visualization; bin-selected points remain the primary result.
    """
    robot_point_cloud = np.asarray(robot_point_cloud, dtype=np.float32)
    valid_depth_mask = np.asarray(valid_depth_mask, dtype=bool)
    if robot_point_cloud.ndim != 3 or robot_point_cloud.shape[2] != 3:
        raise ValueError(
            "robot_point_cloud must have shape (height, width, 3)"
        )
    if valid_depth_mask.shape != robot_point_cloud.shape[:2]:
        raise ValueError("valid_depth_mask must match robot_point_cloud")
    if not np.isfinite(robot_point_cloud).all():
        raise ValueError("robot_point_cloud must contain only finite values")
    if minimum_forward_m < 0.0 or maximum_forward_m <= minimum_forward_m:
        raise ValueError("forward limits must satisfy 0 <= minimum < maximum")
    if maximum_height_m <= minimum_height_m:
        raise ValueError("maximum_height_m must exceed minimum_height_m")
    if map_width_m <= 0.0:
        raise ValueError("map_width_m must be positive")
    if lateral_bins <= 0:
        raise ValueError("lateral_bins must be positive")
    if not 0.0 <= border_margin_ratio < 0.5:
        raise ValueError("border_margin_ratio must be in [0, 0.5)")

    forward = robot_point_cloud[..., 0]
    left = robot_point_cloud[..., 1]
    height_values = robot_point_cloud[..., 2]
    lateral_limit = map_width_m / 2.0
    candidate_mask = (
        valid_depth_mask
        & (forward >= minimum_forward_m)
        & (forward <= maximum_forward_m)
        & (left >= -lateral_limit)
        & (left <= lateral_limit)
        & (height_values >= minimum_height_m)
        & (height_values <= maximum_height_m)
    )

    height, width = candidate_mask.shape
    margin_y = int(round(height * border_margin_ratio))
    margin_x = int(round(width * border_margin_ratio))
    if margin_y > 0:
        candidate_mask[:margin_y] = False
        candidate_mask[height - margin_y :] = False
    if margin_x > 0:
        candidate_mask[:, :margin_x] = False
        candidate_mask[:, width - margin_x :] = False

    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        selected_result = (
            candidate_mask.astype(np.uint8) * 255,
            np.empty((0, 2), dtype=np.float32),
        )
        if return_all_points:
            return selected_result + selected_result
        return selected_result
    candidate_forward = forward.ravel()[candidate_indices]
    candidate_left = left.ravel()[candidate_indices]
    all_obstacle_points = np.column_stack(
        (candidate_forward, candidate_left)
    ).astype(np.float32, copy=False)
    normalized = (candidate_left + lateral_limit) / map_width_m
    bin_indices = np.minimum(
        np.floor(normalized * lateral_bins).astype(np.int32),
        lateral_bins - 1,
    )
    selected_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
    obstacle_points = []
    for bin_index in np.unique(bin_indices):
        members = np.flatnonzero(bin_indices == bin_index)
        nearest = members[np.argmin(candidate_forward[members])]
        selected_mask.ravel()[candidate_indices[nearest]] = 255
        obstacle_points.append(
            (candidate_forward[nearest], candidate_left[nearest])
        )
    selected_result = (
        selected_mask,
        np.asarray(obstacle_points, dtype=np.float32),
    )
    if return_all_points:
        return selected_result + (
            candidate_mask.astype(np.uint8) * 255,
            all_obstacle_points,
        )
    return selected_result


def build_obstacle_occupancy_grid(
    obstacle_points_2d: np.ndarray,
    map_width_m: float = 10.0,
    map_height_m: float = 10.0,
    grid_size: int = 512,
    obstacle_radius_cells: int = 2,
) -> np.ndarray:
    """Create a ROS-style grid: unknown=-1, free=0, occupied=100."""
    obstacle_points_2d = np.asarray(obstacle_points_2d, dtype=np.float32)
    if obstacle_points_2d.ndim != 2 or obstacle_points_2d.shape[1] != 2:
        raise ValueError("obstacle_points_2d must have shape (N, 2)")
    if map_width_m <= 0.0 or map_height_m <= 0.0:
        raise ValueError("map dimensions must be positive")
    if grid_size < 128:
        raise ValueError("grid_size must be at least 128")
    if obstacle_radius_cells < 0:
        raise ValueError("obstacle_radius_cells must be non-negative")

    pixels_per_metre = int(
        np.floor((grid_size - 1) / max(map_width_m, map_height_m))
    )
    if pixels_per_metre < 1:
        raise ValueError("grid_size is too small for the configured map extent")
    grid_width = int(round(map_width_m * pixels_per_metre)) + 1
    grid_height = int(round(map_height_m * pixels_per_metre)) + 1
    grid = np.full((grid_height, grid_width), -1, dtype=np.int8)
    if obstacle_points_2d.shape[0] == 0:
        return grid
    origin = (grid_width // 2, grid_height - 1)
    horizontal_scale = float(pixels_per_metre)
    forward_scale = float(pixels_per_metre)
    # Robot coordinates are (forward, left); screen left is negative pixels.
    pixel_x = np.rint(
        origin[0] - obstacle_points_2d[:, 1] * horizontal_scale
    ).astype(np.int32)
    pixel_y = np.rint(
        origin[1] - obstacle_points_2d[:, 0] * forward_scale
    ).astype(np.int32)
    valid = (
        (pixel_x >= 0)
        & (pixel_x < grid_width)
        & (pixel_y >= 0)
        & (pixel_y < grid_height)
    )
    pixel_x = pixel_x[valid]
    pixel_y = pixel_y[valid]

    free_mask = np.zeros(grid.shape, dtype=np.uint8)
    occupied_mask = np.zeros(grid.shape, dtype=np.uint8)
    for x, y in zip(pixel_x, pixel_y):
        cv2.line(
            free_mask,
            origin,
            (int(x), int(y)),
            255,
            1,
            cv2.LINE_8,
        )
        cv2.circle(
            occupied_mask,
            (int(x), int(y)),
            int(obstacle_radius_cells),
            255,
            -1,
            cv2.LINE_8,
        )
    grid[free_mask > 0] = 0
    grid[occupied_mask > 0] = 100
    return grid


def render_obstacle_occupancy_map(
    occupancy_grid: np.ndarray,
    map_width_m: float = 10.0,
    map_height_m: float = 10.0,
) -> np.ndarray:
    occupancy_grid = np.asarray(occupancy_grid)
    if occupancy_grid.ndim != 2:
        raise ValueError("occupancy_grid must be two-dimensional")
    if map_width_m <= 0.0 or map_height_m <= 0.0:
        raise ValueError("map dimensions must be positive")
    image = np.empty(
        (*occupancy_grid.shape, 3),
        dtype=np.uint8,
    )
    image[occupancy_grid == -1] = (170, 170, 170)
    image[occupancy_grid == 0] = (250, 250, 250)
    image[occupancy_grid == 100] = (20, 20, 20)
    invalid = ~np.isin(occupancy_grid, (-1, 0, 100))
    image[invalid] = (0, 0, 255)

    height, width = occupancy_grid.shape
    origin = (width // 2, height - 1)
    horizontal_scale = (width - 1) / float(map_width_m)
    forward_scale = (height - 1) / float(map_height_m)
    grid_color = (115, 115, 115)
    axis_color = (30, 100, 220)
    text_color = (35, 35, 35)

    def nice_tick_step(span_m: float) -> float:
        raw_step = span_m / 5.0
        magnitude = 10.0 ** np.floor(np.log10(raw_step))
        normalized = raw_step / magnitude
        if normalized <= 1.0:
            factor = 1.0
        elif normalized <= 2.0:
            factor = 2.0
        elif normalized <= 5.0:
            factor = 5.0
        else:
            factor = 10.0
        return float(factor * magnitude)

    lateral_limit = map_width_m / 2.0
    lateral_step = nice_tick_step(map_width_m)
    left_ticks = np.arange(
        -np.floor(lateral_limit / lateral_step) * lateral_step,
        lateral_limit + lateral_step * 0.5,
        lateral_step,
    )
    for left_m in left_ticks:
        pixel_x = int(round(origin[0] - left_m * horizontal_scale))
        if not 0 < pixel_x < width - 1:
            continue
        color = axis_color if np.isclose(left_m, 0.0) else grid_color
        thickness = 2 if np.isclose(left_m, 0.0) else 1
        cv2.line(
            image,
            (pixel_x, 0),
            (pixel_x, height - 1),
            color,
            thickness,
            cv2.LINE_AA,
        )
        if not np.isclose(left_m, 0.0):
            label = f"{left_m:g}"
            label_width = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                1,
            )[0][0]
            cv2.putText(
                image,
                label,
                (pixel_x - label_width // 2, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                text_color,
                1,
                cv2.LINE_AA,
            )

    forward_step = nice_tick_step(map_height_m)
    forward_ticks = np.arange(
        forward_step,
        map_height_m,
        forward_step,
    )
    for forward_m in forward_ticks:
        pixel_y = int(round(origin[1] - forward_m * forward_scale))
        if not 0 < pixel_y < height - 1:
            continue
        cv2.line(
            image,
            (0, pixel_y),
            (width - 1, pixel_y),
            grid_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{forward_m:g}",
            (origin[0] + 6, pixel_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Keep occupied cells readable when a metric grid line crosses them.
    image[occupancy_grid == 100] = (20, 20, 20)
    cv2.arrowedLine(
        image,
        origin,
        (origin[0], 5),
        axis_color,
        2,
        cv2.LINE_AA,
        tipLength=0.025,
    )
    cv2.arrowedLine(
        image,
        origin,
        (5, origin[1]),
        axis_color,
        2,
        cv2.LINE_AA,
        tipLength=0.025,
    )
    robot = np.array(
        [
            (origin[0], origin[1] - 16),
            (origin[0] - 9, origin[1] - 1),
            (origin[0] + 9, origin[1] - 1),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, robot, (190, 80, 30), cv2.LINE_AA)
    cv2.putText(
        image,
        f"near obstacle map ({map_width_m:g} x {map_height_m:g} m)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "x [m] forward",
        (origin[0] + 9, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        axis_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "y [m] left +",
        (6, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        axis_color,
        1,
        cv2.LINE_AA,
    )
    return image


def make_obstacle_occupancy_panel(
    source_rgb: np.ndarray,
    depth_color_bgr: np.ndarray,
    obstacle_mask: np.ndarray,
    map_bgr: np.ndarray,
    panel_size: int = 256,
) -> np.ndarray:
    def fit_square(image):
        height, width = image.shape[:2]
        scale = min(panel_size / width, panel_size / height)
        resized = cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        square = np.zeros(
            (panel_size, panel_size, 3),
            dtype=np.uint8,
        )
        y = (panel_size - resized.shape[0]) // 2
        x = (panel_size - resized.shape[1]) // 2
        square[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return square

    source_rgb = np.asarray(source_rgb)
    depth_color_bgr = np.asarray(depth_color_bgr)
    obstacle_mask = np.asarray(obstacle_mask)
    map_bgr = np.asarray(map_bgr)
    if (
        source_rgb.ndim != 3
        or source_rgb.shape[2] != 3
        or source_rgb.dtype != np.uint8
    ):
        raise ValueError("source_rgb must be a uint8 RGB image")
    if obstacle_mask.shape != source_rgb.shape[:2]:
        raise ValueError("obstacle_mask must match source image size")

    source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    obstacle_overlay = source_bgr.copy()
    display_mask = cv2.dilate(
        obstacle_mask,
        np.ones((5, 5), dtype=np.uint8),
    )
    obstacle_overlay[display_mask == 0] = (
        obstacle_overlay[display_mask == 0] * 0.18
    ).astype(np.uint8)
    images = (
        fit_square(source_bgr),
        fit_square(depth_color_bgr),
        fit_square(obstacle_overlay),
        fit_square(map_bgr),
    )
    labels = (
        "input",
        "metric depth (m)",
        "all filtered obstacle points",
        "all-points robot-frame BEV",
    )
    header_height = 34
    panel = np.full(
        (panel_size + header_height, panel_size * 4, 3),
        20,
        dtype=np.uint8,
    )
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * panel_size
        panel[header_height:, x : x + panel_size] = image
        cv2.putText(
            panel,
            label,
            (x + 8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return panel


@dataclass(frozen=True)
class UniDepthPrediction:
    depth_meters: np.ndarray
    points_camera: np.ndarray
    intrinsics: np.ndarray


class UniDepthV2Small:
    """Metric depth and camera-frame geometry from UniDepthV2 ViT-S."""

    def __init__(
        self,
        depth_config: Dict,
        repository_path: str,
        model_path: str,
        model=None,
        camera_class=None,
    ):
        self.resolution_level = int(depth_config.get("resolution_level", 0))
        if not 0 <= self.resolution_level < 10:
            raise ValueError("resolution_level must be in [0, 10)")
        self.depth_scale = float(depth_config.get("depth_scale", 1.0))
        if self.depth_scale <= 0.0:
            raise ValueError("depth_scale must be positive")
        self.device = resolve_device(depth_config.get("device", "auto"))
        self.camera_model = str(
            depth_config.get("camera_model", "estimated_pinhole")
        )
        if self.camera_model not in (
            "estimated_pinhole",
            "gazebo_equidistant",
        ):
            raise ValueError("unsupported camera_model: {}".format(
                self.camera_model
            ))
        self.horizontal_fov_rad = float(
            depth_config.get("horizontal_fov_rad", 2.62)
        )
        if not 0.0 < self.horizontal_fov_rad < np.pi:
            raise ValueError("horizontal_fov_rad must be in (0, pi)")

        if model is None:
            model = _load_unidepth_model(repository_path, model_path)
        if self.camera_model == "gazebo_equidistant":
            if camera_class is None:
                camera_class = _load_unidepth_fisheye_camera(repository_path)
            self.camera_class = camera_class
        else:
            self.camera_class = None
        self.model = model
        self.model.resolution_level = self.resolution_level
        self.model.to(self.device)
        self.model.eval()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def predict(self, image: Image.Image) -> UniDepthPrediction:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        rgb_image = image.convert("RGB")
        original_width, original_height = rgb_image.size
        rgb = torch.from_numpy(
            np.asarray(rgb_image, dtype=np.uint8).copy()
        ).permute(2, 0, 1)

        configured_intrinsics = None
        with torch.inference_mode():
            if self.camera_model == "gazebo_equidistant":
                parameters, configured_intrinsics = (
                    gazebo_equidistant_camera_parameters(
                        original_width,
                        original_height,
                        self.horizontal_fov_rad,
                    )
                )
                camera = self.camera_class(
                    torch.from_numpy(parameters).unsqueeze(0)
                )
                output = self.model.infer(rgb, camera=camera)
            else:
                output = self.model.infer(rgb)

        required = ("depth", "points", "intrinsics")
        missing = [name for name in required if name not in output]
        if missing:
            raise RuntimeError(
                "UniDepth output is missing: {}".format(", ".join(missing))
            )
        depth_tensor = output["depth"]
        points_tensor = output["points"]
        intrinsics_tensor = output["intrinsics"]
        expected_depth_shape = (1, 1, original_height, original_width)
        expected_points_shape = (1, 3, original_height, original_width)
        if tuple(depth_tensor.shape) != expected_depth_shape:
            raise RuntimeError(
                "UniDepth depth must have shape {}, got {}".format(
                    expected_depth_shape,
                    tuple(depth_tensor.shape),
                )
            )
        if tuple(points_tensor.shape) != expected_points_shape:
            raise RuntimeError(
                "UniDepth points must have shape {}, got {}".format(
                    expected_points_shape,
                    tuple(points_tensor.shape),
                )
            )
        if tuple(intrinsics_tensor.shape) != (1, 3, 3):
            raise RuntimeError(
                "UniDepth intrinsics must have shape (1, 3, 3), got {}".format(
                    tuple(intrinsics_tensor.shape)
                )
            )

        depth = np.ascontiguousarray(
            depth_tensor[0, 0].float().detach().cpu().numpy()
            * self.depth_scale,
            dtype=np.float32,
        )
        points = np.ascontiguousarray(
            points_tensor[0].permute(1, 2, 0).float().detach().cpu().numpy()
            * self.depth_scale,
            dtype=np.float32,
        )
        if configured_intrinsics is None:
            intrinsics = np.ascontiguousarray(
                intrinsics_tensor[0].float().detach().cpu().numpy(),
                dtype=np.float32,
            )
        else:
            intrinsics = np.ascontiguousarray(
                configured_intrinsics,
                dtype=np.float32,
            )
        if not (
            np.isfinite(depth).all()
            and np.isfinite(points).all()
            and np.isfinite(intrinsics).all()
        ):
            raise RuntimeError("UniDepth produced non-finite geometry")
        if np.any(depth < 0.0):
            raise RuntimeError("UniDepth produced negative metric depth")
        if not np.allclose(points[..., 2], depth, rtol=1e-3, atol=1e-4):
            raise RuntimeError(
                "UniDepth point-cloud z coordinates do not match depth"
            )
        return UniDepthPrediction(depth, points, intrinsics)
