from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Box:
    """An upright box in vehicle LiDAR coordinates; yaw is in radians."""

    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    category: int
    object_index: int
    frame: int
    sensor_pose: np.ndarray
    points_inside: np.ndarray | None = field(default=None, init=False, repr=False)
    num_points: int | None = field(default=None, init=False)
    pixel_height: float | None = field(default=None, init=False)
    difficulty: str = field(default="Unclassified", init=False)
    visible_surface_grid: dict | None = field(default=None, init=False, repr=False)
    total_cells: int | None = field(default=None, init=False)
    class_name: str = field(default="Unknown", init=False)
    occupied_cells: dict | None = field(default=None, init=False, repr=False)
    total_occupied_cells: int | None = field(default=None, init=False)
    visibility_score: float | None = field(default=None, init=False)

    def compute_visible_surface_grid(self, sensor_origin=(0, 0, 0), resolution=0.1):
        """Grid sensor-facing faces, without testing occlusion by other objects.

        Sensor origin and cell centers use vehicle LiDAR coordinates, NOT world
        coordinates. Face names (x-/x+/y-/y+/z-/z+) use box-local axes.
        Each face stores its tangent axes, local cell edges, and an (Nu, Nv, 3)
        array of cell centers. Edge cells are clipped to the box dimensions.
        All-zero dimensions mean unavailable (None); no facing faces means {}.
        """
        origin = np.asarray(sensor_origin, dtype=float)
        center = np.array([self.x, self.y, self.z])
        dimensions = np.array([self.length, self.width, self.height])
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("sensor_origin must contain three finite coordinates")
        if not np.isfinite(center).all() or not np.isfinite(self.yaw):
            raise ValueError("Box center and yaw must be finite")
        self.visible_surface_grid = None
        self.total_cells = None
        self.occupied_cells = None
        self.total_occupied_cells = None
        self.visibility_score = None
        if np.all(dimensions == 0):
            return
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("Box dimensions must be finite and positive")

        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        sensor_local = (origin - center) @ rotation
        half = dimensions / 2
        self.visible_surface_grid = {}
        self.total_cells = 0
        for axis, axis_name in enumerate("xyz"):
            # A face is visible only from strictly outside its supporting plane.
            if abs(sensor_local[axis]) <= half[axis] + 1e-9:
                continue
            sign = 1 if sensor_local[axis] > 0 else -1
            tangent_axes = [i for i in range(3) if i != axis]
            edges = []
            for tangent in tangent_axes:
                count = max(1, int(np.ceil(dimensions[tangent] / resolution - 1e-9)))
                edges.append(np.r_[np.arange(count) * resolution,
                                   dimensions[tangent]] - half[tangent])
            u, v = np.meshgrid(*[(e[:-1] + e[1:]) / 2 for e in edges], indexing="ij")
            local = np.empty(u.shape + (3,))
            local[..., axis] = sign * half[axis]
            local[..., tangent_axes[0]] = u
            local[..., tangent_axes[1]] = v
            self.visible_surface_grid[axis_name + ("+" if sign > 0 else "-")] = {
                "tangent_axes": tuple(tangent_axes),
                "edges": tuple(edges),
                "centers": local @ rotation.T + center,
                "total_cells": u.size,
            }
            self.total_cells += u.size

    def compute_surface_occupancy(self, sensor_origin=(0, 0, 0), resolution=0.1):
        """Count unique first-entry cells supported by measured in-box returns.

        Call compute_measurements first. Each return marks at most one cell;
        edge/corner ties choose the first face in x/y/z order. This is sampled
        surface coverage, not a distance-independent measure of occlusion.
        """
        self.compute_visible_surface_grid(sensor_origin, resolution)
        if self.visible_surface_grid is None or self.points_inside is None:
            return
        self.occupied_cells = {
            name: np.zeros(face["centers"].shape[:2], dtype=bool)
            for name, face in self.visible_surface_grid.items()
        }
        self.total_occupied_cells = 0
        if not self.total_cells:
            return

        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = np.array([self.x, self.y, self.z])
        origin = (np.asarray(sensor_origin) - center) @ rotation
        returns = (self.points_inside[:, :3] - center) @ rotation
        directions = returns - origin
        half = np.array([self.length, self.width, self.height]) / 2
        faces = list(self.visible_surface_grid)
        times = np.full((len(returns), len(faces)), -np.inf)
        for index, name in enumerate(faces):
            axis = "xyz".index(name[0])
            plane = half[axis] * (1 if name[1] == "+" else -1)
            np.divide(plane - origin[axis], directions[:, axis],
                      out=times[:, index], where=directions[:, axis] != 0)
        # Endpoints are inside the box: entry occurs after crossing ALL near
        # planes, i.e. at the largest near-plane time, not the smallest.
        entry_face = times.argmax(axis=1)
        entry_time = times.max(axis=1)
        entry = origin + entry_time[:, None] * directions
        for index, name in enumerate(faces):
            face = self.visible_surface_grid[name]
            hits = entry[entry_face == index]
            indices = []
            for axis, edges in zip(face["tangent_axes"], face["edges"]):
                cell = np.searchsorted(edges, hits[:, axis], side="right") - 1
                indices.append(np.clip(cell, 0, len(edges) - 2))
            self.occupied_cells[name][tuple(indices)] = True
        self.total_occupied_cells = sum(int(grid.sum()) for grid in self.occupied_cells.values())
        self.visibility_score = self.total_occupied_cells / self.total_cells

    def compute_measurements(self, points, label):
        """Use local LiDAR points and this object's original DAIR annotation.

        Occlusion states: 0=fully visible, 1=partly occluded, 2=difficult
        to see. Only truncation state 0 is non-truncated. All-zero 3D
        dimensions leave points_inside and num_points unavailable (None);
        image-based difficulty can still be computed.
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape (N, 3+)")
        dimensions = np.array([self.length, self.width, self.height])
        self.points_inside = None
        self.num_points = None
        if not np.all(dimensions == 0):
            if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError("Box dimensions must be finite and positive")

            # Rotate offsets into box axes; retain intensity and other columns.
            offsets = points[:, :3] - [self.x, self.y, self.z]
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            local = offsets @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            inside = np.all(np.abs(local) <= dimensions / 2 + 1e-9, axis=1)
            self.points_inside = points[inside]
            self.num_points = len(self.points_inside)

        self.class_name = label.get("type", "Unknown")
        bounds = label["2d_box"]
        self.pixel_height = float(bounds["ymax"]) - float(bounds["ymin"])
        if not np.isfinite(self.pixel_height) or self.pixel_height < 0:
            raise ValueError("2D box height must be finite and non-negative")
        occlusion = label["occluded_state"]
        truncation = label["truncated_state"]
        self.difficulty = "Unclassified"
        if self.pixel_height >= 40 and occlusion == 0 and truncation == 0:
            self.difficulty = "Easy"
        elif self.pixel_height >= 25 and occlusion in (0, 1):
            self.difficulty = "Moderate"
        elif self.pixel_height >= 25 and occlusion == 2:
            self.difficulty = "Hard"


