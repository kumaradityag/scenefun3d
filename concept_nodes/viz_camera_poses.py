from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import open3d as o3d
import tyro

import viser
import viser.transforms as vtf

from concept_nodes.utils import get_scene_intrinsics, get_scene_trajectories


def load_map(
    map_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Dict[int, List[int]],
    Dict[int, List[np.ndarray]],
    Dict[int, str],
]:
    pcd = o3d.io.read_point_cloud(str(map_dir / "point_cloud.pcd"))
    points = np.asarray(pcd.points)
    colours = np.asarray(pcd.colors)

    with open(map_dir / "segments_anno.json") as f:
        anno = json.load(f)

    point_idxs_per_obj, cam_poses_per_obj, labels_per_obj = {}, {}, {}
    for obj in anno["segGroups"]:
        obj_id = obj["id"]
        point_idxs_per_obj[obj_id] = obj["segments"]
        cam_poses_per_obj[obj_id] = [np.asarray(p) for p in obj["camera_poses"]]
        labels_per_obj[obj_id] = obj["label"]

    return points, colours, point_idxs_per_obj, cam_poses_per_obj, labels_per_obj


def build_object_clouds(
    points: np.ndarray,
    colours: np.ndarray,
    point_idxs_per_obj: Dict[int, List[int]],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    clouds = {}
    for obj_id, idxs in point_idxs_per_obj.items():
        idxs_np = np.asarray(idxs, dtype=int)
        clouds[obj_id] = (points[idxs_np], colours[idxs_np])
    return clouds


# -----------------------------------------------------------------------------
# Visualiser
# -----------------------------------------------------------------------------
class ObjectMapViewer:
    def __init__(
        self,
        map_dir: Path,
        map_trajectories: List[List[np.ndarray]],
        fov: float = 60.0,
        aspect: float = 1.0,
        frustum_scale: float = 0.15,
        point_size: float = 0.02,
    ):
        self.fov, self.aspect = fov, aspect
        self.frustum_scale, self.point_size = frustum_scale, point_size

        (
            self.points,
            self.colours,
            self.point_idxs_per_obj,
            self.cam_poses_per_obj,
            self.labels_per_obj,
        ) = load_map(map_dir)

        # Print available objects with their labels.
        print("Available objects:")
        for obj_id, label in self.labels_per_obj.items():
            print(f"ID: {obj_id}, Label: {label}")

        # ----------------------- store trajectories ----------------------- #
        self.map_trajectories = map_trajectories
        self.trajectory_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        self.trajectories_visible = False

        # -------------------------- shift the scene to origin ------------------------- #
        self.scene_shift = self.points.mean(axis=0)

        # shift points
        self.points -= self.scene_shift

        # shift cameras
        for obj_id, idxs in self.cam_poses_per_obj.items():
            for k in range(len(idxs)):
                idxs[k] = idxs[k].copy()
                idxs[k][:3, 3] -= self.scene_shift

        # shift trajectories
        for i, traj in enumerate(self.map_trajectories):
            for j in range(len(traj)):
                traj[j] = traj[j].copy()
                traj[j][:3, 3] -= self.scene_shift

        # ---------------------- build object clouds --------------------- #
        self.object_clouds = build_object_clouds(
            self.points, self.colours, self.point_idxs_per_obj
        )

        self.server = viser.ViserServer()
        self.server.gui.configure_theme(
            titlebar_content=None, control_layout="collapsible"
        )

        # -------------------------- scene objects --------------------------- #
        self.main_cloud = self.server.scene.add_point_cloud(
            "/scene/full_pcd",
            points=self.points,
            colors=self.colours,
            point_size=self.point_size,
        )
        self.obj_clouds: Dict[int, viser.PointCloudHandle] = {}
        self.active_frustums: List[viser.CameraFrustumHandle] = []

        # ----------------------------- GUI ---------------------------------- #
        with self.server.gui.add_folder("Object selection"):
            self.gui_obj_id = self.server.gui.add_number(
                label="object_id",
                min=min(self.point_idxs_per_obj),
                max=max(self.point_idxs_per_obj),
                step=1,
                initial_value=min(self.point_idxs_per_obj),
            )
            gui_show_obj = self.server.gui.add_button("Show object")
            gui_show_obj.on_click(self._on_show_object)

        gui_show_all = self.server.gui.add_button("Show all objects")
        gui_show_all.on_click(self._show_full_scene)

        with self.server.gui.add_folder("Camera Trajectories"):
            self.btn_toggle_traj = self.server.gui.add_button("Toggle trajectories")
            self.btn_toggle_traj.on_click(self._toggle_trajectories)

    # ----------------------------- callbacks -------------------------------- #
    def _on_show_object(self, _evt: viser.GuiEvent) -> None:
        obj_id = int(round(self.gui_obj_id.value))
        if obj_id not in self.object_clouds:
            print(f"Object {obj_id} does not exist.")
            return
        self._show_single_object(obj_id)

    def _show_single_object(self, obj_id: int) -> None:
        with self.server.atomic():
            # hide main scene pcd
            self.main_cloud.visible = False
            # hide other obj pcds, show selected one
            for other_id, handle in self.obj_clouds.items():
                handle.visible = other_id == obj_id
            if obj_id not in self.obj_clouds:
                pts, colors = self.object_clouds[obj_id]
                self.obj_clouds[obj_id] = self.server.scene.add_point_cloud(
                    f"/scene/object_{obj_id}",
                    points=pts,
                    colors=colors,
                    point_size=self.point_size,
                )
            self._clear_frustums()
            self._draw_frustums(obj_id)

    def _show_full_scene(self, _evt: viser.GuiEvent) -> None:
        with self.server.atomic():
            self.main_cloud.visible = True
            for h in self.obj_clouds.values():
                h.visible = False
            self._clear_frustums()

    def _toggle_trajectories(self, _evt: viser.GuiEvent) -> None:
        if not self.trajectories_visible:
            if not self.trajectory_frustums:
                self._draw_trajectories()
            for fr_list in self.trajectory_frustums.values():
                for fr in fr_list:
                    fr.visible = True
            self.trajectories_visible = True
        else:
            for fr_list in self.trajectory_frustums.values():
                for fr in fr_list:
                    fr.visible = False
            self.trajectories_visible = False

    # ----------------------------- frustums --------------------------------- #
    def _draw_frustums(self, obj_id: int) -> None:
        poses = self.cam_poses_per_obj.get(obj_id, [])
        if not poses:
            return

        cmap = matplotlib.colormaps["viridis"].reversed()
        n = len(poses)

        for i, cam2world in enumerate(poses):
            color = cmap(i / max(1, n - 1))[:3]

            # cam2world is already a 4×4 SE(3) in camera→world form
            se3 = vtf.SE3.from_matrix(cam2world)

            fr = self.server.scene.add_camera_frustum(
                name=f"/scene/object_{obj_id}/frustum_{i}",
                wxyz=se3.rotation().wxyz,
                position=se3.translation(),
                fov=self.fov,
                aspect=self.aspect,
                scale=self.frustum_scale,
                color=color,
                line_width=2.0,
            )
            self.active_frustums.append(fr)

    def _clear_frustums(self) -> None:
        for fr in self.active_frustums:
            fr.remove()
        self.active_frustums.clear()

    def _draw_trajectories(self) -> None:
        cmap = matplotlib.colormaps["viridis"]
        for i, traj in enumerate(self.map_trajectories):
            self.trajectory_frustums[i] = []
            n = len(traj)
            for j, cam2world in enumerate(traj):
                color = cmap(j / max(1, n - 1))[:3]
                se3 = vtf.SE3.from_matrix(cam2world)
                fr = self.server.scene.add_camera_frustum(
                    name=f"/scene/trajectories/{i}/frustum_{j}",
                    wxyz=se3.rotation().wxyz,
                    position=se3.translation(),
                    fov=self.fov,
                    aspect=self.aspect,
                    scale=self.frustum_scale,
                    color=color,
                    line_width=2.0,
                )
                fr.visible = False  # keep hidden initially
                self.trajectory_frustums[i].append(fr)

    # ----------------------------- run loop --------------------------------- #
    def run(self) -> None:
        print("viser running on http://localhost:8080 – open in browser.")
        while True:
            time.sleep(1.0)


def main(
    map_dir: Path,
    scenefun3d_data_dir: Path,
    scene_id: str,
    frustum_scale: float = 0.15,
    point_size: float = 0.02,
) -> None:

    # Load camera intrinsics which is a tuple (w, h, fx, fy, cx, cy)
    w, h, fx, fy, cx, cy = get_scene_intrinsics(
        data_path=scenefun3d_data_dir,
        scene_id=scene_id,
    )

    # Load camera trajectories which is a List[List[np.ndarray]]
    map_trajectories = get_scene_trajectories(
        data_path=scenefun3d_data_dir, scene_id=scene_id, stride=20
    )

    # from the intrinsics, calculate the fov and aspect ratio
    fov_x_radians = 2 * np.arctan2(w, 2 * fx)
    aspect_ratio = w / h

    ObjectMapViewer(
        map_dir=map_dir,
        map_trajectories=map_trajectories,
        fov=fov_x_radians,
        aspect=aspect_ratio,
        frustum_scale=frustum_scale,
        point_size=point_size,
    ).run()


if __name__ == "__main__":
    tyro.cli(main)
