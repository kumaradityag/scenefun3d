from utils.data_parser import DataParser
from pathlib import Path
from typing import List
import numpy as np

def get_scene_intrinsics(data_path: str, scene_id: str) -> tuple:
    """
    Get the camera intrinsics for a specific scene.
    Args:
        data_path (str): Path to the dataset.
        scene_id (str): Identifier of the scene.
    Returns:
        tuple: A tuple containing the camera intrinsics (fx, fy, cx, cy).
    """
    scenefun3d_dataset = DataParser(data_path)
    scene_path = Path(data_path) / scene_id
    video_ids = [d.name for d in scene_path.iterdir() if d.is_dir()]
    random_video_id = video_ids[0]  # select the first video id
    intrinsics_dict = scenefun3d_dataset.get_camera_intrinsics(
        visit_id=scene_id, video_id=random_video_id
    )
    random_timestamp = next(iter(intrinsics_dict))
    intrinsics_path = intrinsics_dict[random_timestamp]
    return scenefun3d_dataset.read_camera_intrinsics(
        intrinsics_file_path=intrinsics_path, format="tuple"
    )

def get_scene_trajectories(data_path: str, scene_id: str, stride: int) -> List[List[np.ndarray]]:
    """
    Get the camera trajectories for all videos in a scene.
    Args:
        data_path (str): Path to the dataset.
        scene_id (str): Identifier of the scene.
    Returns:
        List[List[np.ndarray]]: A list of camera poses for each video in the scene.
    Each video contains a list of poses, where each pose is a 4x4 transformation matrix (camera to world).
    """

    scenefun3d_dataset = DataParser(data_path)
    scene_path = Path(data_path) / scene_id
    video_ids = [d.name for d in scene_path.iterdir() if d.is_dir()]

    common_time_stamps = get_common_timestamps(
        scenefun3d_dataset, visit_id=scene_id, video_ids=video_ids
    )

    video_se3_poses = []

    for idx, video_id in enumerate(video_ids):
        video_common_timestamps = common_time_stamps[idx]
        poses_dict = scenefun3d_dataset.get_camera_trajectory(
            visit_id=scene_id, video_id=video_id
        )
        poses = [poses_dict[ts] for ts in video_common_timestamps]
        poses = poses[::stride]
        video_se3_poses.append(poses)

    return video_se3_poses
    

def get_common_timestamps(scenefun3d_dataset, visit_id: str, video_ids: List[str]) -> List[List[str]]:
    """
    Finds timestamps common across RGB, depth, intrinsics, and poses for each video_id separately.

    Returns:
        List[List[str]]: A list where each element is a sorted list of timestamps common to all data types in a video_id.
    """
    common_timestamps_per_video = []

    for video_id in video_ids:
        rgb = scenefun3d_dataset.get_rgb_frames(visit_id, video_id)
        depth = scenefun3d_dataset.get_depth_frames(visit_id, video_id)
        intrinsics = scenefun3d_dataset.get_camera_intrinsics(visit_id, video_id)
        poses = scenefun3d_dataset.get_camera_trajectory(visit_id, video_id)

        video_common_timestamps = sorted(
            set(rgb.keys())
            & set(depth.keys())
            & set(intrinsics.keys())
            & set(poses.keys())
        )
        common_timestamps_per_video.append(list(video_common_timestamps))

    return common_timestamps_per_video
