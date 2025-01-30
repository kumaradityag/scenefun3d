import numpy as np
import open3d as o3d
import os
import argparse
from pathlib import Path
import json
from scipy.spatial import KDTree
from tqdm import tqdm

from utils.data_parser import DataParser
from utils.pc_process import pc_estimate_normals

# from .CLIP import CLIP


def load_map_pcd_and_segments(map_path):
    """
    Loads the single combined point cloud (point_cloud.pcd) and the segments JSON.
    Returns:
        pcd (o3d.geometry.PointCloud): The full map point cloud.
        segments_anno (dict): The loaded segments_anno from segments_anno.json
    """
    map_path = Path(map_path)
    pcd = o3d.io.read_point_cloud(str(map_path / "point_cloud.pcd"))

    with open(map_path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)

    return pcd, segments_anno


def build_instance_map(pcd, segments_anno):
    """
    Creates a structure that, for each point index in 'pcd', gives the set of instance indices
    that contain that point.

    Args:
        pcd (o3d.geometry.PointCloud): The map point cloud.
        segments_anno (dict): The annotation dictionary with "segGroups" and each
                              group containing a "segments" array.

    Returns:
        instance_ids_for_map_pt: a list of sets, where instance_ids_for_map_pt[i]
                                 is the set of instance indices that contain point i.
        num_instances: integer count of total instances
    """
    num_map_points = np.asarray(pcd.points).shape[0]
    seg_groups = segments_anno["segGroups"]

    # Prepare a list of empty sets for each point index
    instance_ids_for_map_pt = [set() for _ in range(num_map_points)]

    # Fill in which map-point indices belong to which instance
    for inst_idx, ann in enumerate(seg_groups):
        for pt_idx in ann["segments"]:
            instance_ids_for_map_pt[pt_idx].add(inst_idx)

    num_instances = len(seg_groups)
    return instance_ids_for_map_pt, num_instances


def map_laser_scan_to_instances(
    laser_scan_points, pcd, instance_ids_for_map_pt, num_instances, threshold=0.05
):
    """
    For each laser-scan point, find the nearest neighbor in the map pcd via a KDTree.
    If the distance < threshold, that laser point is assigned to the same instance(s)
    as the nearest neighbor.

    Args:
        laser_scan_points (ndarray): shape (N, 3), laser scan points
        pcd (o3d.geometry.PointCloud): the map point cloud
        instance_ids_for_map_pt (List[Set[int]]):
            for each index i in the map pcd, a set of instance indices
        num_instances (int): total number of instances
        threshold (float): distance threshold for matching

    Returns:
        masks (np.ndarray): shape (num_instances, N) boolean array,
                            masks[i, j] = True if laser_scan_points[j] belongs to instance i
    """
    map_pts = np.asarray(pcd.points)
    num_laser_pts = laser_scan_points.shape[0]

    # Build KDTree
    kd_tree = KDTree(map_pts)

    masks = np.zeros((num_instances, num_laser_pts), dtype=bool)

    # For each laser point, find its neighbor in the map pcd
    for j in tqdm(range(num_laser_pts)):
        point = laser_scan_points[j]
        dist, idx = kd_tree.query(point)
        if dist < threshold:
            # This laser point belongs to all instances that that map-pt belongs to
            inst_ids = instance_ids_for_map_pt[idx]
            for inst_id in inst_ids:
                masks[inst_id, j] = True

    return masks


def main(args):
    # -----------------------
    #  Load map + annotations
    # -----------------------
    map_pcd, segments_anno = load_map_pcd_and_segments(args.map_path)

    # Build a structure telling us which map-pt belongs to which instance(s)
    instance_ids_for_map_pt, num_instances = build_instance_map(map_pcd, segments_anno)
    print(f"Found {num_instances} instances in the map.")

    # -----------------------
    #  Potentially load CLIP
    # -----------------------
    # if False:
    #     # Only load if needed
    #     ft_extractor = CLIP(
    #         model_name="ViT-H-14",
    #         pretrained="laion2b_s32b_b79k",
    #         device="cuda",
    #         cache_dir="/home/kumaraditya/checkpoints",
    #     )

    # -----------------------
    #  Load a laser scan
    # -----------------------
    data_parser = DataParser(args.data_dir)
    laser_scan = data_parser.get_laser_scan(args.visit_id)
    laser_scan = data_parser.get_cropped_laser_scan(args.visit_id, laser_scan)
    laser_scan_points = np.asarray(laser_scan.points)

    # -----------------------
    #  Create the instance masks for the laser scan
    # -----------------------
    # For each laser point, find the nearest neighbor in the map pcd (with threshold).
    threshold = 0.025  # or 0.05, or 0.025, etc.
    masks = map_laser_scan_to_instances(
        laser_scan_points,
        map_pcd,
        instance_ids_for_map_pt,
        num_instances,
        threshold=threshold,
    )

    print("masks shape:", masks.shape)  # (num_instances, num_laser_points)

    masks_save_path = Path(args.map_path) / "masks_laser_scan.npy"
    np.save(masks_save_path, masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Path of the data")
    parser.add_argument("--visit_id", required=True, help="Identifier of the scene")
    parser.add_argument(
        "--map_path", required=True, help="Path of the concept-nodes map"
    )
    args = parser.parse_args()

    main(args)
