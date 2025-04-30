import numpy as np
import open3d as o3d
import os
import argparse
from pathlib import Path
import json
from scipy.spatial import KDTree
from tqdm import tqdm
from typing import List, Tuple, Dict

from utils.data_parser import DataParser

# -------------------------------
# Skill mappings
# -------------------------------
SKILL_LIST = [
    "ROTATE",
    "KEY PRESS",
    "TIP PUSH",
    "HOOK PULL",
    "PINCH PULL",
    "HOOK TURN",
    "FOOT PUSH",
    "PLUG IN",
    "UNPLUG",
]
SKILL_TO_IDX = {name: i + 1 for i, name in enumerate(SKILL_LIST)}

# -------------------------------
# Utility: RLE Encoding
# -------------------------------
def rle_encode(mask):
    """
    Perform 1-indexed RLE encoding of a boolean array.
    Returns a string of space-separated start and length pairs.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    starts = np.flatnonzero(np.diff(np.concatenate([[0], mask, [0]])) == 1) + 1
    ends = np.flatnonzero(np.diff(np.concatenate([[0], mask, [0]])) == -1) + 1
    lengths = ends - starts
    rle = " ".join(f"{s} {l}" for s, l in zip(starts, lengths))
    return rle


# -------------------------------
# Save masks in benchmark format
# -------------------------------
def save_benchmark_format(masks, func_types, visit_id, results_dir):
    """
    Save instance masks and metadata in benchmark format.

    Args:
        masks: (num_func_instances, num_laser_points) boolean array
        func_types: list of int, affordance class IDs
        visit_id: str, scene identifier
        results_dir: str or Path, base directory to save benchmark results
    """
    results_dir = Path(results_dir)

    # Delete existing results
    visit_file = results_dir / f"{visit_id}.txt"
    if visit_file.exists():
        visit_file.unlink()

    predicted_masks_dir = results_dir / "predicted_masks"
    if predicted_masks_dir.exists():
        for mask_file in predicted_masks_dir.glob(f"{visit_id}_*"):
            mask_file.unlink()

    predicted_masks_dir = results_dir / "predicted_masks"
    predicted_masks_dir.mkdir(parents=True, exist_ok=True)

    txt_lines = []

    for i, (mask, class_id) in enumerate(zip(masks, func_types)):
        rle_string = rle_encode(mask)
        mask_filename = f"{visit_id}_{i:03d}.txt"
        rel_path = f"predicted_masks/{mask_filename}"
        abs_path = predicted_masks_dir / mask_filename

        # Save the RLE string
        with open(abs_path, "w") as f:
            f.write(rle_string + "\n")

        # Add entry to visit_id.txt
        txt_line = f"{rel_path} {class_id} 1.0"
        txt_lines.append(txt_line)

    # Save the visit_id.txt file
    with open(results_dir / f"{visit_id}.txt", "w") as f:
        f.write("\n".join(txt_lines))

    print(f"Benchmark format saved to {results_dir}")


# -------------------------------
# Load point cloud and annotations
# -------------------------------
def load_map_and_segments(map_path):
    map_path = Path(map_path)
    pcd = o3d.io.read_point_cloud(str(map_path / "point_cloud.pcd"))
    with open(map_path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)
    return pcd, segments_anno


# -------------------------------
# Extract functionality instances
# -------------------------------
def extract_functionality_instances(
    segments_anno: dict,
    map_pcd: "o3d.geometry.PointCloud",
    skill_to_idx: Dict[str, int]
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Build a (point‑cloud, skill_id) list for *every affordance instance* stored
    in `segments_anno` that has a valid mask_3d_indices_global field.

    Returns
    -------
    func_pointclouds : list[np.ndarray]
        Each array has shape (N_i, 3) containing XYZs of one affordance instance.
    func_types       : list[int]
        Parallel list of integer skill indices (from `skill_to_idx`).
    """
    map_points = np.asarray(map_pcd.points)      # (M, 3)
    func_pointclouds: list[np.ndarray] = []
    func_types:       list[int] = []

    for seg in segments_anno["segGroups"]:
        inst_dict = seg.get("affordance_instances", {})

        for inst in inst_dict.values():
            skill_name = inst["type"]                       # e.g. "ROTATE"
            if skill_name not in skill_to_idx:
                print(f"Warning: {skill_name} not in skill list – skipped.")
                continue

            global_idx = inst.get("mask_3d_indices_global", [])
            if not global_idx:              # empty or missing
                continue

            pts = map_points[global_idx]    # shape (Ni, 3)
            func_pointclouds.append(pts)
            func_types.append(skill_to_idx[skill_name])

    return func_pointclouds, func_types


# -------------------------------
# Align to laser scan
# -------------------------------
def compute_masks_for_functionality_instances(func_pointclouds, laser_scan_points, threshold=0.5):
    """
    For each functionality instance (a pointcloud), find all laser scan points within a given threshold distance
    from any point in the functionality pointcloud. The resulting mask marks those laser points.

    Args:
        func_pointclouds: list of (N_i, 3) np.ndarray representing functionality instances
        laser_scan_points: (M, 3) np.ndarray of laser scan points
        threshold: float, distance threshold for matching

    Returns:
        masks: (num_func_instances, num_laser_points) boolean mask array
    """
    kd_tree = KDTree(laser_scan_points)
    num_func_instances = len(func_pointclouds)
    num_laser_pts = laser_scan_points.shape[0]
    masks = np.zeros((num_func_instances, num_laser_pts), dtype=bool)

    for i, pts in tqdm(enumerate(func_pointclouds), total=num_func_instances, desc="Mapping functionality masks"):
        if pts.shape[0] == 0:
            continue

        # Get all laser scan indices within `threshold` distance from each point in the functionality pointcloud
        neighbors = kd_tree.query_ball_point(pts, r=threshold)

        # Flatten the list of neighbor indices and set them in the mask
        for idx_list in neighbors:
            for idx in idx_list:
                if idx < num_laser_pts:
                    masks[i, idx] = True

    return masks

def compute_masks_from_laser_queries(func_pointclouds, laser_scan_points, threshold=0.5):
    """
    For each laser scan point, check which functionality pointclouds it is close to (within threshold).
    Then mark those entries in the mask.

    Args:
        func_pointclouds: list of (N_i, 3) np.ndarray representing functionality instances
        laser_scan_points: (M, 3) np.ndarray of laser scan points
        threshold: float, distance threshold for matching

    Returns:
        masks: (num_func_instances, num_laser_points) boolean mask array
    """
    num_func_instances = len(func_pointclouds)
    num_laser_pts = laser_scan_points.shape[0]
    masks = np.zeros((num_func_instances, num_laser_pts), dtype=bool)

    for i, func_pts in tqdm(enumerate(func_pointclouds), total=num_func_instances, desc="Building KD-Trees and querying"):
        if func_pts.shape[0] == 0:
            continue

        kd_tree = KDTree(func_pts)
        neighbors = kd_tree.query_ball_point(laser_scan_points, r=threshold)

        for laser_idx, is_close in enumerate(neighbors):
            if is_close:  # if the list is non-empty, the laser point is close to this functionality instance
                masks[i, laser_idx] = True

    return masks



# -------------------------------
# Main
# -------------------------------
def main(args):
    map_pcd, segments_anno = load_map_and_segments(args.map_path)

    # Extract functionality instances
    func_pcds, func_types = extract_functionality_instances(segments_anno, map_pcd, SKILL_TO_IDX)
    print(f"Total functionality instances found: {len(func_pcds)}")

    # Load laser scan
    data_parser = DataParser(args.data_dir)
    laser_scan = data_parser.get_laser_scan(args.visit_id)
    laser_scan_points = np.asarray(laser_scan.points)

    # Align and compute masks 
    threshold = 0.01  # 1 cm
    masks = compute_masks_for_functionality_instances(func_pcds, laser_scan_points, threshold)
    # masks = compute_masks_from_laser_queries(func_pcds, laser_scan_points, threshold)

    # Save results
    save_path = Path(args.map_path)
    np.save(save_path / "functional_masks_laser_scan.npy", masks)
    np.save(save_path / "functional_mask_types.npy", np.array(func_types))

    print(f"Saved {masks.shape[0]} functionality masks to {save_path}")

    benchmark_results_path = "/data/kumaraditya/scenefun3d/val_pred"
    save_benchmark_format(masks, func_types, args.visit_id, benchmark_results_path)
    print(f"Saved benchmark format to {benchmark_results_path}")



# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Path of the data")
    parser.add_argument("--visit_id", required=True, help="Identifier of the scene")
    parser.add_argument("--map_path", required=True, help="Path of the concept-nodes map")
    args = parser.parse_args()
    main(args)
