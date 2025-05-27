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
def save_benchmark_format(masks, func_types, func_scores, visit_id, results_dir):
    """
    Save instance masks and metadata in benchmark format.

    Args:
        masks: (num_func_instances, num_laser_points) boolean array
        func_types: list of int, affordance class IDs
        func_scores: list of float, confidence scores
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

    entries = []

    for i, (mask, class_id, score) in enumerate(zip(masks, func_types, func_scores)):
        rle_string = rle_encode(mask)
        mask_filename = f"{visit_id}_{i:03d}.txt"
        rel_path = f"predicted_masks/{mask_filename}"
        abs_path = predicted_masks_dir / mask_filename

        # Save the RLE string
        with open(abs_path, "w") as f:
            f.write(rle_string + "\n")

        score = 1.0
        # Build entry: txt_line = "relative_path class_id score"
        txt_line = f"{rel_path} {class_id} {score:.3f}"
        entries.append((score, txt_line))

    # Sort by score, highest to lowest
    entries.sort(key=lambda x: x[0], reverse=True)

    score_threshold = 0.0
    txt_lines = []
    for score, line in entries:
        if score > score_threshold:
            txt_lines.append(line)

    print(txt_lines)

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
    skill_to_idx: Dict[str, int],
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
    map_points = np.asarray(map_pcd.points)  # (M, 3)
    func_pointclouds: list[np.ndarray] = []
    func_types: list[int] = []
    func_scores: list[float] = []

    for seg in segments_anno["segGroups"]:
        inst_dict = seg.get("affordance_instances", {})

        for inst in inst_dict.values():
            skill_name = inst["type"]  # e.g. "ROTATE"
            if skill_name not in skill_to_idx:
                print(f"Warning: {skill_name} not in skill list – skipped.")
                continue

            global_idx = inst.get("mask_3d_indices_global", [])
            if not global_idx:  # empty or missing
                continue

            score = inst.get("score", 1.0)

            pts = map_points[global_idx]  # shape (Ni, 3)
            func_pointclouds.append(pts)
            func_types.append(skill_to_idx[skill_name])
            func_scores.append(score)

    return func_pointclouds, func_types, func_scores


# -------------------------------
# Align to laser scan
# -------------------------------
def compute_masks_for_functionality_instances(
    func_pointclouds, laser_scan_points, threshold=0.5
):
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

    for i, pts in tqdm(
        enumerate(func_pointclouds),
        total=num_func_instances,
        desc="Mapping functionality masks",
    ):
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


def compute_masks_from_laser_queries(
    func_pointclouds, laser_scan_points, threshold=0.5
):
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

    for i, func_pts in tqdm(
        enumerate(func_pointclouds),
        total=num_func_instances,
        desc="Building KD-Trees and querying",
    ):
        if func_pts.shape[0] == 0:
            continue

        kd_tree = KDTree(func_pts)
        neighbors = kd_tree.query_ball_point(laser_scan_points, r=threshold)

        for laser_idx, is_close in enumerate(neighbors):
            if (
                is_close
            ):  # if the list is non-empty, the laser point is close to this functionality instance
                masks[i, laser_idx] = True

    return masks


def filter_masks(masks, func_types, func_scores, iou_threshold=0.5):
    """
    Perform non-maximum suppression across all masks.
    When two masks overlap above iou_threshold, keep the one with fewer points.
    """
    print(f"Initial number of masks: {len(masks)}")

    # Compute mask sizes (number of True entries per mask)
    sizes = masks.sum(axis=1)
    # Sort indices so smaller masks are tried first
    order = np.argsort(sizes)
    keep = []

    for idx in order:
        curr = masks[idx]
        keep_curr = True
        # Compare against all already kept masks
        for kept_idx in keep:
            other = masks[kept_idx]
            inter = np.logical_and(curr, other).sum()
            union = np.logical_or(curr, other).sum()
            if union > 0 and (inter / union) > iou_threshold:
                keep_curr = False
                break
        if keep_curr:
            keep.append(idx)

    # Gather results in the order they were kept
    filtered_masks = masks[keep]
    filtered_types = [func_types[i] for i in keep]
    filtered_scores = [func_scores[i] for i in keep]

    print(f"Filtered number of masks: {len(filtered_masks)}")

    return (
        np.array(filtered_masks, dtype=bool),
        np.array(filtered_types, dtype=int),
        np.array(filtered_scores, dtype=float),
    )


# -------------------------------
# Main
# -------------------------------
def main(args):
    map_pcd, segments_anno = load_map_and_segments(args.map_path)

    # Extract functionality instances
    func_pcds, func_types, func_scores = extract_functionality_instances(
        segments_anno, map_pcd, SKILL_TO_IDX
    )
    print(f"Total functionality instances found: {len(func_pcds)}")

    # Load laser scan
    data_parser = DataParser(args.data_dir)
    laser_scan = data_parser.get_laser_scan(args.visit_id)
    laser_scan_points = np.asarray(laser_scan.points)

    # Align and compute masks
    threshold = args.threshold
    masks = compute_masks_for_functionality_instances(
        func_pcds, laser_scan_points, threshold
    )
    # masks = compute_masks_from_laser_queries(func_pcds, laser_scan_points, threshold)

    # masks, func_types, func_scores = filter_masks(
    #     masks, func_types, func_scores, iou_threshold=0.25
    # )

    # print(f"Final number of functionality instances: {len(func_types)}")

    # Save results
    save_path = Path(args.map_path)
    np.save(save_path / "functional_masks_laser_scan.npy", masks)
    np.save(save_path / "functional_mask_types.npy", np.array(func_types))

    print(f"Saved {masks.shape[0]} functionality masks to {save_path}")

    benchmark_results_path = args.results_dir
    save_benchmark_format(
        masks, func_types, func_scores, args.visit_id, benchmark_results_path
    )
    print(f"Saved benchmark format to {benchmark_results_path}")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Path of the data")
    parser.add_argument("--visit_id", required=True, help="Identifier of the scene")
    parser.add_argument(
        "--map_path", required=True, help="Path of the concept-nodes map"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Distance threshold for matching functionality instances to laser scan points (in meters)",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default="/datasets/scenefun3d/val_pred",
        help="Directory to save benchmark results",
    )
    args = parser.parse_args()
    main(args)
