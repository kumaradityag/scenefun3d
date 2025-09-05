import numpy as np
import open3d as o3d
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
from scipy.spatial import KDTree
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import zipfile
import re

from utils.data_parser import DataParser

# -------------------------------
# Skill mappings (kept for logging; not used in Task 2 lines)
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
# Utility: RLE Encoding (1-indexed)
# -------------------------------
def rle_encode(mask: np.ndarray) -> str:
    """
    Perform 1-indexed RLE encoding of a boolean 1D array.
    Returns a string of space-separated start length pairs.
    """
    mask = np.asarray(mask, dtype=np.uint8).ravel()
    # sentinel padding at both ends
    diffs = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.flatnonzero(diffs == 1) + 1  # 1-indexed
    ends = np.flatnonzero(diffs == -1) + 1  # 1-indexed exclusive
    lengths = ends - starts
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


# -------------------------------
# Filenames: sanitize desc_id -> filesystem/zip-safe
# -------------------------------
SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_\-]")


def sanitize_id(s: str) -> str:
    """
    Make desc_id filesystem/zip-safe and ensure no spaces.
    """
    s = s.replace(" ", "_")
    return SAFE_CHARS_RE.sub("_", s)


# -------------------------------
# Save masks in Task 2 benchmark format
# -------------------------------
def save_task2_format(
    masks: np.ndarray,  # (num_instances, num_laser_points) bool
    scores: List[float],
    visit_id: str,
    desc_ids: List[str],  # per-instance desc_id
    results_dir: Path,
    use_score: bool = True,
    score_precision: int = 4,
):
    """
    Group instances by desc_id and save:
      - {visit_id}_{desc_id}.txt at root with lines:
            predicted_masks/{visit_id}_{desc_id}_{iii}.txt <score>
      - predicted_masks/{visit_id}_{desc_id}_{iii}.txt each containing 1-line RLE
    """
    results_dir = Path(results_dir)
    pm_dir = results_dir / "predicted_masks"
    pm_dir.mkdir(parents=True, exist_ok=True)

    # Build groups per desc_id
    by_desc: Dict[str, List[int]] = {}
    for idx, d in enumerate(desc_ids):
        if d is None:
            continue
        by_desc.setdefault(d, []).append(idx)

    if not by_desc:
        print("[WARN] No desc_ids found among instances. Nothing to save.")
        return

    # For each description, write its txt index and the masks
    for raw_desc_id, idx_list in by_desc.items():
        # desc_id_safe = sanitize_id(raw_desc_id)
        desc_id_safe = raw_desc_id
        header_path = results_dir / f"{visit_id}_{desc_id_safe}.txt"

        # Delete existing outputs for this (visit, desc)
        if header_path.exists():
            header_path.unlink()
        for old in pm_dir.glob(f"{visit_id}_{desc_id_safe}_*.txt"):
            old.unlink()

        entries = []  # (score, line)
        running_counter = 0
        for inst_idx in idx_list:
            mask = masks[inst_idx].astype(bool)
            if mask.ndim != 1:
                mask = mask.ravel()
            rle_string = rle_encode(mask)

            mask_filename = f"{visit_id}_{desc_id_safe}_{running_counter:03d}.txt"
            rel_path = f"predicted_masks/{mask_filename}"
            abs_path = pm_dir / mask_filename

            with open(abs_path, "w") as f:
                f.write(rle_string + "\n")

            score = float(scores[inst_idx]) if use_score else 1.0
            entries.append((score, f"{rel_path} {score:.{score_precision}f}"))
            running_counter += 1

        # Sort lines by descending score (optional, but common)
        entries.sort(key=lambda x: x[0], reverse=True)
        lines = [ln for _, ln in entries]

        with open(header_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[Task2] Wrote {len(lines)} entries to {header_path.name}")


# -------------------------------
# Load point cloud and annotations
# -------------------------------
def load_map_and_segments(map_path: Path):
    map_path = Path(map_path)
    pcd = o3d.io.read_point_cloud(str(map_path / "point_cloud.pcd"))
    with open(map_path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)
    return pcd, segments_anno


# -------------------------------
# Extract functionality instances (+ desc_id)
# -------------------------------
def extract_functionality_instances_with_desc(
    segments_anno: dict,
    map_pcd: "o3d.geometry.PointCloud",
    skill_to_idx: Dict[str, int],
) -> Tuple[List[np.ndarray], List[Optional[int]], List[float], List[str]]:
    """
    Build lists for each affordance instance that has mask_3d_indices_global:
      - func_pointclouds: list[np.ndarray (N_i,3)]
      - func_types:      list[int or None]  (kept for logging; unused by Task 2 lines)
      - func_scores:     list[float]
      - func_desc_ids:   list[str]
    """
    map_points = np.asarray(map_pcd.points)  # (M, 3)

    func_pointclouds: List[np.ndarray] = []
    func_types: List[Optional[int]] = []
    func_scores: List[float] = []
    func_desc_ids: List[str] = []

    for seg in segments_anno.get("segGroups", []):
        inst_dict = seg.get("affordance_instances", {})
        for inst in inst_dict.values():
            skill_name = inst.get("type", None)
            desc_id = inst.get("desc_id", None)

            if desc_id is None:
                # Skip instances without a desc_id; cannot route them to Task 2 outputs
                print("[Warn] Skipping instance without desc_id.")
                continue

            global_idx = inst.get("mask_3d_indices_global", [])
            if not global_idx:
                continue

            pts = map_points[global_idx]  # (Ni, 3)
            score = float(inst.get("score", 1.0))

            func_pointclouds.append(pts)
            func_scores.append(score)
            func_desc_ids.append(str(desc_id))

            if skill_name in skill_to_idx:
                func_types.append(skill_to_idx[skill_name])
            else:
                func_types.append(None)

    return func_pointclouds, func_types, func_scores, func_desc_ids


# -------------------------------
# Align to laser scan -> build masks
# -------------------------------
def compute_masks_for_functionality_instances(
    func_pointclouds: List[np.ndarray],
    laser_scan_points: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    For each functionality instance (pointcloud), mark laser scan points within 'threshold' distance.
    Returns: masks (num_func_instances, num_laser_points) bool
    """
    kd_tree = KDTree(laser_scan_points)
    num_func = len(func_pointclouds)
    num_laser = laser_scan_points.shape[0]
    masks = np.zeros((num_func, num_laser), dtype=bool)

    for i, pts in tqdm(
        enumerate(func_pointclouds), total=num_func, desc="Mapping functionality masks"
    ):
        if pts.size == 0:
            continue
        neighbors = kd_tree.query_ball_point(pts, r=threshold)
        for idx_list in neighbors:
            for j in idx_list:
                if 0 <= j < num_laser:
                    masks[i, j] = True
    return masks


def compute_masks_from_laser_queries(
    func_pointclouds: List[np.ndarray],
    laser_scan_points: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Alternative: build KDTree on each func instance; query all laser points.
    Returns: masks (num_func_instances, num_laser_points) bool
    """
    num_func = len(func_pointclouds)
    num_laser = laser_scan_points.shape[0]
    masks = np.zeros((num_func, num_laser), dtype=bool)

    for i, func_pts in tqdm(
        enumerate(func_pointclouds),
        total=num_func,
        desc="Building KD-Trees and querying",
    ):
        if func_pts.size == 0:
            continue
        kd_tree = KDTree(func_pts)
        neighbors = kd_tree.query_ball_point(laser_scan_points, r=threshold)
        for laser_idx, hits in enumerate(neighbors):
            if hits:  # non-empty list
                masks[i, laser_idx] = True
    return masks


# -------------------------------
# Optional: mask NMS (kept from your script; off by default)
# -------------------------------
def filter_masks(masks, func_types, func_scores, iou_threshold=0.5):
    """
    Non-maximum suppression across all masks (keep smaller when overlap high).
    """
    print(f"Initial number of masks: {len(masks)}")
    sizes = masks.sum(axis=1)
    order = np.argsort(sizes)  # small first
    keep = []

    for idx in order:
        curr = masks[idx]
        keep_curr = True
        for kept_idx in keep:
            other = masks[kept_idx]
            inter = np.logical_and(curr, other).sum()
            union = np.logical_or(curr, other).sum()
            if union > 0 and (inter / union) > iou_threshold:
                keep_curr = False
                break
        if keep_curr:
            keep.append(idx)

    filtered_masks = masks[keep]
    filtered_types = [func_types[i] for i in keep]
    filtered_scores = [func_scores[i] for i in keep]
    print(f"Filtered number of masks: {len(filtered_masks)}")
    return (
        np.array(filtered_masks, dtype=bool),
        np.array(filtered_types, dtype=object),
        np.array(filtered_scores, dtype=float),
        np.array(keep, dtype=int),  # return index mapping to keep desc_ids aligned
    )


# -------------------------------
# Optional: re-map mask order back to original laser vertex order
# -------------------------------
def apply_vertex_index_map(
    mask: np.ndarray, index_map: Optional[np.ndarray]
) -> np.ndarray:
    """
    If your pipeline produced predictions on a *reordered/cropped* laser cloud,
    'index_map' should map your working laser indices -> original {visit_id}_laser_scan.ply indices.

    mask: bool array length = len(index_map) (working order)
    Returns: bool array length = original_num_vertices
    """
    if index_map is None:
        return mask
    original_len = int(index_map.max()) + 1
    remapped = np.zeros(original_len, dtype=bool)
    remapped[index_map] = mask[: len(index_map)]
    return remapped


# -------------------------------
# Optional: create a flat zip (no top-level dir)
# -------------------------------
def create_flat_zip(results_dir: Path, zip_path: Path):
    """
    Write a .zip where files live at the archive root (no wrapping folder).
    This matches the benchmark requirement: unzipping yields txt files at root
    and a 'predicted_masks/' subfolder.
    """
    results_dir = Path(results_dir)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in results_dir.rglob("*"):
            if p.is_file():
                arcname = str(p.relative_to(results_dir))
                zf.write(p, arcname=arcname)
    print(f"[Zip] Created {zip_path}")


# -------------------------------
# Dummy masks for missed tasks
# -------------------------------
def add_dummy_masks_for_missing_descriptions(
    data_parser: DataParser,
    visit_id: str,
    results_dir: Path,
    laser_scan_points: np.ndarray,
    existing_pred_desc_ids: List[str],
    dummy_score: float = 0.5,
    score_precision: int = 4,
):
    """
    For every description id (from descriptions.json) that does not already have a
    {visit_id}_{desc_id}.txt header file (i.e., no predictions saved), create:
      - predicted_masks/{visit_id}_{desc_id}_000.txt   (RLE of all-zero mask)
      - {visit_id}_{desc_id}.txt with one line:
            predicted_masks/{visit_id}_{desc_id}_000.txt <score>
    """
    descriptions = data_parser.get_descriptions(visit_id)
    all_desc_ids = []
    for d in descriptions:
        if "desc_id" in d:
            all_desc_ids.append(str(d["desc_id"]))
        elif "id" in d:
            all_desc_ids.append(str(d["id"]))

    pm_dir = Path(results_dir) / "predicted_masks"
    pm_dir.mkdir(parents=True, exist_ok=True)

    num_laser = laser_scan_points.shape[0]
    existing_set = set(map(str, existing_pred_desc_ids))

    for desc_id in set(all_desc_ids):
        header_path = Path(results_dir) / f"{visit_id}_{desc_id}.txt"
        if header_path.exists():
            continue  # already has predictions (or previously created dummy)
        # Only create dummy if genuinely missing
        dummy_mask = np.ones(num_laser, dtype=bool)
        rle_string = rle_encode(dummy_mask)  # likely ""
        mask_filename = f"{visit_id}_{desc_id}_000.txt"
        mask_rel_path = f"predicted_masks/{mask_filename}"
        with open(pm_dir / mask_filename, "w") as f:
            f.write(rle_string + "\n")
        with open(header_path, "w") as f:
            f.write(f"{mask_rel_path} {dummy_score:.{score_precision}f}\n")
        print(f"[Task2] Added dummy prediction for missing desc_id={desc_id}")


# -------------------------------
# Save masks in FUN3DU format (one .npz per desc_id)
# -------------------------------
def save_fun3du_format(
    masks: np.ndarray,
    desc_ids: List[str],
    scene_id: str,
    data_parser: DataParser,
    results_dir: Path,
    laser_scan_points: np.ndarray,
):
    """
    For every desc_id in the scene descriptions:
      Write {scene_id}_{desc_id}.npz containing an array 'masks' with shape
      (n, num_cropped_points). n = number of predicted masks for that desc_id.
      If none -> empty array with shape (0, num_cropped_points).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    crop_indices = data_parser.get_crop_mask(scene_id, return_indices=True)
    # Collect all description ids from dataset
    descriptions = data_parser.get_descriptions(scene_id)
    all_desc_ids = []
    for d in descriptions:
        if "desc_id" in d:
            all_desc_ids.append(str(d["desc_id"]))
        elif "id" in d:
            all_desc_ids.append(str(d["id"]))
    all_desc_ids = set(all_desc_ids)

    # Map desc_id -> list of mask indices
    mapping: Dict[str, List[int]] = {}
    for i, d in enumerate(desc_ids):
        if d is None:
            continue
        mapping.setdefault(str(d), []).append(i)

    num_laser = laser_scan_points.shape[0]
    num_cropped = len(crop_indices)

    for did in all_desc_ids:
        idxs = mapping.get(did, [])
        if idxs:
            arr = masks[idxs].astype(bool)[:, crop_indices]
        else:
            arr = np.zeros((0, num_cropped), dtype=bool)
        out_path = results_dir / f"{scene_id}_{did}.npz"
        np.savez(out_path, masks=arr)

    print(f"[FUN3DU] Wrote {len(all_desc_ids)} .npz files to {results_dir}")


# -------------------------------
# Main (Hydra)
# -------------------------------
@hydra.main(version_base=None, config_path="configs", config_name="task_masks")
def main(cfg: DictConfig):

    # Resolve paths
    map_path = Path(cfg.paths.map_dir)
    data_dir = Path(cfg.paths.scenefun3d_dir)
    results_dir = Path(cfg.paths.task2_pred_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load map + annotations
    map_pcd, segments_anno = load_map_and_segments(map_path)

    # Extract instances (with desc_id)
    func_pcds, func_types, func_scores, func_desc_ids = (
        extract_functionality_instances_with_desc(segments_anno, map_pcd, SKILL_TO_IDX)
    )
    print(f"Total functionality instances found (with desc_id): {len(func_pcds)}")

    if len(func_pcds) == 0:
        print("[WARN] No instances with desc_id found. Exiting.")
        return

    # Load laser scan in ORIGINAL vertex order
    data_parser = DataParser(data_dir)
    laser_scan = data_parser.get_laser_scan(
        cfg.scene
    )  # {visit_id}_laser_scan.ply order
    laser_scan_points = np.asarray(laser_scan.points)

    # Compute masks in ORIGINAL order
    threshold = float(cfg.threshold)
    if bool(getattr(cfg, "use_alt_query", False)):
        masks = compute_masks_from_laser_queries(
            func_pcds, laser_scan_points, threshold
        )
    else:
        masks = compute_masks_for_functionality_instances(
            func_pcds, laser_scan_points, threshold
        )

    # Optional: NMS
    keep_idx = None
    if getattr(cfg, "nms", None) and bool(cfg.nms.enable):
        masks, func_types, func_scores, keep_idx = filter_masks(
            masks, func_types, func_scores, iou_threshold=float(cfg.nms.iou_threshold)
        )
        func_desc_ids = [func_desc_ids[i] for i in keep_idx]

    # Persist intermediates (optional but handy)
    save_path = map_path
    np.save(save_path / "aff_masks_laserscan_task2.npy", masks)
    np.save(
        save_path / "aff_desc_ids_laserscan.npy", np.array(func_desc_ids, dtype=object)
    )
    np.save(save_path / "aff_scores_laserscan.npy", np.array(func_scores, dtype=float))
    print(f"Saved intermediates to {save_path}")

    # Save Task 2 submission format
    use_score = bool(cfg.score_flag)
    save_task2_format(
        masks=masks,
        scores=func_scores,
        visit_id=str(cfg.scene),
        desc_ids=func_desc_ids,
        results_dir=results_dir,
        use_score=use_score,
        score_precision=4,
    )
    print(f"[Task2] Saved submission files to {results_dir}")

    # ensure every description has at least one (dummy) mask/header if missing
    add_dummy_masks_for_missing_descriptions(
        data_parser=data_parser,
        visit_id=str(cfg.scene),
        results_dir=results_dir,
        laser_scan_points=laser_scan_points,
        existing_pred_desc_ids=func_desc_ids,
        dummy_score=0.5,
        score_precision=4,
    )

    # # Save FUN3DU grouped format
    # fun3du_dir = Path(cfg.paths.task2_fun3du_pred_dir)
    # fun3du_dir.mkdir(parents=True, exist_ok=True)
    # save_fun3du_format(
    #     masks=masks,
    #     desc_ids=func_desc_ids,
    #     scene_id=str(cfg.scene),
    #     data_parser=data_parser,
    #     results_dir=fun3du_dir,
    #     laser_scan_points=laser_scan_points,
    # )

    # Optional: create a flat zip ready for upload
    if bool(getattr(cfg, "make_zip", False)):
        zip_name = getattr(cfg, "zip_name", "submission_task2.zip")
        zip_path = results_dir.parent / zip_name
        create_flat_zip(results_dir, zip_path)
        print(f"[Task2] Flat zip ready: {zip_path}")


if __name__ == "__main__":
    main()
    main()
