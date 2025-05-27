import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from utils.data_parser import DataParser

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
IDX_TO_SKILL = {i: name for i, name in enumerate(SKILL_LIST, start=1)}


def rle_encode(mask: np.ndarray) -> str:
    """
    Perform 1-indexed RLE encoding of a boolean array.
    Returns a string of space-separated start and length pairs.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    padded = np.concatenate([[0], mask, [0]])
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1) + 1
    ends = np.flatnonzero(diffs == -1) + 1
    lengths = ends - starts
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def compute_mapping_mask(
    gt_pts: np.ndarray, pred_pts: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Mark each GT point as True if any predicted point lies within `threshold`.
    KD-Tree is built on GT points, then we query each pred point.
    """
    tree = KDTree(gt_pts)
    neighbors = tree.query_ball_point(pred_pts, r=threshold)
    mask = np.zeros(len(gt_pts), dtype=bool)
    for nbr_list in neighbors:
        for idx in nbr_list:
            mask[idx] = True
    return mask


def load_gt_labels(gt_txt: Path) -> np.ndarray:
    """Load per-point GT IDs (0,255 background; else 4-digit instance IDs)."""
    return np.loadtxt(str(gt_txt), dtype=int)


def log_coverage_stats(gt_ids: np.ndarray, mask: np.ndarray):
    """Print before/after counts per-class (thousands digit) and total."""
    valid = np.logical_and(gt_ids != 0, gt_ids != 255)
    total_b = valid.sum()
    total_a = int(np.logical_and(valid, mask).sum())
    miss = total_b - total_a
    print(
        f"TOTAL: before={total_b}, after={total_a}, "
        f"missing={miss} ({miss/total_b*100:.1f}%)"
    )
    classes = sorted({i // 1000 for i in gt_ids[valid]})
    for c in classes:
        cls_mask = (gt_ids // 1000) == c
        b = int(np.logical_and(cls_mask, valid).sum())
        a = int(np.logical_and(cls_mask, mask).sum())
        m = b - a
        p = m / b * 100 if b > 0 else 0.0
        print(f"{IDX_TO_SKILL[c]}: before={b}, after={a}, missing={m} ({p:.1f}%)")


def build_instance_masks(masked_ids: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    From masked per-point IDs, build instance masks.
    Returns:
      masks: (num_instances, N_pts) bool array
      types: list of class-IDs (thousands digit)
    """
    inst_ids = np.unique(masked_ids)
    inst_ids = [i for i in inst_ids if i not in (0, 255)]
    masks = []
    types = []
    for iid in sorted(inst_ids):
        masks.append(masked_ids == iid)
        types.append(iid // 1000)
    return np.stack(masks, axis=0), types


def save_benchmark_format(
    masks: np.ndarray,
    func_types: list[int],
    func_scores: list[float],
    visit_id: str,
    results_dir: Path,
):
    """Save masks + metadata exactly in your benchmark format."""
    results_dir = Path(results_dir)
    txt_file = results_dir / f"{visit_id}.txt"
    if txt_file.exists():
        txt_file.unlink()

    pm_dir = results_dir / "predicted_masks"
    if pm_dir.exists():
        for f in pm_dir.glob(f"{visit_id}_*"):
            f.unlink()
    pm_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, (mask, cls_id, score) in enumerate(zip(masks, func_types, func_scores)):
        rle = rle_encode(mask)
        fn = f"{visit_id}_{i:03d}.txt"
        with open(pm_dir / fn, "w") as f:
            f.write(rle + "\n")
        entries.append((score, f"predicted_masks/{fn} {cls_id} {score:.3f}"))

    entries.sort(key=lambda x: x[0], reverse=True)
    with open(results_dir / f"{visit_id}.txt", "w") as f:
        f.write("\n".join(line for _, line in entries))

    print(f"Benchmark saved to {results_dir}")


def main(args):
    # 1) Load GT scan via DataParser
    dp = DataParser(args.data_dir)
    gt_pc = dp.get_laser_scan(args.visit_id)
    gt_pts = np.asarray(gt_pc.points)

    # 2) Load predicted map points
    pred_pc = o3d.io.read_point_cloud(str(args.pred_map_dir / "point_cloud.pcd"))
    pred_pts = np.asarray(pred_pc.points)

    # 3) Compute mapping mask
    mapping_mask = compute_mapping_mask(gt_pts, pred_pts, args.threshold)

    # 4) Load GT labels and log coverage
    gt_ids = load_gt_labels(args.val_gt_folder / f"{args.visit_id}.txt")
    print("--- COVERAGE STATS ---")
    log_coverage_stats(gt_ids, mapping_mask)

    # 5) Zero-out unmapped GT points
    masked_ids = np.where(mapping_mask, gt_ids, 0)

    # 6) Build instance masks/types
    masks, types = build_instance_masks(masked_ids)
    scores = [1.0] * len(types)

    # 7) Save benchmark results
    save_benchmark_format(masks, types, scores, args.visit_id, args.results_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate pseudo-predictions by masking unmapped GT points."
    )
    p.add_argument(
        "--data_dir", type=Path, required=True, help="Root data folder for DataParser"
    )
    p.add_argument(
        "--pred_map_dir",
        type=Path,
        required=True,
        help="Directory containing point_cloud.pcd",
    )
    p.add_argument(
        "--val_gt_folder",
        type=Path,
        required=True,
        help="Folder of <visit_id>.txt GT label files",
    )
    p.add_argument(
        "--visit_id", type=str, required=True, help="Scene identifier / filename prefix"
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Where to write benchmark outputs",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Distance threshold for mapping GT→pred",
    )
    args = p.parse_args()
    main(args)
