import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
GT_FILE = Path("/home/kumaraditya/datasets/scenefun3d/val_gt/420673.txt")
# PRED_MAP_PATH = Path(
#     "/home/kumaraditya/datasets/scenefun3d/cg_outputs_weekend_run/scenefun3d_420693_cg-detector_2025-05-04-09-55-48.559283"
# )
PRED_MAP_PATH = Path(
    "/home/kumaraditya/datasets/scenefun3d/cg_outputs_multi_runscenefun3d_420673_cg-detector_2025-05-06-15-57-36.326029"
)
IOU_THRESHOLDS = [0.10, 0.25, 0.50]  # list of IoU cut‐offs
WEIGHTED_AVERAGE = True

PRED_MASKS_PATH = PRED_MAP_PATH / "functional_masks_laser_scan.npy"
PRED_TYPES_PATH = PRED_MAP_PATH / "functional_mask_types.npy"

# Imports for GT parsing
from eval.functionality_segmentation.eval_utils import util_3d
from eval.functionality_segmentation.eval_utils.benchmark_labels import (
    CLASS_LABELS,
    VALID_CLASS_IDS,
)

ID_TO_LABEL = {vid: lbl for vid, lbl in zip(VALID_CLASS_IDS, CLASS_LABELS)}

# -----------------------------------------------------------------------------
# Ground‐truth loading
# -----------------------------------------------------------------------------
def load_gt_masks_and_types(gt_file: Path):
    """
    Returns
    -------
    gt_masks      : np.ndarray, shape (N_gt, N_pts), dtype=bool
    gt_types      : np.ndarray, shape (N_gt,), dtype=int
        The label_id (1–9) for each mask.
    exclude_mask  : np.ndarray, shape (N_pts,), dtype=bool
        True for points to be excluded from both GT and predictions.
    """
    gt_ids = util_3d.load_ids(gt_file)  # (N_pts,)
    exclude_mask = util_3d.get_excluded_point_mask(gt_ids)  # same shape

    # zero‐out excluded points in gt_ids
    gt_ids_clean = gt_ids.copy()
    gt_ids_clean[exclude_mask] = 0

    gt_instances = util_3d.get_instances(
        gt_ids_clean, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL
    )  # dict: class_name → list of instance‐dicts

    masks, types = [], []
    for inst_list in gt_instances.values():
        for inst in inst_list:
            inst_id = inst["instance_id"]
            label_id = inst["label_id"]
            mask = gt_ids_clean == inst_id
            masks.append(mask)
            types.append(label_id)

    return np.array(masks, dtype=bool), np.array(types, dtype=int), exclude_mask

# -----------------------------------------------------------------------------
# Prediction loading
# -----------------------------------------------------------------------------
def load_pred_masks_and_types(masks_path: Path, types_path: Path):
    masks = np.load(masks_path)  # (N_pred, N_pts)
    types = np.load(types_path)  # (N_pred,)
    return masks.astype(bool), types.astype(int)

# -----------------------------------------------------------------------------
# IoU computation & matching
# -----------------------------------------------------------------------------
def compute_iou_matrix(gt_masks: np.ndarray, pred_masks: np.ndarray):
    M, K = gt_masks.shape[0], pred_masks.shape[0]
    iou = np.zeros((M, K), dtype=float)
    for i in range(M):
        g = gt_masks[i]
        for j in range(K):
            p = pred_masks[j]
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            iou[i, j] = (inter / union) if union > 0 else 0.0
    return iou


def evaluate_recall(gt_masks, gt_types, pred_masks, pred_types, iou_thresholds):
    """
    Returns
    -------
    results : dict[class_id -> list of recalls at each threshold]
    counts  : dict[class_id -> number of GT instances]
    """
    results = {}
    counts = {}
    all_classes = sorted(set(gt_types.tolist()))

    for c in all_classes:
        gt_idx = np.where(gt_types == c)[0]
        pred_idx = np.where(pred_types == c)[0]
        counts[c] = len(gt_idx)

        if counts[c] == 0:
            continue  # no GT for this class

        gt_sub = gt_masks[gt_idx]
        pred_sub = (
            pred_masks[pred_idx]
            if pred_idx.size
            else np.zeros((0, gt_masks.shape[1]), bool)
        )

        if pred_sub.shape[0] == 0:
            # no predictions ⇒ zero recall across all thresholds
            results[c] = [0.0] * len(iou_thresholds)
            continue

        iou_mat = compute_iou_matrix(gt_sub, pred_sub)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_ious = iou_mat[row_ind, col_ind]

        recalls = [(matched_ious >= thr).sum() / counts[c] for thr in iou_thresholds]
        results[c] = recalls

    return results, counts

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_recall_table(results, counts, iou_thresholds, weighted):
    """
    Prints a Recall@IoU table per class and one final row:
    - if weighted=True: weighted average by counts
    - else: simple mean
    """
    # --- build header and rows ---
    headers = ["Class"] + [f"R@{thr:.2f}" for thr in iou_thresholds]
    rows = [
        [ID_TO_LABEL.get(c, str(c))] + [f"{r:.3f}" for r in recs]
        for c, recs in results.items()
    ]

    # --- compute average row ---
    if weighted:
        total = sum(counts[c] for c in results)
        avg_vals = [
            sum(results[c][i] * counts[c] for c in results) / total
            for i in range(len(iou_thresholds))
        ]
        avg_label = "Weighted Avg"
    else:
        avg_vals = np.mean(list(results.values()), axis=0)
        avg_label = "Unweighted"

    avg_row = [avg_label] + [f"{v:.3f}" for v in avg_vals]

    # --- determine column widths ---
    table = [headers] + rows + [avg_row]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    # --- helper to format a single row ---
    def fmt(row):
        return "  ".join(f"{str(cell):>{col_widths[i]}}" for i, cell in enumerate(row))

    # --- print header ---
    print(fmt(headers))

    # --- print per-class rows ---
    for row in rows:
        print(fmt(row))

    # --- midrule before average ---
    print("  ".join("-" * w for w in col_widths))

    # --- print average row ---
    print(fmt(avg_row))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # load GT + exclusion mask
    gt_masks, gt_types, exclude_mask = load_gt_masks_and_types(GT_FILE)

    # load predictions, then apply exclusion
    pred_masks, pred_types = load_pred_masks_and_types(PRED_MASKS_PATH, PRED_TYPES_PATH)
    pred_masks[:, exclude_mask] = False

    # evaluate and print
    results, counts = evaluate_recall(
        gt_masks, gt_types, pred_masks, pred_types, IOU_THRESHOLDS
    )
    print_recall_table(results, counts, IOU_THRESHOLDS, WEIGHTED_AVERAGE)
