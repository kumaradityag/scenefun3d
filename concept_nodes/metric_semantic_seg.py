import numpy as np
from pathlib import Path

from eval.functionality_segmentation.eval_utils import util_3d
from eval.functionality_segmentation.eval_utils.benchmark_labels import (
    CLASS_LABELS,
    VALID_CLASS_IDS,
)

# -----------------------------------------------------------------------------
# User‐configurable inputs
# -----------------------------------------------------------------------------

GT_FILE = Path("/home/kumaraditya/datasets/scenefun3d/val_gt/420693.txt")
PRED_MAP_PATH = Path(
    "/home/kumaraditya/datasets/scenefun3d/cg_outputs_weekend_run/scenefun3d_420693_cg-detector_2025-05-04-09-55-48.559283"
)
# PRED_MAP_PATH = Path(
#     "/home/kumaraditya/datasets/scenefun3d/cg_outputs_weekend_run/scenefun3d_420693_cg-detector_2025-05-04-10-50-08.575507"
# )
PRED_MASKS_PATH = PRED_MAP_PATH / "functional_masks_laser_scan.npy"
PRED_TYPES_PATH = PRED_MAP_PATH / "functional_mask_types.npy"

# Map class‐IDs (1–9) to human‐readable names
ID_TO_LABEL = {vid: lbl for vid, lbl in zip(VALID_CLASS_IDS, CLASS_LABELS)}


# -----------------------------------------------------------------------------
# Ground‐truth semantic labels
# -----------------------------------------------------------------------------
def load_gt_semantic(gt_file: Path):
    """
    Returns
    -------
    gt_labels    : np.ndarray, shape (N_pts,), dtype=int
      class ID per point (0 = unlabeled/background)
    exclude_mask : np.ndarray, shape (N_pts,), dtype=bool
      True for points to ignore
    """
    gt_ids = util_3d.load_ids(gt_file)  # (N_pts,)
    exclude_mask = util_3d.get_excluded_point_mask(gt_ids)  # (N_pts,)

    # zero‐out excluded
    gt_ids_clean = gt_ids.copy()
    gt_ids_clean[exclude_mask] = 0

    # class = instance_id // 1000
    gt_labels = (gt_ids_clean // 1000).astype(int)
    return gt_labels, exclude_mask


# -----------------------------------------------------------------------------
# Convert instance‐based preds -> semantic labels
# -----------------------------------------------------------------------------
def load_pred_semantic(masks: np.ndarray, types: np.ndarray, exclude_mask: np.ndarray):
    """
    masks:  (N_inst, N_pts) bool
    types:  (N_inst,)         int
    exclude_mask: (N_pts,)     bool

    Returns
    -------
    pred_labels    : np.ndarray, shape (N_pts,), dtype=int
      per‐point class labels (0 = background)
    conflict_count : int
      number of points covered by ≥2 masks (before tie‐break)
    """
    # 1) remove excluded pnts
    masks = masks.copy()
    masks[:, exclude_mask] = False

    # 2) compute mask sizes
    sizes = masks.sum(axis=1)  # (N_inst,)

    N_pts = masks.shape[1]
    pred_labels = np.zeros(N_pts, dtype=int)
    conflict_count = 0

    # 3) assign each point
    for i in range(N_pts):
        inst_idxs = np.nonzero(masks[:, i])[0]
        if inst_idxs.size == 0:
            continue
        if inst_idxs.size == 1:
            pred_labels[i] = types[inst_idxs[0]]
        else:
            conflict_count += 1
            # pick the instance with largest mask
            # best = inst_idxs[np.argmax(sizes[inst_idxs])]
            best = inst_idxs[np.argmin(sizes[inst_idxs])]
            pred_labels[i] = types[best]

    return pred_labels, conflict_count, N_pts


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------
def compute_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray):
    """
    Returns
    -------
    metrics : dict[class_id -> dict of {'precision','recall','f1','iou'}]
    counts  : dict[class_id -> int]  # number of GT points for that class
    """
    metrics = {}
    counts = {}
    # classes = sorted(set(gt_labels.tolist()) - {0})
    classes = sorted(set(gt_labels.tolist()))
    for c in classes:
        gt_c = gt_labels == c
        pred_c = pred_labels == c

        tp = np.logical_and(gt_c, pred_c).sum()
        fp = np.logical_and(~gt_c, pred_c).sum()
        fn = np.logical_and(gt_c, ~pred_c).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
        counts[c] = int(gt_c.sum())  # total GT points for class c

    return metrics, counts


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_metrics_table(metrics: dict, counts: dict):
    """
    Prints per-class Precision/Recall/F1/IoU,
    then:
      - macro-average
      - frequency-weighted average
    """
    headers = ["Class", "Precision", "Recall", "F1", "IoU"]
    rows = []
    for c, m in metrics.items():
        name = ID_TO_LABEL.get(c, str(c))
        rows.append(
            [
                name,
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['iou']:.3f}",
            ]
        )

    # --- compute macro-averages ---
    macro = {
        k: np.mean([m[k] for m in metrics.values()])
        for k in ["precision", "recall", "f1", "iou"]
    }
    macro_row = [
        "UnweightedAvg",
        f"{macro['precision']:.3f}",
        f"{macro['recall']:.3f}",
        f"{macro['f1']:.3f}",
        f"{macro['iou']:.3f}",
    ]

    # --- compute frequency-weighted averages ---
    total_pts = sum(counts.values())
    fw = {}
    for k in ["precision", "recall", "f1", "iou"]:
        fw[k] = sum(metrics[c][k] * counts[c] for c in metrics) / total_pts
    fw_row = [
        "FreqWeighted",
        f"{fw['precision']:.3f}",
        f"{fw['recall']:.3f}",
        f"{fw['f1']:.3f}",
        f"{fw['iou']:.3f}",
    ]

    # --- figure out column widths for neat alignment ---
    table = [headers] + rows + [macro_row, fw_row]
    col_w = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    def fmt(r):
        return "  ".join(f"{str(r[i]):>{col_w[i]}}" for i in range(len(r)))

    # --- print ---
    print(fmt(headers))
    for r in rows:
        print(fmt(r))
    # midrule before summaries
    print("  ".join("-" * w for w in col_w))
    print(fmt(macro_row))
    print(fmt(fw_row))

    label_counts = {ID_TO_LABEL.get(c, str(c)): counts[c] for c in counts}
    print(f"\nClass Frequencies: {label_counts}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) load GT
    gt_labels, exclude_mask = load_gt_semantic(GT_FILE)

    # 2) load preds
    masks = np.load(PRED_MASKS_PATH)  # (N_inst, N_pts), bool
    types = np.load(PRED_TYPES_PATH)  # (N_inst,), int

    pred_labels, conflict_count, N_pts = load_pred_semantic(masks, types, exclude_mask)

    # 3) drop excluded points entirely
    valid_idx = ~exclude_mask
    gt_labels = gt_labels[valid_idx]
    pred_labels = pred_labels[valid_idx]

    # 4) compute & print
    metrics, counts = compute_metrics(gt_labels, pred_labels)
    print_metrics_table(metrics, counts)

    print(f"\nConflicting points (masked by size): {conflict_count}")
    print(f"Total points: {N_pts}")
