import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import hydra
from omegaconf import DictConfig

from eval.functionality_segmentation.eval_utils import util_3d
from eval.functionality_segmentation.eval_utils.benchmark_labels import (
    CLASS_LABELS,
    VALID_CLASS_IDS,
)

DUMMY_CLASS_ID = 10
DUMMY_CLASS_LABEL = "all"

ID_TO_LABEL = {vid: lbl for vid, lbl in zip(VALID_CLASS_IDS, CLASS_LABELS)}
ID_TO_LABEL[DUMMY_CLASS_ID] = DUMMY_CLASS_LABEL


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
    )  # dict: class_name -> list of instance‐dicts

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
    results   : dict[class_id -> list of recalls at each threshold]
    counts    : dict[class_id -> number of GT instances]
    tp_dict   : dict[class_id -> list of TP counts at each threshold]
    fp_dict   : dict[class_id -> list of FP counts at each threshold]
    fn_dict   : dict[class_id -> list of FN counts at each threshold]
    """
    results = {}
    counts = {}
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
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
            # no predictions ⇒ zero recall, zero TP/FP, all GT as FN
            results[c] = [0.0] * len(iou_thresholds)
            tp_dict[c] = [0] * len(iou_thresholds)
            fp_dict[c] = [0] * len(iou_thresholds)
            fn_dict[c] = [counts[c]] * len(iou_thresholds)
            continue

        iou_mat = compute_iou_matrix(gt_sub, pred_sub)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_ious = iou_mat[row_ind, col_ind]

        recalls = [(matched_ious >= thr).sum() / counts[c] for thr in iou_thresholds]
        tps = [(matched_ious >= thr).sum() for thr in iou_thresholds]
        fps = [pred_sub.shape[0] - tp for tp in tps]
        fns = [counts[c] - tp for tp in tps]

        results[c] = recalls
        tp_dict[c] = tps
        fp_dict[c] = fps
        fn_dict[c] = fns

    return results, counts, tp_dict, fp_dict, fn_dict


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def print_recall_table(results, counts, iou_thresholds):
    """
    Prints a Recall@IoU table per class and two final rows:
    - Weighted Avg: weighted by counts
    - Unweighted: simple mean
    """
    # --- build header and rows ---
    headers = ["Class"] + [f"R@{thr:.2f}" for thr in iou_thresholds]
    rows = [
        [ID_TO_LABEL.get(c, str(c))] + [f"{r:.3f}" for r in recs]
        for c, recs in results.items()
    ]

    # --- compute average rows ---
    total = sum(counts[c] for c in results)
    avg_w = [
        sum(results[c][i] * counts[c] for c in results) / total
        for i in range(len(iou_thresholds))
    ]
    weighted_row = ["Weighted Avg"] + [f"{v:.3f}" for v in avg_w]

    avg_unw = np.mean(list(results.values()), axis=0)
    unweighted_row = ["Unweighted"] + [f"{v:.3f}" for v in avg_unw]

    # --- determine column widths ---
    table = [headers] + rows + [weighted_row, unweighted_row]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    # --- helper to format a single row ---
    def fmt(row):
        return "  ".join(f"{str(cell):>{col_widths[i]}}" for i, cell in enumerate(row))

    # --- print header ---
    print(fmt(headers))

    # --- print per-class rows ---
    for row in rows:
        print(fmt(row))

    # --- midrule before averages ---
    print("  ".join("-" * w for w in col_widths))

    # --- print average rows ---
    print(fmt(weighted_row))
    print(fmt(unweighted_row))


def print_tp_fp_fn(tp_dict, fp_dict, fn_dict, iou_thresholds):
    """
    Prints TP, FP, FN counts per class and per IoU threshold.
    """
    print("\nPer-class TP, FP, FN:")
    for c in sorted(tp_dict.keys()):
        label = ID_TO_LABEL.get(c, str(c))
        tps = tp_dict[c]
        fps = fp_dict[c]
        fns = fn_dict[c]
        metrics = ", ".join(
            f"@{thr:.2f}: TP={tp}, FP={fp}, FN={fn}"
            for thr, tp, fp, fn in zip(iou_thresholds, tps, fps, fns)
        )
        print(f"{label}: {metrics}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="configs", config_name="metrics")
def main(cfg: DictConfig):

    gt_file = Path(cfg.paths.scenefun3d_gt_dir) / f"{cfg.scene}.txt"

    if cfg.pseudo_pred:
        pred_masks_path = Path(cfg.paths.map_dir) / "pseudo_aff_masks_laserscan.npy"
        pred_types_path = Path(cfg.paths.map_dir) / "pseudo_aff_types_laserscan.npy"
    else:
        pred_masks_path = Path(cfg.paths.map_dir) / "aff_masks_laserscan.npy"
        pred_types_path = Path(cfg.paths.map_dir) / "aff_types_laserscan.npy"

    # load GT + exclusion mask
    gt_masks, gt_types, exclude_mask = load_gt_masks_and_types(gt_file)
    if cfg.class_agnostic:
        gt_types = np.full_like(gt_types, DUMMY_CLASS_ID)

    # load predictions, then apply exclusion
    pred_masks, pred_types = load_pred_masks_and_types(pred_masks_path, pred_types_path)
    pred_masks[:, exclude_mask] = False
    if cfg.class_agnostic:
        pred_types = np.full_like(pred_types, DUMMY_CLASS_ID)

    # evaluate
    results, counts, tp_counts, fp_counts, fn_counts = evaluate_recall(
        gt_masks, gt_types, pred_masks, pred_types, cfg.iou_thresholds
    )

    # print single Recall table with both weighted and unweighted averages
    print_recall_table(results, counts, cfg.iou_thresholds)

    print_tp_fp_fn(tp_counts, fp_counts, fn_counts, cfg.iou_thresholds)


if __name__ == "__main__":
    main()
