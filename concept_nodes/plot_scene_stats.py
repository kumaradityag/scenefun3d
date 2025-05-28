"""
Compute basic statistics (instance counts + vertex-count histograms) for
functional-affordance predictions vs. ground truth and save pastel-coloured charts.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Constants & canonical skill names
# -----------------------------------------------------------------------------
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
CANONICAL_NAMES = [s.lower().replace(" ", "_") for s in SKILL_LIST]  # e.g. "key_press"
SKILL_TO_IDX = {s: i + 1 for i, s in enumerate(SKILL_LIST)}
IDX_TO_CANON = {i + 1: name for i, name in enumerate(CANONICAL_NAMES)}


# -----------------------------------------------------------------------------
#  Loading helpers
# -----------------------------------------------------------------------------
def load_predictions(
    pred_dir: Path,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Returns
    -------
    counts   : {canon_name: [vertex_count, …]}
    vert_cnt : same dictionary, but the value is a list of vertex counts (one per instance)
    """
    masks_path = pred_dir / "aff_masks_laserscan.npy"
    types_path = pred_dir / "aff_types_laserscan.npy"

    if not masks_path.exists() or not types_path.exists():
        raise FileNotFoundError("Prediction files not found in %s" % pred_dir)

    masks = np.load(masks_path)  # (N_inst, N_pts), bool / {0,1}
    types = np.load(types_path)  # (N_inst,)  with ints 1-9

    counts: Dict[str, List[int]] = {name: [] for name in CANONICAL_NAMES}
    for mask, t in zip(masks, types):
        canon = IDX_TO_CANON.get(int(t))  # unknown labels are ignored
        if canon is None:
            continue
        counts[canon].append(int(mask.sum()))

    return counts


def load_ground_truth(gt_file: Path) -> Dict[str, List[int]]:
    """
    Convert the gt_instances dict (as produced by util_3d.get_instances) into
    {canon_name: [vertex_count, …]}.
    """
    from eval.functionality_segmentation.eval_utils import util_3d

    gt_ids = util_3d.load_ids(gt_file)
    exclude_mask = util_3d.get_excluded_point_mask(gt_ids)
    gt_ids = gt_ids[np.logical_not(exclude_mask)]
    from eval.functionality_segmentation.eval_utils.benchmark_labels import (
        CLASS_LABELS,
        VALID_CLASS_IDS,
    )

    ID_TO_LABEL = {vid: lbl for vid, lbl in zip(VALID_CLASS_IDS, CLASS_LABELS)}
    gt_instances = util_3d.get_instances(
        gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL
    )

    counts: Dict[str, List[int]] = {name: [] for name in CANONICAL_NAMES}
    for raw_key, inst_list in gt_instances.items():
        canon = raw_key.strip().lower()  # already e.g. "key_press"
        if canon in counts:
            counts[canon].extend([int(d["vert_count"]) for d in inst_list])
    return counts


# -----------------------------------------------------------------------------
#  Statistics helpers
# -----------------------------------------------------------------------------
def tally_instances(counts: Dict[str, List[int]]) -> Dict[str, int]:
    return {k: len(v) for k, v in counts.items()}


def flatten_vertices(counts: Dict[str, List[int]]) -> List[int]:
    merged: List[int] = []
    for lst in counts.values():
        merged.extend(lst)
    return merged


# -----------------------------------------------------------------------------
#  Plotting helpers
# -----------------------------------------------------------------------------
PALETTE = sns.color_palette()


def _write_safe(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_bar(counts: Dict[str, int], title: str, out: Path):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    keys = [k.replace("_", " ").title() for k in CANONICAL_NAMES]
    values = [counts.get(k, 0) for k in CANONICAL_NAMES]

    sns.barplot(x=keys, y=values, palette=PALETTE[: len(keys)], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_ylabel("# instances")
    ax.set_title(title)

    # ─── total on top ───
    total = sum(values)
    ax.text(
        0.99,
        0.95,
        f"Total = {total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="medium",
        weight="bold",
    )
    _write_safe(fig, out)


def plot_bar_comparison(gt: Dict[str, int], pred: Dict[str, int], out: Path):
    sns.set_style("whitegrid")
    labels = [k.replace("_", " ").title() for k in CANONICAL_NAMES]
    gt_vals = [gt.get(k, 0) for k in CANONICAL_NAMES]
    pr_vals = [pred.get(k, 0) for k in CANONICAL_NAMES]

    x, width = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, gt_vals, width, label="GT", color=PALETTE[0])
    ax.bar(x + width / 2, pr_vals, width, label="Pred", color=PALETTE[1])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("# instances")
    ax.set_title("Affordance counts – GT vs. Prediction")
    ax.legend()

    ax.text(
        0.99,
        0.95,
        f"GT total = {sum(gt_vals)}    Pred total = {sum(pr_vals)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="medium",
        weight="bold",
    )
    _write_safe(fig, out)


# -----------------------------------------------------------------------------
#   Histograms, tidy overflow bucket
#  Fixed-width buckets: 0-50 … 950-1000 + overflow (>1000)
# --------------------------------------------------------------------------
STEP = 50
MAX_EDGE = 1000  # upper edge of last regular bin
EDGES = np.arange(0, MAX_EDGE + STEP, STEP)  # 0, 50, 100, … 1000
OVERFLOW_CENTER = MAX_EDGE + STEP / 2  # x-coord for the >1000 bar


def _annotate_avg(ax, vals):
    if vals:
        ax.text(
            0.97,
            0.85,
            f"avg ≈ {np.mean(vals):.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="small",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7),
        )


def _hist_fixed_bins(ax, values, color):
    """Histogram with 50-vertex buckets and one overflow bar (>1000)."""
    values = np.asarray(values)
    if not values.size:
        return

    in_range = values[values <= MAX_EDGE]
    overflow_n = (values > MAX_EDGE).sum()

    # regular bars
    ax.hist(in_range, bins=EDGES, color=color)

    # overflow bar
    ax.bar(OVERFLOW_CENTER, overflow_n, width=STEP, color=color, align="center")

    # x-axis ticks & labels
    tick_pos = list(EDGES[:-1]) + [OVERFLOW_CENTER]
    tick_label = [f"{int(l)}" for l in EDGES[:-1]] + [f">{MAX_EDGE}"]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_label, rotation=40, ha="right")


def plot_histograms(
    gt_vert: Dict[str, List[int]],
    pr_vert: Dict[str, List[int]],
    out_dir: Path,
):
    """
    • One histogram per affordance (GT top / Pred bottom)
    • One combined histogram for all affordances
    • Bins: 0-50 … 950-1000, >1000
    """
    sns.set_style("whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    for canon in CANONICAL_NAMES:
        vals_gt, vals_pr = gt_vert.get(canon, []), pr_vert.get(canon, [])
        if not vals_gt and not vals_pr:
            continue

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
        _hist_fixed_bins(axs[0], vals_gt, PALETTE[2])  # GT
        _hist_fixed_bins(axs[1], vals_pr, PALETTE[3])  # Pred

        # ─── make y-axis equal ───
        y_max = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
        for ax in axs:
            ax.set_ylim(0, y_max)

        # titles, labels, averages (same as before)
        axs[0].set_title(f"{canon.replace('_', ' ').title()} – GT")
        axs[1].set_title(f"{canon.replace('_', ' ').title()} – Pred")
        axs[1].set_xlabel("Vertex count (last bin = >1000)")
        for ax, vals in zip(axs, (vals_gt, vals_pr)):
            ax.set_ylabel("# instances")
            _annotate_avg(ax, vals)

        _write_safe(fig, out_dir / f"hist_{canon}.png")

    all_gt, all_pr = flatten_vertices(gt_vert), flatten_vertices(pr_vert)
    if not all_gt and not all_pr:
        return

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    _hist_fixed_bins(axs[0], all_gt, PALETTE[2])  # GT
    _hist_fixed_bins(axs[1], all_pr, PALETTE[3])  # Pred

    y_max = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    for ax in axs:
        ax.set_ylim(0, y_max)

    axs[0].set_title("All affordances – GT")
    axs[1].set_title("All affordances – Pred")
    axs[1].set_xlabel("Vertex count (last bin = >1000)")
    for ax, vals in zip(axs, (all_gt, all_pr)):
        ax.set_ylabel("# instances")
        _annotate_avg(ax, vals)

    _write_safe(fig, out_dir / "all_affordances_combined.png")


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="configs", config_name="plots")
def main(cfg: DictConfig):

    pred_dir = Path(cfg.paths.map_dir)
    gt_file = Path(cfg.paths.scenefun3d_gt_dir) / f"{cfg.scene}.txt"
    save_dir = Path(cfg.paths.stat_plots_dir) / str(cfg.scene)
    fig_dir = save_dir / "figures"
    hist_dir = save_dir / "histograms"

    # Load data
    pred_vert = load_predictions(pred_dir)
    gt_vert = load_ground_truth(gt_file)

    # Counts
    counts_pred = tally_instances(pred_vert)
    counts_gt = tally_instances(gt_vert)

    # Print counts per class and total
    print("Counts per class:")
    for canon in CANONICAL_NAMES:
        readable = canon.replace("_", " ").title()
        print(
            f"{readable}: GT={counts_gt.get(canon,0)}, Pred={counts_pred.get(canon,0)}"
        )
    print(f"Total: GT={sum(counts_gt.values())}, Pred={sum(counts_pred.values())}")

    # Bar plots
    plot_bar(counts_gt, "Affordance counts – Ground Truth", fig_dir / "counts_gt.png")
    plot_bar(counts_pred, "Affordance counts – Prediction", fig_dir / "counts_pred.png")
    plot_bar_comparison(counts_gt, counts_pred, fig_dir / "counts_comparison.png")

    # Histograms
    plot_histograms(gt_vert, pred_vert, hist_dir)

    print(f"Figures saved under {save_dir.resolve()}")


if __name__ == "__main__":
    main()
