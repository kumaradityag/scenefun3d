#!/usr/bin/env python3
import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="configs", config_name="plots")
def plot_confusion_matrices(cfg: DictConfig):
    # Extract config values
    input_dir = cfg.paths.results_dir
    colormap = cfg.recall_plot.colormap

    # Filenames
    csv_name = "tp_fp_fn.csv"
    output_dir_name = "plots"

    # Load CSV
    csv_path = os.path.join(input_dir, csv_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Identify thresholds and classes
    thresh_cols = [c for c in df.columns if "@" in c]
    thresholds = sorted({col.split("@")[1] for col in thresh_cols}, key=float)
    classes = df["Class"].tolist()

    # Prepare output directory
    out_dir = os.path.join(input_dir, output_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Per-class plots ---
    for i, cls in enumerate(classes):
        # Create one-row grid for this class
        n_thresh = len(thresholds)
        fig, axes = plt.subplots(1, n_thresh, figsize=(4 * n_thresh, 4))
        if n_thresh == 1:
            axes = [axes]

        # Determine row max for color scaling
        row_max = max(
            int(df.loc[i, f"TP@{th}"])
            + int(df.loc[i, f"FP@{th}"])
            + int(df.loc[i, f"FN@{th}"])
            for th in thresholds
        )

        for j, th in enumerate(thresholds):
            tp = int(df.loc[i, f"TP@{th}"])
            fp = int(df.loc[i, f"FP@{th}"])
            fn = int(df.loc[i, f"FN@{th}"])
            tn = 0
            cm = np.array([[tp, fp], [fn, tn]])

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            ax = axes[j]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cbar=True,
                cmap=colormap,
                # vmin=0,
                # vmax=row_max,
                xticklabels=["Actual Pos", "Actual Neg"],
                yticklabels=["Pred Pos", "Pred Neg"],
                ax=ax,
            )
            ax.set_title(f"{cls} @ IoU {th}")

            # Annotate counts
            n_pred = tp + fp
            n_gt = tp + fn
            ax.text(
                0.5,
                -0.2,
                f"Pred: {n_pred}\nGT: {n_gt}\nRecall: {recall:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
            )

        plt.tight_layout()
        per_path = os.path.join(out_dir, f"confusion_{cls}.png")
        plt.savefig(per_path, dpi=300)
        plt.close(fig)
        print(f"Saved per-class plot for '{cls}': {per_path}")

    # --- Combined plot across all classes ---
    # Sum metrics for each threshold
    sum_tp = {th: df[f"TP@{th}"].sum() for th in thresholds}
    sum_fp = {th: df[f"FP@{th}"].sum() for th in thresholds}
    sum_fn = {th: df[f"FN@{th}"].sum() for th in thresholds}

    n_thresh = len(thresholds)
    fig, axes = plt.subplots(1, n_thresh, figsize=(4 * n_thresh, 4))
    if n_thresh == 1:
        axes = [axes]

    # Determine max for combined row
    comb_max = max(sum_tp[th] + sum_fp[th] + sum_fn[th] for th in thresholds)

    for j, th in enumerate(thresholds):
        tp, fp, fn = sum_tp[th], sum_fp[th], sum_fn[th]
        tn = 0
        cm = np.array([[tp, fp], [fn, tn]])

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        ax = axes[j]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cbar=True,
            cmap=colormap,
            # vmin=0,
            # vmax=comb_max,
            xticklabels=["Actual Pos", "Actual Neg"],
            yticklabels=["Pred Pos", "Pred Neg"],
            ax=ax,
        )
        ax.set_title(f"Combined @ IoU {th}")

        # Annotate combined counts
        ax.text(
            0.5,
            -0.2,
            f"Pred: {tp+fp}\nGT: {tp+fn}\nRecall: {recall:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    plt.tight_layout()
    comb_path = os.path.join(out_dir, "confusion_combined.png")
    plt.savefig(comb_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined plot: {comb_path}")


if __name__ == "__main__":
    plot_confusion_matrices()
