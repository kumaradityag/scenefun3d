import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

from utils.data_parser import DataParser
from eval.functionality_segmentation.eval_utils import util_3d
from eval.functionality_segmentation.eval_utils.benchmark_labels import (
    CLASS_LABELS,
    VALID_CLASS_IDS,
)

# map id to name
ID_TO_LABEL = {vid: lbl for vid, lbl in zip(VALID_CLASS_IDS, CLASS_LABELS)}


def load_gt_masks_and_types(gt_file: Path):
    gt_ids = util_3d.load_ids(gt_file)
    exclude = util_3d.get_excluded_point_mask(gt_ids)
    clean = gt_ids.copy()
    clean[exclude] = 0
    inst = util_3d.get_instances(clean, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    masks, types = [], []
    for lst in inst.values():
        for o in lst:
            masks.append(clean == o["instance_id"])
            types.append(o["label_id"])
    return np.array(masks, bool), np.array(types, int), exclude


def load_pred_masks_and_types(masks_path: Path, types_path: Path):
    aff_masks = np.load(masks_path).astype(bool)
    aff_types = np.load(types_path).astype(int)
    return aff_masks, aff_types


class AffordanceViewer:
    def __init__(self, gt_dir, map_dir, sf3d_dir, scene):
        # define an explicit ALL AFF
        self.ALL_AFF = -1

        # load point cloud
        dp = DataParser(sf3d_dir)
        pcd = dp.get_laser_scan(scene)
        self.main_pcd = dp.get_cropped_laser_scan(scene, pcd)

        # load masks
        ground_truth_masks, ground_truth_types, _ = load_gt_masks_and_types(
            Path(gt_dir) / f"{scene}.txt"
        )
        predicted_masks, predicted_types = load_pred_masks_and_types(
            Path(map_dir) / "aff_masks_laserscan.npy",
            Path(map_dir) / "aff_types_laserscan.npy",
        )

        # remove excluded points entirely (crop masks to match main_pcd)
        valid_points = dp.get_crop_mask(scene)
        ground_truth_masks = ground_truth_masks[:, valid_points]
        predicted_masks = predicted_masks[:, valid_points]

        # Debug print
        print(
            f"Main PCD points: {len(self.main_pcd.points)}, "
            f"Ground Truth Masks shape: {ground_truth_masks.shape}, "
            f"Predicted Masks shape: {predicted_masks.shape}"
        )

        # group by type
        self.type_indices = {}
        all_ground_truth_indices, all_predicted_indices = set(), set()
        total_points = len(self.main_pcd.points)

        for mask, type_id in zip(ground_truth_masks, ground_truth_types):
            idxs = np.nonzero(mask)[0]
            self.type_indices.setdefault(
                type_id, {"ground_truth": set(), "predicted": set()}
            )
            self.type_indices[type_id]["ground_truth"].update(idxs)
            all_ground_truth_indices.update(idxs)

        for mask, type_id in zip(predicted_masks, predicted_types):
            idxs = np.nonzero(mask)[0]
            self.type_indices.setdefault(
                type_id, {"ground_truth": set(), "predicted": set()}
            )
            self.type_indices[type_id]["predicted"].update(idxs)
            all_predicted_indices.update(idxs)

        # add explicit ALL affordance entry
        self.type_indices[self.ALL_AFF] = {
            "ground_truth": all_ground_truth_indices,
            "predicted": all_predicted_indices,
        }

        # compute intersection and only-gt, only-predicted sets
        for type_id, sets in self.type_indices.items():
            gt_set = sets["ground_truth"]
            pred_set = sets["predicted"]
            inter = gt_set & pred_set
            sets["intersection"] = inter
            sets["ground_truth"] = gt_set - inter
            sets["predicted"] = pred_set - inter

        # affordance types list
        self.affordance_types = [self.ALL_AFF] + sorted(
            [tid for tid in self.type_indices if tid != self.ALL_AFF]
        )

        # viewer state
        self.current_affordance = None
        self.main_name = "pcd_main"
        self.affordance_rgb_name = "pcd_aff_rgb"
        self.affordance_semantic_name = "pcd_aff_sem"
        self.visible_names = set()

        # build GUI
        self._build_gui()

    def _build_gui(self):
        self.app = gui.Application.instance
        self.app.initialize()
        self.vis = o3d.visualization.O3DVisualizer("Affordances", 1024, 768)
        self.vis.set_background([1.0, 1.0, 1.0, 1.0], bg_image=None)
        self.vis.show_skybox(False)
        self.vis.enable_raw_mode(True)
        self.vis.show_settings = True

        # add main point cloud
        self.vis.add_geometry(self.main_name, self.main_pcd)
        self.visible_names.add(self.main_name)

        # buttons
        self.vis.add_action("Main", self._cb_main)
        self.vis.add_action("All Aff", self._make_affordance_callback(self.ALL_AFF))
        for type_id in self.affordance_types:
            if type_id == self.ALL_AFF:
                continue
            lbl = ID_TO_LABEL[type_id]
            self.vis.add_action(lbl, self._make_affordance_callback(type_id))

        self.vis.add_action("Toggle Aff RGB", self._cb_aff_rgb)
        self.vis.add_action("Toggle Aff Pred GT", self._cb_aff_sem)
        self.vis.reset_camera_to_default()

        self.app.add_window(self.vis)
        self.app.run()

    def _update(self, show_main, show_rgb, show_semantic):
        # clear all geometries (including fast variants)
        for name in list(self.visible_names):
            self.vis.remove_geometry(name)
            self.vis.remove_geometry(f"{name}.__fast__")
        self.visible_names.clear()

        # re-add requested geometries
        if show_main:
            self.vis.add_geometry(self.main_name, self.main_pcd)
            self.visible_names.add(self.main_name)
        if show_rgb and hasattr(self, "affordance_rgb_pcd"):
            self.vis.add_geometry(self.affordance_rgb_name, self.affordance_rgb_pcd)
            self.visible_names.add(self.affordance_rgb_name)
        if show_semantic and hasattr(self, "affordance_semantic_pcd"):
            self.vis.add_geometry(
                self.affordance_semantic_name, self.affordance_semantic_pcd
            )
            self.visible_names.add(self.affordance_semantic_name)

    def _load_aff(self, affordance_type):
        sets = self.type_indices[affordance_type]
        gt_idxs = sorted(sets["ground_truth"])
        pred_idxs = sorted(sets["predicted"])
        inter_idxs = sorted(sets["intersection"])

        total = len(self.main_pcd.points)
        used = set(gt_idxs) | set(pred_idxs) | set(inter_idxs)
        rest = sorted(set(range(total)) - used)

        pcd_gt = self.main_pcd.select_by_index(gt_idxs)
        pcd_pred = self.main_pcd.select_by_index(pred_idxs)
        pcd_inter = self.main_pcd.select_by_index(inter_idxs)
        pcd_rest = self.main_pcd.select_by_index(rest)

        # NOTE: If you want to visualize the remaining points in gray, uncomment the line below
        # pcd_rest.paint_uniform_color([0.7, 0.7, 0.7])

        # RGB view
        self.affordance_rgb_pcd = pcd_gt + pcd_pred + pcd_inter
        # NOTE: If you want to see the remaining points, uncomment the line below
        # self.affordance_rgb_pcd += pcd_rest

        # semantic view
        pcd_gt.paint_uniform_color([0, 0, 1])
        pcd_pred.paint_uniform_color([1, 0, 0])
        pcd_inter.paint_uniform_color([0, 1, 0])

        self.affordance_semantic_pcd = pcd_gt + pcd_pred + pcd_inter
        # NOTE: If you want to see the remaining points, uncomment the line below
        # self.affordance_semantic_pcd += pcd_rest

        save_dir = Path("/home/kumaraditya/sf3d_aff_viz")
        save_dir.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(
            save_dir / f"affordance_semantic_{affordance_type}.pcd",
            self.affordance_semantic_pcd,
        )

        self.current_affordance = affordance_type

    def _make_affordance_callback(self, affordance_type):
        def callback(vis):
            self._load_aff(affordance_type)
            # default to semantic view on new selection
            self._update(show_main=False, show_rgb=False, show_semantic=True)

        return callback

    def _cb_main(self, vis):
        self._update(show_main=True, show_rgb=False, show_semantic=False)

    def _cb_aff_rgb(self, vis):
        if not hasattr(self, "affordance_rgb_pcd"):
            return
        self._update(show_main=False, show_rgb=True, show_semantic=False)

    def _cb_aff_sem(self, vis):
        if not hasattr(self, "affordance_semantic_pcd"):
            return
        self._update(show_main=False, show_rgb=False, show_semantic=True)


@hydra.main(version_base=None, config_path="configs", config_name="viz")
def main(cfg: DictConfig):
    scenefun3d_gt_dir = cfg.paths.scenefun3d_gt_dir
    map_dir = cfg.paths.map_dir
    scenefun3d_dir = cfg.paths.scenefun3d_dir
    scene = cfg.scene

    AffordanceViewer(
        scenefun3d_gt_dir,
        map_dir,
        scenefun3d_dir,
        scene,
    )


if __name__ == "__main__":
    main()
