## Evaluating on ConceptNodes Output

This doc provides instructions to run evaluation on the SceneFun3D dataset using maps generated from **ConceptNodes**.

---

### 1. Setup

First, set up the `scenefun3d` environment:

```bash
git clone https://github.com/SceneFun3D/scenefun3d.git
cd scenefun3d

conda create --name scenefun3d python=3.10
conda activate scenefun3d
pip install -r requirements.txt
```

**Dataset paths:**

* Complete dataset: `/data/kumaraditya/scenefun3d` (on uberduck)
* Subset: `/home/kumaraditya/datasets/scenefun3d` (on omegaduck)

---

### 2. Prepare Ground Truth

Before evaluation, you must prepare the SceneFun3D ground truth for the scenes you wish to evaluate. The toolkit expects predictions for all scenes for which GT is prepared.

1. **Edit the scenes list**
   Specify the scenes to evaluate in:

   ```
   benchmark_file_lists/val_scenes.txt
   ```

2. **Prepare GT files:**
   Run the following command (edit the dataset paths as needed):

   ```bash
   python -m eval.functionality_segmentation.prepare_gt_val_data \
       --data_dir /path/to/scenefun3d/val \
       --val_scenes_list ./benchmark_file_lists/val_scenes.txt \
       --out_gt_dir /path/to/scenefun3d/val_gt
   ```

   > **Note:**
   > Ensure you erase any files in the `val_gt` folder before running the script to avoid mixing old and new results.

---

### 3. Align Masks

All paths for evaluation are specified in:
`concept_nodes/configs/paths/paths.yaml`
Make sure to edit these paths before proceeding.

* The config file used for mask alignment is:
  `concept_nodes/configs/masks.yaml`

**To align ConceptNodes affordance predictions to the GT laserscan:**

```bash
python -m concept_nodes.find_aff_masks
```

**To compute masks for pseudo predictions (using GT data):**

```bash
python -m concept_nodes.find_pseudo_aff_masks
```

---

### 4. Metrics

#### AP Metric

Run Average Precision evaluation:

```bash
python -m eval.functionality_segmentation.evaluate \
    --pred_dir /path/to/scenefun3d/val_pred \
    --gt_dir /path/to/scenefun3d/val_gt
```

Replace the `pred_dir` argument with the path to the pseudo predictions (`/path/to/scenefun3d/val_pseudo_pred`) to get AP numbers on these pseudo preds.

#### Instance Recall & Semantic Segmentation

* Config: `concept_nodes/configs/metrics.yaml`
* These metrics run on **one scene at a time** (need to modify the code to run over multiple scenes).

**To compute instance segmentation recall:**

```bash
python -m concept_nodes.metric_instance_recall
```

**To compute semantic segmentation metrics:**

```bash
python -m concept_nodes.metric_semantic_seg
```

---

### 5. Plots

* Config file: `concept_nodes/configs/plots.yaml`

**To get scene statistics:**

```bash
python -m concept_nodes.plot_scene_stats
```

**To plot confusion matrices for recall metrics:**

```bash
python -m concept_nodes.plot_recall_cf
```

---

**Note:**

* Ensure all configuration files (`paths.yaml`, `masks.yaml`, `metrics.yaml`, `plots.yaml`) are updated before running these steps.
* For evaluation across multiple scenes, the recall/sematic seg scripts need to be modified.

