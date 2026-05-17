# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

PyTorch benchmark of four point-cloud semantic-segmentation architectures (PointNet, PointNet++ SSG, PointNet++ MSG, PointCNN) on two outdoor LiDAR datasets (Paris-Lille-3D and Toronto-3D), unified to a shared 5-class label space (Ground / Building / Vehicle / Vegetation / Unclassified). Companion paper is under review at ISPRS Journal — Manuscript ID `ISRS-D-25-00980R1`.

## Common commands

```bash
# Preprocess raw PLYs → unified 5-class segments
python preprocess_datasets.py --dataset Toronto-3D     --data_dir data/
python preprocess_datasets.py --dataset Paris-Lille-3D --data_dir data/
# Or both with custom split sizes:
python preprocess_datasets.py --dataset both --data_dir data/ \
    --num_segments 16 --train_segments 14 --val_segments 1 --test_segments 1

# Train — edit config.py first; there are no CLI flags
python train.py

# Visual inference on a chosen split (default test)
python test.py --split test     # also: train | val
```

There is **no test suite, no linter, no build step** in this repo. Don't invent one. Iteration is `edit config.py → python train.py`.

## How model/dataset choice works

You change models and datasets **only by editing [config.py](config.py)** — never by editing `train.py` or `test.py`. The relevant fields:

- `dataset_name`: `"paris_lille_3d"` or `"toronto_3d"` → `Config.dataset_path` resolves to `paris_lille_path` or `toronto_3d_path`.
- `model_type`: `"pointnet"`, `"pointnet2"`, `"pointnet2_msg"`, `"pointcnn"` → `train.build_model(cfg)` and `test.get_model(...)` dispatch on this string.

When adding a new model, add a new branch to **both** dispatchers ([train.py:32-56](train.py#L32-L56) and [test.py:45-65](test.py#L45-L65)) — they are duplicated, intentionally.

## Architectural facts that aren't obvious from filenames

### `dataset.py` — three-mode block generator

`PreprocessedDataset` reads PLY segments and slices them into spatial blocks. On the **train** split (when `use_multi_view_blocks=True`, which is the default in `get_dataloaders`), it builds blocks via three modes and concatenates them:

1. `_build_equal_grid_blocks` — regular XY grid using `(block_size, stride)`.
2. `_build_random_shape_blocks` — random overlapping XY rectangles, sizes drawn from `random_shape_config` (independent of `block_size`).
3. `_build_multi_scale_blocks` — random square windows with sizes in `multi_scale_block_sizes` (default `(1.0, 100.0)`).

Val / test always use mode 1 only, with `stride = block_size` (non-overlapping tiling).

### Block cache

`get_dataloaders(config)` passes `load=config.load_blocks` into each `PreprocessedDataset`. When `True`, the dataset writes/reads `<split>_blocks_cache.pt` inside each split dir. **The cache is keyed only by split name — not by `block_size`, `stride`, or augmentation config.** If you change any of those, delete the `*_blocks_cache.pt` files manually or set `load_blocks=False` for one run. The `.gitignore` excludes these caches.

### Logit-shape polymorphism

Models return different shapes:
- PointNet → tuple `(logits [B,N,C], trans [B,3,3], trans_feat [B,64,64])`
- PointNet++ SSG / MSG / PointCNN → tensor `[B,N,C]`

`train.unpack_model_output` and `test.logits_to_preds` handle both cases. `train.flatten_logits_and_labels` additionally supports `[B,C,N]` and `[B,C]` shapes by matching dimensions against `config.num_classes`. **If you add a model, prefer returning `[B,N,C]` to match the majority convention.**

### PointNet-only regularizer

`feature_transform_regularizer(trans_feat)` is added to the loss **only when** `config.model_type.lower() == "pointnet"` AND `trans_feat is not None` AND `config.feature_transform_reg > 0` ([train.py:185-191](train.py#L185-L191)). Don't accidentally enable it for the other models — `trans_feat` will be `None` so the check guards it, but the gate is also keyed on `model_type`.

### `query_ball_point` is k-NN, not radius

[models/utils.py:73-95](models/utils.py#L73-L95) — the `radius` argument is **ignored**. Grouping is always `argsort`-based k-NN over `nsample` neighbors. All `radius=` values in `pointnet2.py` / `pointnet2_msg.py` / `pointcnn.py` are kept for PointNet++ API compatibility only. **Don't suggest "tuning the radius"** to anyone — it does nothing.

### Class-imbalance handling stack (all three layered together)

Outdoor LiDAR is dominated by Ground + Unclassified. The pipeline counters this with three independent mechanisms:

1. **Per-class weights** — computed in `PreprocessedDataset._calculate_class_weights` as `(max_freq / freq_c)^(1/3)`, passed to the loss when `config.use_class_weights=True`.
2. **Block-level WeightedRandomSampler** — in `get_dataloaders`, each train block's weight is `class_weights[unique_labels].max()`, biasing toward blocks containing rare classes.
3. **Loss choice** — `config.loss_function` ∈ `{"ce", "focal", "combo"}`; `combo` (CE + Dice) is the default.

Helpers for swapping in alternative weighting schemes (`inv`, `balanced_inv`, `median_freq`, `effective_num`) live in [utils/class_weights.py](utils/class_weights.py) but are not wired in by default.

### Label remapping

Both datasets get collapsed to 5 unified classes in [preprocess_datasets.py](preprocess_datasets.py) via `PARIS_LILLE_10_TO_5_MAP` and `Toronto_3D_LABEL_MAP`. The preprocessed PLYs have a `label` field that is **already** in unified 0-4 space — the model never sees the raw 10-class / fine-grained labels.

## Known broken / weird stuff (don't "fix" without asking)

- **`Config.use_normization`** (typo of *normalization*) is intentional and referenced throughout `dataset.py` and `get_dataloaders`. Don't rename it without sweeping all references.
- **`test.py` imports `utils.test_dataset.get_dataloaders`** but that module does not exist in the repo. `test.py` will fail at import time. The fix is either to alias `utils.test_dataset` to `dataset.get_dataloaders` or to change the import to `from dataset import get_dataloaders`. **Do not silently rewrite this — ask the user which they want.**
- **PointNet returns a 3-tuple, others return a tensor** — see "Logit-shape polymorphism" above. Don't normalize to a single return type without auditing every caller.

## W&B / logging

`train.py` calls `wandb.init(project="pointcloud-semseg", ...)` unconditionally. If running offline, the user is expected to either `wandb login` first or comment out the `wandb.init` / `wandb.log` / `wandb.save` lines. Checkpointing to disk works regardless. Best mIoU on validation triggers a save to `config.checkpoint_dir/<dataset>_<model>_best_epoch<N>.pth`.

## Inference outputs

`test.py` → `InferenceManager` writes everything under `./inference_results/<dataset>/<model>/<YYYYMMDD_HHMMSS>/`:
- `<split>_full_prediction.ply` — ascii PLY with `x y z red green blue pred_label true_label` columns (colors come from a 9-color built-in palette indexed by predicted class).
- `<split>_vis_metrics.json` — overall accuracy, mIoU, mean accuracy, per-class IoU/accuracy, full confusion matrix.
- `summary.json` — experiment metadata + per-class metrics.

This directory is `.gitignore`d.

## Repo conventions

- Branch: `master` (not `main`). The remote is `origin`.
- Commit author identity is set at repo level only, not global.
- Result figures in [results/](results/) are committed and linked from the README — don't gitignore them.
- The companion paper title and BibTeX live in [README.md](README.md#citation). When the paper gets a DOI, update the `@article` entry there.
