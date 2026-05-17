import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import warnings

# 🔇 Put filters BEFORE importing config to catch Pydantic warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\._internal\._generate_schema",
)

from config import config
from utils.test_dataset import get_dataloaders
from utils.metrics import compute_metrics, print_metrics
from utils.inference_utils import InferenceManager


# ----------------------------
# Checkpoint utils
# ----------------------------
def get_latest_checkpoint(checkpoint_dir: str) -> Path:
    """Automatically find the latest .pth checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    pth_files = sorted(
        checkpoint_dir.glob("*.pth"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not pth_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_ckpt = pth_files[0]
    print(f"✅ Found latest checkpoint: {latest_ckpt.name}")
    return latest_ckpt


# ----------------------------
# Model factory
# ----------------------------
def get_model(model_type: str, num_classes: int) -> torch.nn.Module:
    """Return the correct model based on config.model_type."""
    mt = model_type.lower()

    if mt == "pointnet":
        from models.pointnet import PointNetSemSeg
        return PointNetSemSeg(num_classes=num_classes)

    if mt == "pointnet2":
        from models.pointnet2 import PointNet2SemSeg
        return PointNet2SemSeg(num_classes=num_classes)

    if mt == "pointnet2_msg":
        from models.pointnet2_msg import PointNet2MSGSemSeg
        return PointNet2MSGSemSeg(num_classes=num_classes)

    if mt == "pointcnn":
        from models.pointcnn import PointCNNSemSeg
        return PointCNNSemSeg(num_classes=num_classes)

    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------
# Logits → predictions helper
# ----------------------------
def logits_to_preds(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Robustly convert logits to per-point predictions.
    Supports:
      - [B, N, C]
      - [B, C, N]
      - [B*N, C] (reshaped to labels)
      - Tuple/list outputs from PointNet: (logits, trans, trans_feat)
    """
    num_classes = config.num_classes

    # 🔑 Handle PointNet-style outputs: (logits, trans, trans_feat)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.dim() == 3:
        b, d1, d2 = logits.shape
        if d2 == num_classes:           # [B, N, C]
            return logits.argmax(dim=-1)
        if d1 == num_classes:           # [B, C, N]
            return logits.permute(0, 2, 1).argmax(dim=-1)
        raise ValueError(
            f"Cannot infer class dim from logits shape {logits.shape} "
            f"(num_classes={num_classes})"
        )

    if logits.dim() == 2 and logits.shape[1] == num_classes:
        return logits.argmax(dim=-1).view_as(labels)

    raise ValueError(
        f"Unsupported logits shape {logits.shape} for num_classes={num_classes}"
    )


# ----------------------------
# Visualization / inference loop
# ----------------------------
def run_vis(model, loader, device):
    """
    Run inference for visualization:
    - Keeps original points
    - Centers XYZ block-wise for the model (like training)
    - Returns flattened points, labels, predictions
    """
    model.eval()
    all_points, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Visual Testing"):
            # Expect: (points, labels, [mean, std]) but handle minimal case too.
            if len(batch) >= 2:
                points, labels = batch[:2]
            else:
                raise ValueError("Batch must contain at least (points, labels).")

            # Keep original for saving / visualization
            original_points = points.clone()

            # Block-wise centering on XYZ for the model
            centered_points = original_points.clone()
            centered_points[..., :3] = (
                centered_points[..., :3]
                - centered_points[..., :3].mean(dim=1, keepdim=True)
            )

            centered_points = centered_points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(centered_points)
            pred_choice = logits_to_preds(logits, labels)

            all_points.append(original_points.cpu().numpy())
            all_preds.append(pred_choice.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_points = np.concatenate(all_points, axis=0)   # [B, N, D]
    all_preds = np.concatenate(all_preds, axis=0)     # [B, N]
    all_labels = np.concatenate(all_labels, axis=0)   # [B, N]

    num_blocks, num_points, feat_dim = all_points.shape
    flat_points = all_points.reshape(-1, feat_dim)
    flat_preds = all_preds.reshape(-1)
    flat_labels = all_labels.reshape(-1)

    return flat_points, flat_labels, flat_preds


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to run VIS inference on.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if getattr(config, "use_cuda", True) and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    print(f"Loading {args.split} data for visualization...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(config)
    loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }[args.split]

    # Latest checkpoint
    checkpoint_path = get_latest_checkpoint(config.checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", None)
    else:
        state_dict = checkpoint
        epoch = None

    # Build model
    model = get_model(config.model_type, config.num_classes)

    # Handle DataParallel checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    if epoch is not None:
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print("Loaded latest checkpoint (no epoch info).")

    # Inference manager
    inference_manager = InferenceManager(
        model_name=config.model_type,
        dataset_name=config.dataset_name,
        checkpoint_path=str(checkpoint_path),
        config=config,
    )

    print(
        f"Running VIS inference on '{args.split}' split "
        f"with model='{config.model_type}', dataset='{config.dataset_name}'"
    )

    points, labels, predictions = run_vis(model, loader, device)

    # Optional: metrics (nice to see even for vis)
    print("\nComputing metrics (for reference)...")
    metrics = compute_metrics(labels, predictions, config.num_classes)
    print_metrics(metrics, config.class_names)

    # Save metrics
    metrics_file = inference_manager.output_dir / f"{args.split}_vis_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "overall_accuracy": float(metrics["overall_accuracy"]),
                "mean_iou": float(metrics["mean_iou"]),
                "mean_accuracy": float(metrics["mean_accuracy"]),
                "class_iou": [float(v) for v in metrics["class_iou"]],
                "class_accuracy": [float(v) for v in metrics["class_accuracy"]],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=4,
        )
    print(f"✓ VIS metrics saved to: {metrics_file}")

    # Extra outputs: visualization artifacts
    inference_manager.save_comparison_data(metrics, split=args.split)
    print("\nSaving merged prediction PLY file for visualization...")
    inference_manager.save_full_ply(points, predictions, labels, split=args.split)

    print("\n" + "=" * 80)
    print("VIS Inference Complete!")
    print(f"Visualization results saved to: {inference_manager.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
