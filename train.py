import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb
from config import config
from dataset import get_dataloaders
from models.losses import FocalLoss, ComboLoss

warnings.filterwarnings("ignore")

# ----------------------------
# Device utils
# ----------------------------
def get_device():
    if torch.cuda.is_available() and getattr(config, "use_cuda", True):
        print("Using CUDA:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# ----------------------------
# Model factory
# ----------------------------
def build_model(cfg) -> nn.Module:
    """
    Return the correct model based on cfg.model_type.
    You only change cfg.model_type; no train.py edits needed.
    Adjust imports/class names here to match your actual implementations.
    """
    mt = cfg.model_type.lower()

    if mt == "pointnet":
        from models.pointnet import PointNetSemSeg
        return PointNetSemSeg(num_classes=cfg.num_classes)

    if mt == "pointnet2":
        from models.pointnet2 import PointNet2SemSeg
        return PointNet2SemSeg(num_classes=cfg.num_classes)

    if mt == "pointnet2_msg":
        from models.pointnet2_msg import PointNet2MSGSemSeg
        return PointNet2MSGSemSeg(num_classes=cfg.num_classes)

    if mt == "pointcnn":
        from models.pointcnn import PointCNNSemSeg
        return PointCNNSemSeg(num_classes=cfg.num_classes)

    raise ValueError(f"Unknown model_type: {cfg.model_type}")


# ----------------------------
# Metrics
# ----------------------------
def per_class_iou(conf_matrix: np.ndarray):
    num_classes = conf_matrix.shape[0]
    ious = []
    for i in range(num_classes):
        intersection = conf_matrix[i, i]
        union = (
            conf_matrix[i, :].sum()
            + conf_matrix[:, i].sum()
            - intersection
        )
        ious.append(0.0 if union == 0 else float(intersection) / float(union))
    return ious


def flatten_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor):
    """
    Support multiple model output shapes:
    - [B, N, C]
    - [B, C, N]
    - [B, C] (already flat per-point)
    """
    if logits.dim() == 3:
        b, d1, d2 = logits.shape
        # guess which dim is classes
        if d2 == config.num_classes:
            # [B, N, C]
            logits_flat = logits.reshape(-1, config.num_classes)
        elif d1 == config.num_classes:
            # [B, C, N] -> [B, N, C]
            logits_flat = logits.permute(0, 2, 1).reshape(-1, config.num_classes)
        else:
            raise ValueError(
                f"Cannot infer class dim from logits shape {logits.shape}; "
                f"expected one dim to be num_classes={config.num_classes}."
            )
    elif logits.dim() == 2 and logits.shape[1] == config.num_classes:
        logits_flat = logits
    else:
        raise ValueError(
            f"Unsupported logits shape {logits.shape} for num_classes={config.num_classes}"
        )

    if labels.dim() > 1:
        labels_flat = labels.reshape(-1)
    else:
        labels_flat = labels

    return logits_flat, labels_flat


# ----------------------------
# PointNet helpers (safe for others)
# ----------------------------
def unpack_model_output(output):
    """
    Handle both:
    - models that return logits
    - PointNet that returns (logits, trans, trans_feat)
    """
    if isinstance(output, (tuple, list)):
        logits = output[0]
        trans = output[1] if len(output) > 1 else None
        trans_feat = output[2] if len(output) > 2 else None
        return logits, trans, trans_feat
    else:
        return output, None, None


def feature_transform_regularizer(trans: torch.Tensor):
    """
    Standard PointNet feature transform regularizer.
    trans: [B, K, K]
    """
    if trans is None:
        return 0.0

    B, K, _ = trans.shape
    I = torch.eye(K, device=trans.device, dtype=trans.dtype).unsqueeze(0)  # [1, K, K]
    mat = torch.bmm(trans, trans.transpose(2, 1))  # [B, K, K]
    loss = torch.mean(torch.norm(mat - I, dim=(1, 2)))
    return loss


# ----------------------------
# Training / Validation
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, use_amp):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc=f"Training Epoch {epoch + 1}", leave=False):
        # Expect: (points, labels, *extras)
        points, labels, *rest = batch

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            output = model(points)
            logits, trans, trans_feat = unpack_model_output(output)

            # Optional sanity check on logits shape
            if logits.dim() == 3:
                if logits.size(-1) != config.num_classes and logits.size(1) != config.num_classes:
                    raise ValueError(
                        f"Logits shape {logits.shape} does not match num_classes={config.num_classes}"
                    )
            elif logits.dim() == 2:
                if logits.size(1) != config.num_classes:
                    raise ValueError(
                        f"Logits shape {logits.shape} does not match num_classes={config.num_classes}"
                    )

            logits_flat, labels_flat = flatten_logits_and_labels(logits, labels)
            loss = criterion(logits_flat, labels_flat)

            # PointNet feature transform regularization
            # Only applied when:
            #   - model_type is 'pointnet'
            #   - trans_feat is not None
            reg_lambda = getattr(config, "feature_transform_reg", 0.0)
            if (
                reg_lambda > 0.0
                and trans_feat is not None
                and config.model_type.lower() == "pointnet"
            ):
                loss = loss + reg_lambda * feature_transform_regularizer(trans_feat)

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        preds = logits.argmax(-1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    conf = confusion_matrix(
        all_labels.flatten(),
        all_preds.flatten(),
        labels=list(range(config.num_classes)),
    )
    per_class_acc = conf.diagonal() / (conf.sum(axis=1) + 1e-6)
    per_class_iou_vals = per_class_iou(conf)

    return running_loss / len(loader), per_class_acc, per_class_iou_vals


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, use_amp):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc=f"Validation Epoch {epoch + 1}", leave=False):
        points, labels, *rest = batch

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            output = model(points)
            logits, trans, trans_feat = unpack_model_output(output)

            logits_flat, labels_flat = flatten_logits_and_labels(logits, labels)
            loss = criterion(logits_flat, labels_flat)

        val_loss += float(loss.item())
        preds = logits.argmax(-1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    conf = confusion_matrix(
        all_labels.flatten(),
        all_preds.flatten(),
        labels=list(range(config.num_classes)),
    )
    per_class_acc = conf.diagonal() / (conf.sum(axis=1) + 1e-6)
    per_class_iou_vals = per_class_iou(conf)

    return val_loss / len(loader), per_class_acc, per_class_iou_vals


# ----------------------------
# Main
# ----------------------------
def main():
    device = get_device()
    use_amp = bool(getattr(config, "mixed_precision", False) and device.type == "cuda")

    # DataLoader creation uses cfg.dataset_name / cfg.dataset_path internally
    train_loader, val_loader, test_loader, train_class_weights = get_dataloaders(config)

    # W&B
    wandb.init(
        project="pointcloud-semseg",
        config={
            "dataset": getattr(config, "dataset_name", None),
            "model": getattr(config, "model_type", None),
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "num_points": config.num_points,
            "learning_rate": config.learning_rate,
            "loss_function": config.loss_function,
            "block_size": getattr(config, "block_size", None),
            "stride": getattr(config, "stride", None),
            "use_fps": getattr(config, "use_fps", False),
            "use_augmentation": getattr(config, "use_augmentation", True),
            "use_normization": getattr(config, "use_normization", False),
        },
        name=f"{config.dataset_name}_{config.model_type}",
    )

    # Model
    model = build_model(config).to(device)
    if getattr(config, "multi_gpu", False) and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # Loss / class weights
    if getattr(config, "use_class_weights", False) and train_class_weights is not None:
        class_weights = train_class_weights.to(device=device, dtype=torch.float)
    else:
        class_weights = None

    if config.loss_function == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
    elif config.loss_function == "combo":
        criterion = ComboLoss(alpha=config.combo_alpha, weights=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    # Scheduler
    if config.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
        )
    else:
        scheduler = None

    best_miou = 0.0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}]")

        train_loss, train_acc, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_amp
        )
        val_loss, val_acc, val_iou = validate(
            model, val_loader, criterion, device, epoch, use_amp
        )

        if scheduler is not None:
            scheduler.step()

        mean_iou = float(np.mean(val_iou))

        # Log to wandb
        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_iou": mean_iou,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        for i in range(config.num_classes):
            cls_name = (
                config.class_names[i]
                if hasattr(config, "class_names") and len(config.class_names) > i
                else f"class_{i}"
            )
            log[f"train_acc_{cls_name}"] = float(train_acc[i])
            log[f"train_iou_{cls_name}"] = float(train_iou[i])
            log[f"val_acc_{cls_name}"] = float(val_acc[i])
            log[f"val_iou_{cls_name}"] = float(val_iou[i])

        wandb.log(log)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Per-class IoU:", [f"{iou:.3f}" for iou in val_iou])
        print("Per-class Acc:", [f"{acc:.3f}" for acc in val_acc])

        # Checkpointing on best mIoU
        if mean_iou > best_miou:
            best_miou = mean_iou
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            save_path = os.path.join(
                config.checkpoint_dir,
                f"{config.dataset_name}_{config.model_type}_best_epoch{epoch + 1}.pth",
            )
            torch.save(
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                save_path,
            )
            try:
                wandb.save(save_path)
            except OSError as e:
                print(f"Warning: wandb.save failed (likely symlink issue on some platforms): {e}")

            print(f"New best model saved at {save_path}")

    wandb.finish()
    print("\nTraining complete!")
    print(f"Best mIoU: {best_miou:.3f}")


if __name__ == "__main__":
    main()
