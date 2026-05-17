import torch
from typing import Iterable, List, Optional, Tuple, Dict


def _accumulate_label_counts(
    data_blocks: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
) -> torch.Tensor:
    """
    data_blocks: iterable of (points, labels)
        - labels can be any shape; will be flattened.
    """
    label_counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, lbls in data_blocks:
        lbls = lbls.view(-1)
        valid = (lbls >= 0) & (lbls < num_classes)
        lbls = lbls[valid]
        if lbls.numel() == 0:
            continue

        unique, counts = torch.unique(lbls, return_counts=True)
        label_counts[unique] += counts

    return label_counts


# 1. Simple inverse frequency: w_c = 1 / (freq_c)
def inverse_frequency_weights(label_counts: torch.Tensor) -> torch.Tensor:
    freqs = label_counts / (label_counts.sum() + 1e-6)
    return 1.0 / (freqs + 1e-6)


# 2. Balanced inverse frequency (your first implementation):
#    w_c = total / (K * count_c)
def balanced_inverse_frequency_weights(label_counts: torch.Tensor) -> torch.Tensor:
    num_classes = label_counts.numel()
    total = label_counts.sum()
    return total / (num_classes * (label_counts + 1e-6))


# 3. Power inverse frequency (your beta-style version):
#    w_c = (max_freq / freq_c) ** beta
def power_inverse_frequency_weights(
    label_counts: torch.Tensor,
    beta: float = 1.0 / 3.0,
) -> torch.Tensor:
    total_points = label_counts.sum()
    freqs = label_counts / (total_points + 1e-6)
    max_freq = freqs.max()
    return (max_freq / (freqs + 1e-6)) ** beta


# 4. Median frequency balancing (standard seg trick):
#    w_c = median_freq / freq_c
def median_frequency_balancing_weights(label_counts: torch.Tensor) -> torch.Tensor:
    total_points = label_counts.sum()
    freqs = label_counts / (total_points + 1e-6)
    median_freq = freqs[freqs > 0].median() if (freqs > 0).any() else torch.tensor(1.0)
    return median_freq / (freqs + 1e-6)


# 5. Effective number of samples (Cui et al. CVPR19):
#    w_c = (1 - beta) / (1 - beta^{n_c})
def effective_number_weights(
    label_counts: torch.Tensor,
    beta: float = 0.999,
) -> torch.Tensor:
    # Avoid zero-count issues
    counts = label_counts.clone().float()
    counts[counts == 0] = 1.0

    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-6)
    # Normalize so that mean weight ~ 1
    return weights / (weights.mean() + 1e-6)


def compute_class_weights(
    data_blocks: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    mode: str = "balanced_inv",
    class_names: Optional[List[str]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    High-level helper:
    - mode in {
        'inv', 'balanced_inv', 'power_inv',
        'median_freq', 'effective_num'
      }
    - kwargs:
        - beta for 'power_inv' or 'effective_num'
    """
    label_counts = _accumulate_label_counts(data_blocks, num_classes)

    if mode == "inv":
        weights = inverse_frequency_weights(label_counts)
    elif mode == "balanced_inv":
        weights = balanced_inverse_frequency_weights(label_counts)
    elif mode == "power_inv":
        beta = float(kwargs.get("beta", 1.0 / 3.0))
        weights = power_inverse_frequency_weights(label_counts, beta=beta)
    elif mode == "median_freq":
        weights = median_frequency_balancing_weights(label_counts)
    elif mode == "effective_num":
        beta = float(kwargs.get("beta", 0.999))
        weights = effective_number_weights(label_counts, beta=beta)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    # Optional: pretty print with class names (e.g. Ground, Building, ...)
    if class_names is not None and len(class_names) == num_classes:
        print("Class weights:")
        for i, (name, w) in enumerate(zip(class_names, weights)):
            print(f"  [{i}] {name:<12}: {float(w):.6f}")
    else:
        print(f"Class weights: {weights}")

    return weights



# from class_weights import compute_class_weights

# def _calculate_class_weights(self):
#     self.class_weights = compute_class_weights(
#         self.data_blocks,
#         num_classes=self.num_classes,
#         mode="balanced_inv",          # or 'power_inv', 'median_freq', 'effective_num', 'inv'
#         class_names=getattr(self, "class_names", None),
#         beta=1.0/3.0,                 # used for 'power_inv' or 'effective_num' if selected
#     )
