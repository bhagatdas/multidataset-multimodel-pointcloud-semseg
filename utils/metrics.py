import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def compute_metrics(labels, predictions, num_classes):
    """
    Compute semantic segmentation metrics
    
    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    overall_accuracy = accuracy_score(labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    
    # Per-class metrics
    class_iou = []
    class_accuracy = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-10)
        class_iou.append(iou)
        
        # Accuracy
        acc = tp / (tp + fn + 1e-10)
        class_accuracy.append(acc)
    
    # Mean metrics
    mean_iou = np.mean(class_iou)
    mean_accuracy = np.mean(class_accuracy)
    
    return {
        'overall_accuracy': overall_accuracy,
        'mean_iou': mean_iou,
        'mean_accuracy': mean_accuracy,
        'class_iou': class_iou,
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm
    }


def print_metrics(metrics, class_names=None):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<20} {'IoU':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for i, (iou, acc) in enumerate(zip(metrics['class_iou'], metrics['class_accuracy'])):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name:<20} {iou:<10.4f} {acc:<10.4f}")
    
    print("="*60 + "\n")
