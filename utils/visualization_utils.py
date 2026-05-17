import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import List, Optional
import seaborn as sns


class PointCloudVisualizer:
    """
    Visualize point cloud predictions and comparisons
    """
    
    def __init__(self, class_names: List[str], output_dir: str):
        """
        Initialize visualizer
        
        Args:
            class_names: Names of classes
            output_dir: Directory to save visualizations
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create color map for classes
        self.colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        self.cmap = ListedColormap(self.colors)
    
    def visualize_prediction(
        self,
        points: np.ndarray,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        sample_idx: int
    ):
        """
        Visualize prediction vs ground truth for a single sample
        
        Args:
            points: (N, 3) point coordinates
            prediction: (N,) predicted labels
            ground_truth: (N,) ground truth labels
            sample_idx: Sample index
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Ground truth
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=ground_truth, cmap=self.cmap, s=1, vmin=0, vmax=len(self.class_names)-1
        )
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Prediction
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=prediction, cmap=self.cmap, s=1, vmin=0, vmax=len(self.class_names)-1
        )
        ax2.set_title('Prediction')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Error map (correct = 0, wrong = 1)
        ax3 = fig.add_subplot(133, projection='3d')
        errors = (prediction != ground_truth).astype(int)
        scatter3 = ax3.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=errors, cmap='RdYlGn_r', s=1, vmin=0, vmax=1
        )
        ax3.set_title(f'Errors (Accuracy: {(1-errors.mean())*100:.1f}%)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_ticks(range(len(self.class_names)))
        cbar.set_ticklabels(self.class_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'sample_{sample_idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved: {save_path}")
    
    def plot_confusion_matrix(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        num_classes: int
    ):
        """
        Plot confusion matrix
        
        Args:
            ground_truth: Ground truth labels
            predictions: Predicted labels
            num_classes: Number of classes
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(ground_truth, predictions, labels=range(num_classes))
        
        # Normalize by row (ground truth)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved: {save_path}")
    
    def plot_per_class_iou(self, class_iou: List[float]):
        """
        Plot per-class IoU
        
        Args:
            class_iou: List of IoU scores per class
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(class_iou))
        bars = ax.bar(x, class_iou, color='skyblue', edgecolor='navy')
        
        # Color bars by performance
        for i, (bar, iou) in enumerate(zip(bars, class_iou)):
            if iou < 0.3:
                bar.set_color('red')
            elif iou < 0.5:
                bar.set_color('orange')
            elif iou < 0.7:
                bar.set_color('yellow')
            else:
                bar.set_color('green')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('IoU')
        ax.set_title('Per-Class IoU Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.axhline(y=np.mean(class_iou), color='r', linestyle='--', label=f'Mean IoU: {np.mean(class_iou):.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'per_class_iou.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class IoU plot saved: {save_path}")


def plot_model_comparison(comparison_data: dict, save_path: str):
    """
    Plot comparison between multiple models
    
    Args:
        comparison_data: Dictionary with model comparisons
        save_path: Path to save figure
    """
    models = list(comparison_data.keys())
    metrics = ['overall_accuracy', 'mean_iou', 'mean_accuracy']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [comparison_data[model][metric] for model in models]
        
        axes[idx].bar(models, values, color='skyblue', edgecolor='navy')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model comparison saved: {save_path}")
