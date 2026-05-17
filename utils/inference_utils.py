import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict

class InferenceManager:
    """
    Manages inference results and saving as a single PLY for visualization.
    """
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        checkpoint_path: str,
        config: Any,
        base_dir: str = "./inference_results",
        save_individual_batches: bool = False
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.checkpoint_path = checkpoint_path
        self.config = config

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{model_name}_{dataset_name}_{timestamp}"

        self.output_dir = Path(base_dir) / dataset_name / model_name / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Inference Manager Initialized")
        print(f"Output Directory: {self.output_dir}")

    def save_full_ply(
        self, 
        points: np.ndarray, 
        pred_labels: np.ndarray, 
        true_labels: np.ndarray, 
        split: str = 'test', 
        color_map: Optional[List[List[int]]] = None
    ):
        """
        Save the full test set as a colored PLY file with both predicted and true labels.
        """
        if points.ndim > 2:
            pts = points.reshape(-1, 3)
        else:
            pts = points
        preds = pred_labels.flatten()
        trues = true_labels.flatten()

        if color_map is None:
            color_map = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 0, 0], [0, 128, 0], [0, 0, 128]
            ]

        ply_file = self.output_dir / f'{split}_full_prediction.ply'
        with open(ply_file, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {pts.shape[0]}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('property uchar pred_label\n')
            f.write('property uchar true_label\n')
            f.write('end_header\n')
            for i in range(pts.shape[0]):
                x, y, z = pts[i]
                plabel = int(preds[i])
                tlabel = int(trues[i])
                color = color_map[plabel % len(color_map)]
                f.write(f'{x} {y} {z} {color[0]} {color[1]} {color[2]} {plabel} {tlabel}\n')
        print(f"✓ Full test prediction saved as PLY: {ply_file}")

    def save_comparison_data(self, metrics: Dict[str, Any], split: str):
        comparison_data = {
            'experiment_id': self.experiment_id,
            'model': self.model_name,
            'dataset': self.dataset_name,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'summary_metrics': {
                'overall_accuracy': float(metrics['overall_accuracy']),
                'mean_iou': float(metrics['mean_iou']),
                'mean_accuracy': float(metrics['mean_accuracy'])
            },
            'per_class_metrics': {}
        }
        for i, (iou, acc) in enumerate(zip(metrics['class_iou'], metrics['class_accuracy'])):
            class_name = self.config.class_names[i] if hasattr(self.config, "class_names") and i < len(self.config.class_names) else f'class_{i}'
            comparison_data['per_class_metrics'][class_name] = {
                'iou': float(iou),
                'accuracy': float(acc)
            }
        comparison_file = self.output_dir / 'summary.json'
        with open(comparison_file, 'w') as f:
            import json
            json.dump(comparison_data, f, indent=4)
        print(f"✓ Comparison summary saved: {comparison_file}")
