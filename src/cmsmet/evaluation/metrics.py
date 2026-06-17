"""
Evaluation Metrics for Cross-Modal Speech-Music Emotion Transfer

CRITICAL: Reports metrics that match SER publication standards:
- UA (Unweighted Accuracy): Primary metric for imbalanced datasets
- WA (Weighted Accuracy): Secondary metric for completeness
- Confusion Matrix: Shows per-class performance
- Per-class F1: Reveals minority class learning
- Statistical significance: Tracks across multiple seeds
"""
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
from typing import Dict, List, Tuple, Optional
import json


class EmotionMetrics:
    """Compute emotion recognition metrics following SER standards"""
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        """
        Args:
            num_classes: Number of emotion classes (4 for standard IEMOCAP)
            class_names: List of class names (e.g., ['happy', 'sad', 'angry', 'neutral'])
        """
        self.num_classes = num_classes
        
        if class_names is None:
            if num_classes == 4:
                class_names = ['happy', 'sad', 'angry', 'neutral']
            elif num_classes == 5:
                class_names = ['happy', 'sad', 'angry', 'neutral', 'frustrated']
            else:
                class_names = [f'class_{i}' for i in range(num_classes)]
        
        self.class_names = class_names
    
    def compute_metrics(
        self,
        predictions: np.ndarray,  # shape: (N,)
        targets: np.ndarray,      # shape: (N,)
    ) -> Dict:
        """
        Compute comprehensive metrics following SER standards
        
        Args:
            predictions: Predicted class indices (0-3)
            targets: Ground truth class indices (0-3)
        
        Returns:
            Dictionary with:
            - ua: Unweighted Accuracy (primary metric)
            - wa: Weighted Accuracy (secondary metric)
            - conf_matrix: Confusion matrix
            - per_class_f1: F1 score per class
            - per_class_precision: Precision per class
            - per_class_recall: Recall per class
            - overall_f1: Macro-averaged F1
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        
        # Ensure integer type
        predictions = predictions.astype(int)
        targets = targets.astype(int)
        
        # CRITICAL: UA vs WA
        # UA = unweighted accuracy (same weight for all classes)
        # WA = weighted accuracy (weight by class frequency) - can hide minority class failure
        per_class_accuracies = []
        for class_idx in range(self.num_classes):
            class_mask = targets == class_idx
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == targets[class_mask]).mean()
                per_class_accuracies.append(class_acc)
            else:
                per_class_accuracies.append(0.0)
        
        ua = np.mean(per_class_accuracies)
        wa = accuracy_score(targets, predictions)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions, labels=list(range(self.num_classes)))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions,
            labels=list(range(self.num_classes)),
            zero_division=0,
        )
        
        # Class imbalance analysis
        class_distribution = np.bincount(targets, minlength=self.num_classes)
        
        return {
            'ua': float(ua),  # Unweighted Accuracy (PRIMARY)
            'wa': float(wa),  # Weighted Accuracy
            'conf_matrix': conf_matrix,
            'per_class_f1': {self.class_names[i]: float(f1[i]) for i in range(self.num_classes)},
            'per_class_precision': {self.class_names[i]: float(precision[i]) for i in range(self.num_classes)},
            'per_class_recall': {self.class_names[i]: float(recall[i]) for i in range(self.num_classes)},
            'per_class_support': {self.class_names[i]: int(support[i]) for i in range(self.num_classes)},
            'macro_f1': float(f1_score(targets, predictions, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(targets, predictions, average='micro', zero_division=0)),
            'class_distribution': {self.class_names[i]: int(class_distribution[i]) for i in range(self.num_classes)},
        }
    
    def format_metrics_for_table(self, metrics: Dict) -> str:
        """Format metrics as table for paper/report"""
        lines = []
        lines.append("=" * 70)
        lines.append("EMOTION RECOGNITION PERFORMANCE (SER Standards)")
        lines.append("=" * 70)
        lines.append(f"UA (Unweighted Accuracy): {metrics['ua']:.4f}")
        lines.append(f"WA (Weighted Accuracy):   {metrics['wa']:.4f}")
        lines.append(f"Macro F1:                 {metrics['macro_f1']:.4f}")
        lines.append("")
        lines.append("Per-Class Performance:")
        lines.append("-" * 70)
        lines.append(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
        lines.append("-" * 70)
        
        for class_name in self.class_names:
            prec = metrics['per_class_precision'].get(class_name, 0.0)
            rec = metrics['per_class_recall'].get(class_name, 0.0)
            f1 = metrics['per_class_f1'].get(class_name, 0.0)
            supp = metrics['per_class_support'].get(class_name, 0)
            lines.append(f"{class_name:<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {supp:<12d}")
        
        lines.append("-" * 70)
        lines.append("")
        lines.append("Class Distribution:")
        lines.append("-" * 70)
        for class_name in self.class_names:
            count = metrics['class_distribution'].get(class_name, 0)
            total = sum(metrics['class_distribution'].values())
            pct = 100.0 * count / total if total > 0 else 0.0
            lines.append(f"{class_name:<12} {count:>5d} ({pct:>5.1f}%)")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def format_confusion_matrix(self, metrics: Dict) -> str:
        """Format confusion matrix for paper/report"""
        conf = metrics['conf_matrix']
        lines = []
        lines.append("\nConfusion Matrix:")
        lines.append("-" * 70)
        
        # Header
        header = "Predicted".ljust(12)
        for class_name in self.class_names:
            header += f"{class_name:<10}"
        lines.append(header)
        lines.append("-" * 70)
        
        # Rows
        for true_idx, class_name in enumerate(self.class_names):
            row = class_name.ljust(12)
            for pred_idx in range(self.num_classes):
                count = conf[true_idx, pred_idx]
                row += f"{count:<10d}"
            lines.append(row)
        
        lines.append("-" * 70)
        return "\n".join(lines)


class ExperimentResults:
    """Track results across multiple seeds with statistical significance"""
    
    def __init__(self, name: str, num_seeds: int = 3):
        """
        Args:
            name: Experiment name (e.g., 'Baseline', 'Contrastive-Frozen')
            num_seeds: Number of random seeds to use
        """
        self.name = name
        self.num_seeds = num_seeds
        self.results = []  # List of dicts from each seed
    
    def add_result(self, metrics: Dict, seed: int):
        """Add result from one seed"""
        result = metrics.copy()
        result['seed'] = seed
        self.results.append(result)
    
    def get_statistics(self) -> Dict:
        """Compute mean ± std across seeds"""
        if not self.results:
            return {}
        
        ua_values = [r['ua'] for r in self.results]
        wa_values = [r['wa'] for r in self.results]
        macro_f1_values = [r['macro_f1'] for r in self.results]
        
        # Collect per-class F1 across seeds
        per_class_f1_stats = {}
        for class_name in self.results[0]['per_class_f1'].keys():
            f1_values = [r['per_class_f1'][class_name] for r in self.results]
            per_class_f1_stats[class_name] = {
                'mean': float(np.mean(f1_values)),
                'std': float(np.std(f1_values)),
            }
        
        return {
            'name': self.name,
            'num_seeds': len(self.results),
            'ua_mean': float(np.mean(ua_values)),
            'ua_std': float(np.std(ua_values)),
            'wa_mean': float(np.mean(wa_values)),
            'wa_std': float(np.std(wa_values)),
            'macro_f1_mean': float(np.mean(macro_f1_values)),
            'macro_f1_std': float(np.std(macro_f1_values)),
            'per_class_f1': per_class_f1_stats,
        }
    
    def format_statistical_summary(self) -> str:
        """Format results with significance for paper"""
        stats = self.get_statistics()
        if not stats:
            return "[No results]"
        
        lines = []
        lines.append(f"\n{self.name} (N={stats['num_seeds']} seeds):")
        lines.append("-" * 70)
        lines.append(f"UA: {stats['ua_mean']:.4f} ± {stats['ua_std']:.4f}")
        lines.append(f"WA: {stats['wa_mean']:.4f} ± {stats['wa_std']:.4f}")
        lines.append(f"Macro F1: {stats['macro_f1_mean']:.4f} ± {stats['macro_f1_std']:.4f}")
        lines.append("")
        lines.append("Per-Class F1 (Mean ± Std):")
        for class_name in sorted(stats['per_class_f1'].keys()):
            f1_stat = stats['per_class_f1'][class_name]
            lines.append(f"  {class_name}: {f1_stat['mean']:.4f} ± {f1_stat['std']:.4f}")
        
        return "\n".join(lines)


def create_ablation_result_table(experiments: List[ExperimentResults]) -> str:
    """
    Create ablation study table for paper (SER standards)
    
    Shows: Model | Pretrain | Data | Contrastive | UA | WA | Macro F1
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("ABLATION STUDY: Cross-Modal Speech-Music Emotion Transfer")
    lines.append("=" * 100)
    lines.append(f"{'Experiment':<25} {'Pretraining':<15} {'Data':<20} {'Contrastive':<15} {'UA':<12} {'WA':<12} {'Macro F1':<12}")
    lines.append("-" * 100)
    
    for exp in experiments:
        stats = exp.get_statistics()
        if not stats:
            continue
        
        # Parse experiment name to extract ablation parameters
        # Expected format: "Model-Pretrain-Data-Loss"
        ua_str = f"{stats['ua_mean']:.4f}±{stats['ua_std']:.4f}"
        wa_str = f"{stats['wa_mean']:.4f}±{stats['wa_std']:.4f}"
        f1_str = f"{stats['macro_f1_mean']:.4f}±{stats['macro_f1_std']:.4f}"
        
        lines.append(f"{stats['name']:<25} {'baseline':<15} {'mixed':<20} {'none':<15} {ua_str:<12} {wa_str:<12} {f1_str:<12}")
    
    lines.append("=" * 100)
    return "\n".join(lines)


def report_significance_warning(ua_std: float, improvement_pct: float) -> str:
    """Check if results are statistically significant"""
    lines = []
    
    if improvement_pct < ua_std:
        lines.append("\n[WARNING] Improvement may not be statistically significant!")
        lines.append(f"  Improvement: {improvement_pct:.2f}%")
        lines.append(f"  Std Dev:     {ua_std:.2f}%")
        lines.append("  A paper showing improvements smaller than ±1 std will be rejected.")
    
    return "\n".join(lines)


# Export as JSON for reproducibility
def export_results_json(experiments: List[ExperimentResults], filepath: str):
    """Export all results to JSON for supplementary materials"""
    export_data = {
        'experiments': []
    }
    
    for exp in experiments:
        exp_data = {
            'name': exp.name,
            'num_seeds': len(exp.results),
            'statistics': exp.get_statistics(),
            'seed_results': [
                {
                    'seed': r['seed'],
                    'ua': r['ua'],
                    'wa': r['wa'],
                    'macro_f1': r['macro_f1'],
                    'per_class_f1': r['per_class_f1'],
                }
                for r in exp.results
            ]
        }
        export_data['experiments'].append(exp_data)
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Results exported to {filepath}")
