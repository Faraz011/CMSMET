"""
Generate paper-quality figures from CMSMET experiment results.

Reads experiment_results.json and produces:
  1. per_fold_comparison.png   — grouped bar chart: baseline vs contrastive per fold
  2. per_class_accuracy.png    — grouped bar chart: per-class accuracy comparison
  3. confusion_matrices.png    — side-by-side normalized confusion matrices
  4. radar_chart.png           — radar overlay of per-class accuracy
  5. summary_panel.png         — composite 2×2 panel of the above (single figure for paper)

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --results-json outputs/paper_results/experiment_results.json
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Segoe UI', 'Helvetica', 'Arial'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

EMOTION_NAMES = ['Happy', 'Sad', 'Angry', 'Neutral']

# Professional colour palette
C_BASE = '#5B8FB9'       # Steel blue
C_CONT = '#E07A5F'       # Terra cotta
C_BASE_LIGHT = '#a0c4e0'
C_CONT_LIGHT = '#f0b8a6'
C_BG = '#FAFAFA'
C_GRID = '#E0E0E0'


# ============================================================================
# HELPER: extract per-class stats from results dict
# ============================================================================

def per_class_stats(results, model_type):
    """Return (means, stds) arrays of shape (4,)."""
    all_pc = []
    for fold in results[model_type]['folds']:
        for sr in fold['seed_results']:
            all_pc.append(sr['per_class_acc'])
    all_pc = np.array(all_pc)
    return all_pc.mean(axis=0), all_pc.std(axis=0)


# ============================================================================
# FIGURE 1 — Per-Fold UA Comparison (grouped bar chart)
# ============================================================================

def plot_per_fold(results, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    folds = [f"Fold {i+1}\n(Ses {results['baseline']['folds'][i]['session']})"
             for i in range(5)]
    base_means = [f['ua_mean'] for f in results['baseline']['folds']]
    base_stds  = [f['ua_std']  for f in results['baseline']['folds']]
    cont_means = [f['ua_mean'] for f in results['contrastive']['folds']]
    cont_stds  = [f['ua_std']  for f in results['contrastive']['folds']]

    x = np.arange(len(folds))
    w = 0.32

    bars1 = ax.bar(x - w/2, base_means, w, yerr=base_stds, capsize=4,
                   color=C_BASE, edgecolor='white', linewidth=0.8,
                   label='Baseline (random proj.)', zorder=3,
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})
    bars2 = ax.bar(x + w/2, cont_means, w, yerr=cont_stds, capsize=4,
                   color=C_CONT, edgecolor='white', linewidth=0.8,
                   label='Contrastive (pretrained proj.)', zorder=3,
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})

    # Value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f'{h:.1%}', ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel('Unweighted Accuracy (UA)')
    ax.set_title('Per-Fold LOSO Comparison: Baseline vs. Contrastive', fontweight='bold', pad=12)
    ax.set_ylim(0.40, 0.80)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis='y', color=C_GRID, linewidth=0.7, zorder=0)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#CCC')

    # Overall annotation
    b_ua = results['baseline']['overall_ua_mean']
    c_ua = results['contrastive']['overall_ua_mean']
    delta = c_ua - b_ua
    p_val = results['significance']['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    note = f'ΔUA = {delta:+.2%}  (p = {p_val:.3f}, {sig})'
    ax.text(0.5, 0.02, note, transform=ax.transAxes, ha='center',
            fontsize=10, fontstyle='italic', color='#555')

    plt.tight_layout()
    path = out_dir / 'per_fold_comparison.png'
    plt.savefig(path)
    plt.close()
    print(f"  [OK] {path}")
    return path


# ============================================================================
# FIGURE 2 — Per-Class Accuracy Comparison
# ============================================================================

def plot_per_class(results, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    b_means, b_stds = per_class_stats(results, 'baseline')
    c_means, c_stds = per_class_stats(results, 'contrastive')

    x = np.arange(4)
    w = 0.32

    bars1 = ax.bar(x - w/2, b_means, w, yerr=b_stds, capsize=4,
                   color=C_BASE, edgecolor='white', linewidth=0.8,
                   label='Baseline', zorder=3,
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})
    bars2 = ax.bar(x + w/2, c_means, w, yerr=c_stds, capsize=4,
                   color=C_CONT, edgecolor='white', linewidth=0.8,
                   label='Contrastive', zorder=3,
                   error_kw={'linewidth': 1.2, 'capthick': 1.2})

    # Delta annotations
    for i in range(4):
        delta = c_means[i] - b_means[i]
        max_h = max(b_means[i] + b_stds[i], c_means[i] + c_stds[i])
        color = '#2E7D32' if delta > 0 else '#C62828'
        ax.text(x[i], max_h + 0.025, f'{delta:+.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    # Value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f'{h:.1%}', ha='center', va='bottom', fontsize=8, color='#444')

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_NAMES)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy Comparison', fontweight='bold', pad=12)
    ax.set_ylim(0.30, 0.85)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis='y', color=C_GRID, linewidth=0.7, zorder=0)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#CCC')

    plt.tight_layout()
    path = out_dir / 'per_class_accuracy.png'
    plt.savefig(path)
    plt.close()
    print(f"  [OK] {path}")
    return path


# ============================================================================
# FIGURE 3 — Confusion Matrices (side-by-side)
# ============================================================================

def plot_confusion_matrices(results, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(C_BG)

    for idx, model_type in enumerate(['baseline', 'contrastive']):
        total_cm = np.zeros((4, 4))
        for fold in results[model_type]['folds']:
            for sr in fold['seed_results']:
                total_cm += np.array(sr['confusion_matrix'])
        row_sums = total_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_cm = total_cm / row_sums

        ax = axes[idx]
        ax.set_facecolor(C_BG)
        im = ax.imshow(norm_cm, cmap='Blues', vmin=0, vmax=1, aspect='equal')

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        emo_lower = [e.lower() for e in EMOTION_NAMES]
        ax.set_xticklabels(emo_lower, fontsize=10)
        ax.set_yticklabels(emo_lower, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)

        title = 'Baseline (random proj.)' if model_type == 'baseline' else 'Contrastive (pretrained proj.)'
        ua = results[model_type]['overall_ua_mean']
        ax.set_title(f'{title}\nUA = {ua:.4f}', fontsize=12, fontweight='bold')

        for i in range(4):
            for j in range(4):
                color = 'white' if norm_cm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{norm_cm[i,j]:.2f}', ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout(w_pad=3)
    path = out_dir / 'confusion_matrices.png'
    plt.savefig(path)
    plt.close()
    print(f"  [OK] {path}")
    return path


# ============================================================================
# FIGURE 4 — Radar Chart
# ============================================================================

def plot_radar(results, out_dir: Path):
    b_means, _ = per_class_stats(results, 'baseline')
    c_means, _ = per_class_stats(results, 'contrastive')

    labels = EMOTION_NAMES
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    b_vals = b_means.tolist() + [b_means[0]]
    c_vals = c_means.tolist() + [c_means[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.plot(angles, b_vals, 'o-', linewidth=2, color=C_BASE, label='Baseline', markersize=7)
    ax.fill(angles, b_vals, alpha=0.15, color=C_BASE)
    ax.plot(angles, c_vals, 's-', linewidth=2, color=C_CONT, label='Contrastive', markersize=7)
    ax.fill(angles, c_vals, alpha=0.15, color=C_CONT)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0.40, 0.75)
    ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_yticklabels(['45%', '50%', '55%', '60%', '65%', '70%'], fontsize=8, color='#666')
    ax.set_title('Per-Class Accuracy Profile', fontweight='bold', pad=20, fontsize=13)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), framealpha=0.9, edgecolor='#CCC')

    plt.tight_layout()
    path = out_dir / 'radar_chart.png'
    plt.savefig(path)
    plt.close()
    print(f"  [OK] {path}")
    return path


# ============================================================================
# FIGURE 5 — Composite 2×2 Summary Panel
# ============================================================================

def plot_summary_panel(results, out_dir: Path):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(C_BG)

    # ── PANEL A: Per-Fold bars ──
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor(C_BG)
    folds_labels = [f"Fold {i+1}" for i in range(5)]
    base_means = [f['ua_mean'] for f in results['baseline']['folds']]
    base_stds  = [f['ua_std']  for f in results['baseline']['folds']]
    cont_means = [f['ua_mean'] for f in results['contrastive']['folds']]
    cont_stds  = [f['ua_std']  for f in results['contrastive']['folds']]
    x = np.arange(5)
    w = 0.32
    ax1.bar(x - w/2, base_means, w, yerr=base_stds, capsize=3,
            color=C_BASE, edgecolor='white', label='Baseline', zorder=3,
            error_kw={'linewidth': 1})
    ax1.bar(x + w/2, cont_means, w, yerr=cont_stds, capsize=3,
            color=C_CONT, edgecolor='white', label='Contrastive', zorder=3,
            error_kw={'linewidth': 1})
    for i in range(5):
        for val, xpos in [(base_means[i], x[i]-w/2), (cont_means[i], x[i]+w/2)]:
            ax1.text(xpos, val + 0.012, f'{val:.1%}', ha='center', fontsize=7.5, fontweight='bold', color='#333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds_labels)
    ax1.set_ylabel('UA')
    ax1.set_title('(a) Per-Fold LOSO Comparison', fontweight='bold')
    ax1.set_ylim(0.42, 0.78)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.grid(axis='y', color=C_GRID, linewidth=0.5, zorder=0)
    ax1.legend(fontsize=9, framealpha=0.9)

    # ── PANEL B: Per-Class bars ──
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor(C_BG)
    b_means, b_stds = per_class_stats(results, 'baseline')
    c_means, c_stds = per_class_stats(results, 'contrastive')
    x2 = np.arange(4)
    ax2.bar(x2 - w/2, b_means, w, yerr=b_stds, capsize=3,
            color=C_BASE, edgecolor='white', label='Baseline', zorder=3,
            error_kw={'linewidth': 1})
    ax2.bar(x2 + w/2, c_means, w, yerr=c_stds, capsize=3,
            color=C_CONT, edgecolor='white', label='Contrastive', zorder=3,
            error_kw={'linewidth': 1})
    for i in range(4):
        delta = c_means[i] - b_means[i]
        max_h = max(b_means[i] + b_stds[i], c_means[i] + c_stds[i])
        color = '#2E7D32' if delta > 0 else '#C62828'
        ax2.text(x2[i], max_h + 0.02, f'{delta:+.1%}', ha='center', fontsize=9,
                 fontweight='bold', color=color)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(EMOTION_NAMES)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Per-Class Accuracy', fontweight='bold')
    ax2.set_ylim(0.30, 0.85)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.grid(axis='y', color=C_GRID, linewidth=0.5, zorder=0)
    ax2.legend(fontsize=9, framealpha=0.9)

    # ── PANEL C: Confusion Matrix — Baseline ──
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor(C_BG)
    total_cm = np.zeros((4, 4))
    for fold in results['baseline']['folds']:
        for sr in fold['seed_results']:
            total_cm += np.array(sr['confusion_matrix'])
    row_sums = total_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_cm_b = total_cm / row_sums
    im3 = ax3.imshow(norm_cm_b, cmap='Blues', vmin=0, vmax=1)
    ax3.set_xticks(range(4)); ax3.set_yticks(range(4))
    emo_lower = [e.lower() for e in EMOTION_NAMES]
    ax3.set_xticklabels(emo_lower); ax3.set_yticklabels(emo_lower)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')
    b_ua = results['baseline']['overall_ua_mean']
    ax3.set_title(f'(c) Baseline — UA = {b_ua:.2%}', fontweight='bold')
    for i in range(4):
        for j in range(4):
            color = 'white' if norm_cm_b[i, j] > 0.5 else 'black'
            ax3.text(j, i, f'{norm_cm_b[i,j]:.2f}', ha='center', va='center',
                     color=color, fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # ── PANEL D: Confusion Matrix — Contrastive ──
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor(C_BG)
    total_cm = np.zeros((4, 4))
    for fold in results['contrastive']['folds']:
        for sr in fold['seed_results']:
            total_cm += np.array(sr['confusion_matrix'])
    row_sums = total_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_cm_c = total_cm / row_sums
    im4 = ax4.imshow(norm_cm_c, cmap='Oranges', vmin=0, vmax=1)
    ax4.set_xticks(range(4)); ax4.set_yticks(range(4))
    ax4.set_xticklabels(emo_lower); ax4.set_yticklabels(emo_lower)
    ax4.set_xlabel('Predicted'); ax4.set_ylabel('True')
    c_ua = results['contrastive']['overall_ua_mean']
    ax4.set_title(f'(d) Contrastive — UA = {c_ua:.2%}', fontweight='bold')
    for i in range(4):
        for j in range(4):
            color = 'white' if norm_cm_c[i, j] > 0.5 else 'black'
            ax4.text(j, i, f'{norm_cm_c[i,j]:.2f}', ha='center', va='center',
                     color=color, fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # ── Suptitle with key result ──
    delta = c_ua - b_ua
    p_val = results['significance']['p_value']
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    fig.suptitle(
        f'CMSMET: Cross-Modal Speech-Music Emotion Transfer — IEMOCAP 4-class LOSO\n'
        f'ΔUA = {delta:+.2%}   |   p = {p_val:.4f} ({sig})   |   3 seeds × 5 folds',
        fontsize=15, fontweight='bold', y=0.98, color='#222'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / 'summary_panel.png'
    plt.savefig(path)
    plt.close()
    print(f"  [OK] {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures from experiment results')
    parser.add_argument('--results-json', type=str,
                        default='outputs/paper_results/experiment_results.json',
                        help='Path to experiment_results.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as results json)')
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    results_path = project_root / args.results_json
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING PAPER-QUALITY FIGURES")
    print(f"{'='*60}")
    print(f"  Results: {results_path}")
    print(f"  Output:  {out_dir}\n")

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Print key results
    b_ua = results['baseline']['overall_ua_mean']
    c_ua = results['contrastive']['overall_ua_mean']
    print(f"  Baseline UA:     {b_ua:.4f} +/- {results['baseline']['overall_ua_std']:.4f}")
    print(f"  Contrastive UA:  {c_ua:.4f} +/- {results['contrastive']['overall_ua_std']:.4f}")
    print(f"  Delta UA:        {c_ua - b_ua:+.4f}")
    print(f"  p-value:         {results['significance']['p_value']:.4f}\n")

    print("Generating figures...")
    plot_per_fold(results, out_dir)
    plot_per_class(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_radar(results, out_dir)
    plot_summary_panel(results, out_dir)

    print(f"\n{'='*60}")
    print(f"ALL FIGURES SAVED TO: {out_dir}")
    print(f"{'='*60}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
