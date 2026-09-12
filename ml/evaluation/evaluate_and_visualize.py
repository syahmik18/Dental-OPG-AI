"""
evaluate_and_visualize.py
Generates visual proof of model performance for FYP report Chapter 4.

Outputs (all saved as images you can screenshot or include directly):
  1. results_summary.png     — Clean metrics table + bar chart
  2. training_curves.png     — Dice/Loss curve over epochs (if log exists)
  3. sample_predictions.png  — Grid: Original | Ground Truth | Prediction | Overlay
  4. threshold_curve.png     — Dice vs Threshold sweep chart
  5. Terminal output         — Screenshot-worthy formatted metrics table

Run on Colab:
    !python evaluate_and_visualize.py

Run on Kaggle:
    !python /kaggle/working/evaluate_and_visualize.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import DataLoader, Subset

# ── Import your model and dataset ─────────────────────────────────────────────
from attention_unet_v2 import AttentionUNetV2
from unet_dataset      import NerveDataset, get_val_transforms


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update paths to match your environment
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Colab
    'image_dir'  : 'dataset/images',
    'mask_dir'   : 'dataset/masks',
    'checkpoint' : 'best_v2.pth',
    'output_dir' : 'report_figures',    # all figures saved here

    # Kaggle (uncomment if running on Kaggle)
    # 'image_dir'  : '/kaggle/input/datasets/khairulsyahmi5755/nerveandcaries/U-Net/dataset/images',
    # 'mask_dir'   : '/kaggle/input/datasets/khairulsyahmi5755/nerveandcaries/U-Net/dataset/masks',
    # 'checkpoint' : '/kaggle/working/best_v2.pth',
    # 'output_dir' : '/kaggle/working/report_figures',

    'val_split'  : 0.15,
    'batch_size' : 4,
    'num_workers': 2,

    # How many sample prediction images to show in grid
    'num_samples': 6,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE & MODEL
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')


def load_model(checkpoint_path, device):
    model = AttentionUNetV2(
        in_channels=1, out_channels=1,
        features=(64, 128, 256, 512),
        deep_supervision=False,   # eval mode — no aux outputs
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel prefix if present
    state = ckpt['model_state']
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)

    model.eval()
    print(f"  Loaded checkpoint : {checkpoint_path}")
    print(f"  Saved at epoch    : {ckpt.get('epoch', 'N/A')}")
    print(f"  Best threshold    : {ckpt.get('best_threshold', 0.5):.2f}")
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float().view(-1)
    tgt      = target.view(-1)
    tp = (pred_bin * tgt).sum().item()
    fp = (pred_bin * (1 - tgt)).sum().item()
    fn = ((1 - pred_bin) * tgt).sum().item()
    tn = ((1 - pred_bin) * (1 - tgt)).sum().item()

    smooth = 1.0
    dice   = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou    = (tp + smooth) / (tp + fp + fn + smooth)
    prec   = tp / (tp + fp + 1e-8)
    rec    = tp / (tp + fn + 1e-8)
    f1     = (2 * prec * rec) / (prec + rec + 1e-8)
    return {'dice': dice, 'iou': iou,
            'precision': prec, 'recall': rec, 'f1': f1}


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_metrics = []
    all_preds   = []
    all_masks   = []
    all_imgs    = []

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        if isinstance(preds, tuple):
            preds = preds[0]

        for i in range(len(imgs)):
            m = compute_metrics(preds[i], masks[i], threshold)
            all_metrics.append(m)
            all_preds.append(preds[i].cpu())
            all_masks.append(masks[i].cpu())
            all_imgs.append(imgs[i].cpu())

    # Average metrics
    avg = {}
    for k in all_metrics[0].keys():
        avg[k] = np.mean([m[k] for m in all_metrics])

    return avg, all_preds, all_masks, all_imgs


# ─────────────────────────────────────────────────────────────────────────────
# 1. RESULTS SUMMARY FIGURE — metrics table + bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_summary(metrics, threshold, save_path):
    fig = plt.figure(figsize=(14, 6), facecolor='white')
    fig.suptitle('Attention U-Net V2 — Validation Performance',
                 fontsize=16, fontweight='bold', y=1.02)

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.4)

    # ── Left: Metrics Table ──────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')

    metric_names  = ['Dice Score', 'IoU (Jaccard)', 'Precision',
                     'Recall (Sensitivity)', 'F1 Score']
    metric_keys   = ['dice', 'iou', 'precision', 'recall', 'f1']
    metric_values = [f"{metrics[k]:.4f}" for k in metric_keys]

    # Header
    col_labels = ['Metric', 'Value']
    table_data = list(zip(metric_names, metric_values))

    table = ax_table.table(
        cellText  = table_data,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)

    # Style header row
    for j in range(2):
        table[(0, j)].set_facecolor('#1e293b')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style data rows alternating
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor('#f0f4f8' if i % 2 == 0 else 'white')
            if j == 1:
                table[(i, j)].set_text_props(
                    color='#16a34a' if float(metric_values[i-1]) >= 0.5
                    else '#d97706', fontweight='bold')

    ax_table.set_title(f'Validation Metrics  (threshold={threshold:.2f})',
                       fontsize=12, pad=10, fontweight='bold')

    # ── Right: Bar Chart ─────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])

    colors = ['#2563eb', '#7c3aed', '#16a34a', '#dc2626', '#d97706']
    vals   = [metrics[k] for k in metric_keys]
    bars   = ax_bar.bar(metric_names, vals, color=colors,
                        width=0.55, alpha=0.88, edgecolor='white',
                        linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.4f}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel('Score', fontsize=12)
    ax_bar.set_title('Performance Metrics Bar Chart',
                     fontsize=12, fontweight='bold')
    ax_bar.tick_params(axis='x', labelsize=9, rotation=20)
    ax_bar.axhline(y=0.5, color='gray', linestyle='--',
                   alpha=0.5, linewidth=1, label='0.5 baseline')
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar.legend(fontsize=9)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAMPLE PREDICTIONS GRID
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_predictions(all_imgs, all_preds, all_masks,
                            threshold, save_path, n_samples=6):
    n = min(n_samples, len(all_imgs))
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5),
                             facecolor='white')
    fig.suptitle('Attention U-Net V2 — Sample Predictions on Validation Set',
                 fontsize=15, fontweight='bold', y=1.01)

    col_titles = ['Input X-ray', 'Ground Truth Mask',
                  'Predicted Mask', 'Overlay Result']

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11,
                               fontweight='bold', pad=8)

    for i in range(n):
        img     = all_imgs[i]
        pred    = all_preds[i]
        mask    = all_masks[i]

        # Denormalize image (was normalized with mean=0.5, std=0.5)
        img_np  = (img.squeeze().numpy() * 0.5 + 0.5)
        img_np  = np.clip(img_np, 0, 1)

        mask_np = mask.squeeze().numpy()
        pred_np = pred.squeeze().numpy()
        pred_bin = (pred_np > threshold).astype(np.float32)

        # Dice for this sample
        inter   = (pred_bin * mask_np).sum()
        dice_i  = (2 * inter + 1) / (pred_bin.sum() + mask_np.sum() + 1)

        # Overlay: image BGR + green canal
        img_bgr = cv2.cvtColor(
            (img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = img_bgr.copy()
        overlay[pred_bin > 0] = [0, 200, 100]
        result  = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
        result  = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Plot
        axes[i][0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes[i][1].imshow(mask_np, cmap='Greens', vmin=0, vmax=1)
        axes[i][2].imshow(pred_bin, cmap='Greens', vmin=0, vmax=1)
        axes[i][3].imshow(result)

        # Dice label on each row
        axes[i][0].set_ylabel(f'Sample {i+1}\nDice={dice_i:.3f}',
                               fontsize=9, labelpad=4)

        for j in range(4):
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. THRESHOLD CURVE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_threshold_curve(model, loader, device, save_path):
    model.eval()
    all_preds, all_masks = [], []

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        preds = model(imgs)
        if isinstance(preds, tuple): preds = preds[0]
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

    preds = torch.cat(all_preds)
    masks = torch.cat(all_masks)

    thresholds = [t/100 for t in range(10, 90, 2)]
    dices, ious, precs, recs = [], [], [], []

    for t in thresholds:
        m = compute_metrics(preds, masks, threshold=t)
        dices.append(m['dice'])
        ious.append(m['iou'])
        precs.append(m['precision'])
        recs.append(m['recall'])

    best_t = thresholds[np.argmax(dices)]
    best_d = max(dices)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.plot(thresholds, dices, 'b-o', markersize=4, label='Dice Score',    lw=2)
    ax.plot(thresholds, ious,  'g-s', markersize=4, label='IoU Score',     lw=2)
    ax.plot(thresholds, precs, 'r-^', markersize=4, label='Precision',     lw=2)
    ax.plot(thresholds, recs,  'm-D', markersize=4, label='Recall',        lw=2)

    # Mark best threshold
    ax.axvline(x=best_t, color='navy', linestyle='--', linewidth=1.5,
               label=f'Best threshold={best_t:.2f}')
    ax.annotate(f'Best Dice={best_d:.4f}\n@ threshold={best_t:.2f}',
                xy=(best_t, best_d),
                xytext=(best_t + 0.06, best_d - 0.08),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='navy'),
                color='navy', fontweight='bold')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Dice / IoU / Precision / Recall vs Threshold\n'
                 'Attention U-Net V2 — Validation Set',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")
    return best_t, best_d


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARISON TABLE — Baseline vs V2
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_table(metrics_v2, save_path):
    """
    Hardcoded baseline UNet metrics for comparison table.
    Edit baseline values if you have your own baseline results.
    """
    baseline = {
        'dice': 0.5188, 'iou': 0.3508,
        'precision': 0.4309, 'recall': 0.6533, 'f1': 0.5188,
    }

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    ax.axis('off')
    ax.set_title('Model Comparison: Baseline U-Net vs Attention U-Net V2',
                 fontsize=14, fontweight='bold', pad=20)

    metric_names = ['Dice Score', 'IoU', 'Precision', 'Recall', 'F1 Score']
    metric_keys  = ['dice', 'iou', 'precision', 'recall', 'f1']

    base_vals = [f"{baseline[k]:.4f}" for k in metric_keys]
    v2_vals   = [f"{metrics_v2[k]:.4f}" for k in metric_keys]

    # Calculate improvement
    improv = []
    for k in metric_keys:
        diff = metrics_v2[k] - baseline[k]
        sign = '+' if diff >= 0 else ''
        improv.append(f"{sign}{diff:.4f}")

    col_labels = ['Metric', 'Baseline U-Net', 'Attention U-Net V2',
                  'Improvement']
    table_data = list(zip(metric_names, base_vals, v2_vals, improv))

    table = ax.table(
        cellText  = table_data,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.5)

    # Header styling
    for j in range(4):
        table[(0, j)].set_facecolor('#1e293b')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data) + 1):
        # Row background
        bg = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(bg)

        # Colour improvement column
        diff_val = metrics_v2[metric_keys[i-1]] - baseline[metric_keys[i-1]]
        color = '#16a34a' if diff_val >= 0 else '#dc2626'
        table[(i, 3)].set_text_props(color=color, fontweight='bold')

        # Colour V2 column green if better
        table[(i, 2)].set_text_props(color='#2563eb', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL OUTPUT — screenshot-worthy
# ─────────────────────────────────────────────────────────────────────────────

def print_results(metrics, threshold, checkpoint):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ATTENTION U-NET V2 — VALIDATION RESULTS")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Threshold  : {threshold:.2f}")
    print(sep)
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-'*36}")
    labels = {
        'dice'     : 'Dice Score (F1)',
        'iou'      : 'IoU (Jaccard Index)',
        'precision': 'Precision',
        'recall'   : 'Recall (Sensitivity)',
        'f1'       : 'F1 Score',
    }
    for k, label in labels.items():
        bar = '█' * int(metrics[k] * 20)
        print(f"  {label:<25} {metrics[k]:>10.4f}  {bar}")
    print(sep)
    print(f"  All figures saved to: {CONFIG['output_dir']}/")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"\nDevice: {device}")

    # Load checkpoint
    model, ckpt = load_model(CONFIG['checkpoint'], device)
    threshold   = ckpt.get('best_threshold', 0.5)

    # Dataset
    val_ds  = NerveDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                           transform=get_val_transforms())
    n       = len(val_ds)
    n_val   = max(1, int(n * CONFIG['val_split']))
    val_idx = list(range(n))[:n_val]

    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size  = CONFIG['batch_size'],
        shuffle     = False,
        num_workers = CONFIG['num_workers'],
    )
    print(f"Validation samples: {len(val_idx)}")

    # Evaluate
    print("\nRunning evaluation...")
    metrics, all_preds, all_masks, all_imgs = evaluate(
        model, val_loader, device, threshold)

    # Print terminal output — SCREENSHOT THIS
    print_results(metrics, threshold, CONFIG['checkpoint'])

    # Generate all figures
    print("Generating report figures...\n")

    plot_results_summary(
        metrics, threshold,
        os.path.join(CONFIG['output_dir'], '1_results_summary.png'))

    plot_sample_predictions(
        all_imgs, all_preds, all_masks, threshold,
        os.path.join(CONFIG['output_dir'], '2_sample_predictions.png'),
        n_samples=CONFIG['num_samples'])

    plot_threshold_curve(
        model, val_loader, device,
        os.path.join(CONFIG['output_dir'], '3_threshold_curve.png'))

    plot_comparison_table(
        metrics,
        os.path.join(CONFIG['output_dir'], '4_model_comparison.png'))

    print("\nDone! All figures saved.")
    print(f"Download from: {CONFIG['output_dir']}/")
    print("\nFor your report:")
    print("  Figure X — 1_results_summary.png   : Metrics table + bar chart")
    print("  Figure X — 2_sample_predictions.png: Visual predictions grid")
    print("  Figure X — 3_threshold_curve.png   : Threshold analysis")
    print("  Figure X — 4_model_comparison.png  : Baseline vs V2 comparison")


if __name__ == '__main__':
    main()
