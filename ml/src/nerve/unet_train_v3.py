"""
unet_train_v3.py
Improved training script for mandibular canal segmentation.

Key improvements over v2:
  1. Focal Tversky Loss  – precision/recall tradeoff tunable via alpha/beta
  2. Weighted BCE        – pos_weight handles canal vs. background imbalance
  3. Deep supervision    – auxiliary decoder outputs improve gradient flow
  4. Warm-up + Cosine    – 10-epoch warmup before cosine decay
  5. Threshold optimiser – finds best threshold on val set after training
  6. Test-Time Augmentation (TTA) – H-flip average at inference
  7. Gradient clipping   – prevents exploding gradients

Usage:
    python unet_train_v3.py --mode single  --model v2     # recommended
    python unet_train_v3.py --mode kfold   --model v2
    python unet_train_v3.py --mode compare               # v1 vs v2
"""

import os
import time
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from attention_unet_model  import AttentionUNet          # v1 (original)
from attention_unet_v2     import AttentionUNetV2        # v2 (improved)
from unet_dataset          import NerveDataset, get_train_transforms, get_val_transforms

try:
    from unet_model import UNet
except ImportError:
    UNet = None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    'image_dir'    : 'dataset/images',
    'mask_dir'     : 'dataset/masks',
    'epochs'       : 200,
    'batch_size'   : 8,
    'lr'           : 3e-4,          # lower than v2 (was 1e-3); warmup handles cold start
    'warmup_epochs': 10,
    'n_folds'      : 5,
    'num_workers'  : 2,
    'model_type'   : 'v2',          # 'v2' | 'v1' | 'unet'
    'checkpoint'   : 'best_v2.pth',
    # Tversky loss hyperparams
    # alpha = FP weight,  beta = FN weight.
    # alpha < beta  → penalise missed canals more → higher recall
    # alpha > beta  → penalise false positives more → higher precision
    # 0.4/0.6 is a good starting balance; adjust if recall/precision still off
    'tversky_alpha': 0.4,
    'tversky_beta' : 0.6,
    'focal_gamma'  : 0.75,          # Focal Tversky exponent
    # Class imbalance – estimated ratio of background:canal pixels
    # Set to actual ratio if you know it (e.g., 20 if canal is ~5% of image)
    'pos_weight'   : 15.0,
    # Loss weights
    'loss_main_w'  : 1.0,
    'loss_aux3_w'  : 0.4,           # weight for deep-supervision aux3
    'loss_aux2_w'  : 0.2,           # weight for deep-supervision aux2
}


# ─────────────────────────────────────────────────────────────────────────────
# SEED / DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    if torch.cuda.is_available():   return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan, 2019).
    Generalises Dice/IoU loss with tunable FP/FN weighting.

    TI  = TP / (TP + alpha*FP + beta*FN)
    FTL = (1 - TI) ** gamma

    alpha + beta = 1.0
    • alpha=0.5, beta=0.5 → Dice loss
    • alpha<0.5, beta>0.5 → penalise FN more → higher recall
    • alpha>0.5, beta<0.5 → penalise FP more → higher precision
    gamma < 1 focuses learning on hard examples.
    """
    def __init__(self, alpha=0.4, beta=0.6, gamma=0.75, smooth=1.0):
        super().__init__()
        assert abs(alpha + beta - 1.0) < 1e-6, "alpha + beta must equal 1.0"
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        p   = pred.view(-1)
        t   = target.view(-1)
        tp  = (p * t).sum()
        fp  = (p * (1 - t)).sum()
        fn  = ((1 - p) * t).sum()
        ti  = (tp + self.smooth) / \
              (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - ti) ** self.gamma


class CombinedLossV3(nn.Module):
    """
    Weighted BCE  +  Focal Tversky Loss
    BCE handles per-pixel calibration; FTL handles shape/recall.
    pos_weight addresses the heavy class imbalance.
    """
    def __init__(self, pos_weight=15.0,
                 alpha=0.4, beta=0.6, gamma=0.75):
        super().__init__()
        pw        = torch.tensor([pos_weight])
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)   # uses logits
        self.ftl  = FocalTverskyLoss(alpha, beta, gamma)

        # Separate sigmoid for FTL (BCEWithLogitsLoss applies it internally)
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits, target):
        """
        logits : raw model output (BEFORE sigmoid) – only needed for BCE.
        pred   : sigmoid(logits) – for FTL.
        If your model already outputs sigmoid probabilities, wrap with
        a no-op and use BCELoss instead.
        """
        bce_loss = self.bce(logits, target)
        pred     = self.sigmoid(logits)
        ftl_loss = self.ftl(pred, target)
        return 0.4 * bce_loss + 0.6 * ftl_loss


class SigmoidOutputLossV3(nn.Module):
    """
    Version for models that output sigmoid probabilities (not logits).
    Uses BCELoss + FocalTverskyLoss.
    """
    def __init__(self, pos_weight=15.0,
                 alpha=0.4, beta=0.6, gamma=0.75):
        super().__init__()
        self.bce = nn.BCELoss()
        self.ftl = FocalTverskyLoss(alpha, beta, gamma)
        # Manual pos_weight for BCELoss
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        # Weighted BCE: scale loss on positive pixels
        bce_raw  = F.binary_cross_entropy(pred, target, reduction='none')
        weight   = torch.where(target > 0.5,
                               torch.full_like(target, self.pos_weight),
                               torch.ones_like(target))
        bce_loss = (bce_raw * weight).mean()
        ftl_loss = self.ftl(pred, target)
        return 0.4 * bce_loss + 0.6 * ftl_loss


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred, target, threshold=0.5, smooth=1.0):
    pred   = (pred > threshold).float().view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return ((2 * inter + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    pred   = (pred > threshold).float().view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def precision_recall(pred, target, threshold=0.5):
    pred   = (pred > threshold).float().view(-1)
    target = target.view(-1)
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def find_best_threshold(model, loader, device, thresholds=None):
    """
    Sweeps threshold values on the validation set and returns the
    threshold that maximises Dice score.
    Run once after training finishes.
    """
    if thresholds is None:
        thresholds = [t / 100 for t in range(20, 80, 2)]   # 0.20 → 0.78

    model.eval()
    all_preds, all_masks = [], []

    for imgs, masks in tqdm(loader, desc='  Threshold sweep', leave=False):
        imgs  = imgs.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    best_thresh = 0.5
    best_dice   = 0.0

    print(f"\n  {'Threshold':>10} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
    print("  " + "-" * 50)
    for t in thresholds:
        d = dice_score(all_preds, all_masks, threshold=t)
        i = iou_score(all_preds, all_masks, threshold=t)
        p, r = precision_recall(all_preds, all_masks, threshold=t)
        print(f"  {t:>10.2f} {d:>8.4f} {i:>8.4f} {p:>8.4f} {r:>8.4f}")
        if d > best_dice:
            best_dice   = d
            best_thresh = t

    print(f"\n  Best threshold : {best_thresh:.2f}  (Dice={best_dice:.4f})")
    return best_thresh, best_dice


# ─────────────────────────────────────────────────────────────────────────────
# TTA (TEST-TIME AUGMENTATION)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, imgs):
    """
    Horizontal-flip TTA: average prediction of original + h-flipped image.
    Robust for panoramic X-rays where left/right are anatomically symmetric.
    """
    pred_orig = model(imgs)
    pred_flip = model(torch.flip(imgs, dims=[3]))
    pred_flip = torch.flip(pred_flip, dims=[3])     # flip back
    return (pred_orig + pred_flip) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER: WARMUP + COSINE
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `eta_min`.
    Drop-in replacement for CosineAnnealingLR + warmup.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, eta_min=1e-6):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lr        = base_lr
        self.eta_min        = eta_min
        self._epoch         = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_deep_supervision=True):
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc='  Train', leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        output = model(imgs)

        if use_deep_supervision and isinstance(output, tuple):
            main, aux3, aux2 = output
            loss = (CONFIG['loss_main_w'] * criterion(main,  masks) +
                    CONFIG['loss_aux3_w'] * criterion(aux3,  masks) +
                    CONFIG['loss_aux2_w'] * criterion(aux2,  masks))
        else:
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5, use_tta=False):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    total_prec = total_rec  = 0.0

    for imgs, masks in tqdm(loader, desc='  Val  ', leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        preds = tta_predict(model, imgs) if use_tta else model(imgs)

        total_loss += criterion(preds, masks).item()
        total_dice += dice_score(preds, masks, threshold)
        total_iou  += iou_score(preds, masks, threshold)
        p, r        = precision_recall(preds, masks, threshold)
        total_prec += p
        total_rec  += r

    n = len(loader)
    return (total_loss / n, total_dice / n,
            total_iou / n, total_prec / n, total_rec / n)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(model_type):
    if model_type == 'v2':
        return AttentionUNetV2(in_channels=1, out_channels=1,
                               features=(64, 128, 256, 512),
                               deep_supervision=True)
    if model_type == 'v1':
        return AttentionUNet(in_channels=1, out_channels=1)
    if model_type == 'unet' and UNet is not None:
        return UNet(in_channels=1, out_channels=1)
    raise ValueError(f"Unknown model type: {model_type}")


def _build_criterion():
    """
    Use SigmoidOutputLossV3 because our models output sigmoid probabilities.
    """
    return SigmoidOutputLossV3(
        pos_weight=CONFIG['pos_weight'],
        alpha=CONFIG['tversky_alpha'],
        beta=CONFIG['tversky_beta'],
        gamma=CONFIG['focal_gamma'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────

def train_single(model_type='v2'):
    set_seed(42)
    device     = get_device()
    print(f"\nDevice     : {device}")
    print(f"Model      : {model_type}")
    print("=" * 70)

    train_ds = NerveDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                            transform=get_train_transforms())
    val_ds   = NerveDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                            transform=get_val_transforms())

    n        = len(train_ds)
    n_val    = max(1, int(n * 0.15))
    indices  = list(range(n))
    train_idx = indices[n_val:]
    val_idx   = indices[:n_val]

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=CONFIG['batch_size'],
                              shuffle=True,
                              num_workers=CONFIG['num_workers'],
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=CONFIG['batch_size'],
                              shuffle=False,
                              num_workers=CONFIG['num_workers'],
                              pin_memory=(device.type == 'cuda'))

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    model     = _build_model(model_type).to(device)
    criterion = _build_criterion()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=CONFIG['warmup_epochs'],
        total_epochs=CONFIG['epochs'],
        base_lr=CONFIG['lr'],
    )

    best_dice   = 0.0
    checkpoint  = CONFIG['checkpoint']
    use_ds      = (model_type == 'v2')   # deep supervision only for v2

    print(f"\n{'Ep':>4} | {'TrLoss':>7} {'VlLoss':>7} {'Dice':>7} "
          f"{'IoU':>7} {'Prec':>7} {'Rec':>7} | {'LR':>8}")
    print("-" * 70)

    for epoch in range(1, CONFIG['epochs'] + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, use_ds)
        val_loss, val_dice, val_iou, val_prec, val_rec = validate(
            model, val_loader, criterion, device)
        lr = scheduler.step()

        print(f"{epoch:>4} | {train_loss:>7.4f} {val_loss:>7.4f} "
              f"{val_dice:>7.4f} {val_iou:>7.4f} "
              f"{val_prec:>7.4f} {val_rec:>7.4f} | {lr:>8.6f}"
              f"  [{time.time()-t0:.0f}s]")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch'      : epoch,
                'model_type' : model_type,
                'model_state': model.state_dict(),
                'val_dice'   : val_dice,
                'val_iou'    : val_iou,
                'val_prec'   : val_prec,
                'val_rec'    : val_rec,
            }, checkpoint)
            print(f"  ✓ Saved  Dice={val_dice:.4f}")

    # ── Post-training: find optimal threshold ───────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMISATION (on validation set)")
    print("=" * 70)
    # Reload best model
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    best_thresh, best_thresh_dice = find_best_threshold(
        model, val_loader, device)

    # Save threshold into checkpoint
    ckpt['best_threshold'] = best_thresh
    torch.save(ckpt, checkpoint)
    print(f"\nBest Dice (default 0.5) : {best_dice:.4f}")
    print(f"Best Dice (tuned)       : {best_thresh_dice:.4f}  @ threshold={best_thresh:.2f}")
    print(f"Checkpoint saved to     : {checkpoint}")

    return best_dice, best_thresh


# ─────────────────────────────────────────────────────────────────────────────
# K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def train_kfold(model_type='v2'):
    set_seed(42)
    device = get_device()
    print(f"\nDevice     : {device}")
    print(f"Model      : {model_type}")
    print(f"K-Folds    : {CONFIG['n_folds']}")
    print("=" * 70)

    full_ds = NerveDataset(CONFIG['image_dir'], CONFIG['mask_dir'])
    n       = len(full_ds)
    indices = list(range(n))
    np.random.seed(42);  np.random.shuffle(indices)

    fold_size    = n // CONFIG['n_folds']
    fold_metrics = []

    for fold in range(CONFIG['n_folds']):
        print(f"\n{'='*70}\nFOLD {fold+1}/{CONFIG['n_folds']}\n{'='*70}")

        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = indices[:fold * fold_size] + \
                    indices[(fold + 1) * fold_size:]

        train_ds = NerveDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                                transform=get_train_transforms())
        val_ds   = NerveDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                                transform=get_val_transforms())

        train_loader = DataLoader(Subset(train_ds, train_idx),
                                  batch_size=CONFIG['batch_size'],
                                  shuffle=True,
                                  num_workers=CONFIG['num_workers'],
                                  pin_memory=(device.type == 'cuda'))
        val_loader   = DataLoader(Subset(val_ds, val_idx),
                                  batch_size=CONFIG['batch_size'],
                                  shuffle=False,
                                  num_workers=CONFIG['num_workers'],
                                  pin_memory=(device.type == 'cuda'))

        print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

        model     = _build_model(model_type).to(device)
        criterion = _build_criterion()
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=CONFIG['warmup_epochs'],
            total_epochs=CONFIG['epochs'],
            base_lr=CONFIG['lr'],
        )

        best_dice    = 0.0
        best_metrics = {}
        fold_ckpt    = f'fold_{fold+1}_{model_type}_best.pth'
        use_ds       = (model_type == 'v2')

        for epoch in range(1, CONFIG['epochs'] + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         criterion, device, use_ds)
            val_loss, val_dice, val_iou, val_prec, val_rec = validate(
                model, val_loader, criterion, device)
            lr = scheduler.step()

            print(f"  Ep {epoch:3d} | Loss {train_loss:.4f}/{val_loss:.4f} | "
                  f"Dice {val_dice:.4f} | IoU {val_iou:.4f} | "
                  f"Prec {val_prec:.4f} | Rec {val_rec:.4f} | {time.time()-t0:.0f}s")

            if val_dice > best_dice:
                best_dice    = val_dice
                best_metrics = dict(fold=fold+1, dice=val_dice, iou=val_iou,
                                    precision=val_prec, recall=val_rec)
                torch.save(dict(epoch=epoch, model_type=model_type,
                                model_state=model.state_dict(),
                                **best_metrics), fold_ckpt)
                print(f"    ✓ Fold {fold+1} best  Dice={val_dice:.4f}")

        # Threshold optimisation per fold
        ckpt = torch.load(fold_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        best_thresh, _ = find_best_threshold(model, val_loader, device)
        best_metrics['best_threshold'] = best_thresh

        fold_metrics.append(best_metrics)
        print(f"\nFold {fold+1}: Dice={best_metrics['dice']:.4f}  "
              f"IoU={best_metrics['iou']:.4f}  "
              f"Prec={best_metrics['precision']:.4f}  "
              f"Rec={best_metrics['recall']:.4f}  "
              f"Threshold={best_thresh:.2f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY ({CONFIG['n_folds']}-Fold)")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for metric in ['dice', 'iou', 'precision', 'recall']:
        vals = [m[metric] for m in fold_metrics]
        print(f"{metric:<15} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    best_fold = max(fold_metrics, key=lambda x: x['dice'])
    best_ckpt = f"fold_{best_fold['fold']}_{model_type}_best.pth"
    import shutil
    shutil.copy(best_ckpt, CONFIG['checkpoint'])
    print(f"\nBest fold: {best_fold['fold']}  Dice={best_fold['dice']:.4f}")
    print(f"Saved best fold model to: {CONFIG['checkpoint']}")
    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE V1 vs V2
# ─────────────────────────────────────────────────────────────────────────────

def compare_models():
    print("\nCOMPARING Attention U-Net V1 vs V2")
    print("=" * 70)
    results = {}
    for mt in ['v1', 'v2']:
        print(f"\n>>> Training {mt.upper()} <<<")
        CONFIG['checkpoint'] = f'{mt}_best.pth'
        CONFIG['model_type'] = mt
        dice, thresh = train_single(mt)
        results[mt]  = dict(dice=dice, threshold=thresh)

    print(f"\n{'='*70}\nCOMPARISON RESULTS\n{'='*70}")
    for mt, r in results.items():
        print(f"  {mt:<6}  Dice={r['dice']:.4f}  Threshold={r['threshold']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  default='single',
                        choices=['single', 'kfold', 'compare'])
    parser.add_argument('--model', default='v2',
                        choices=['v2', 'v1', 'unet'])
    args = parser.parse_args()

    CONFIG['model_type'] = args.model
    CONFIG['checkpoint'] = f'{args.model}_best.pth'

    if   args.mode == 'single':  train_single(args.model)
    elif args.mode == 'kfold':   train_kfold(args.model)
    elif args.mode == 'compare': compare_models()
