"""
unet_inference_v2.py
Inference for both U-Net and Attention U-Net.
Supports Test Time Augmentation (TTA) for better predictions.
Drop-in replacement for unet_inference.py
"""

import cv2
import numpy as np
import torch

from unet_model             import UNet
from attention_unet_model   import AttentionUNet
from attention_unet_v2      import AttentionUNetV2
from unet_dataset           import IMG_H, IMG_W


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_model  = None
_device = None


def _load_model(checkpoint_path='attention_unet_best.pth'):
    global _model, _device

    if _model is not None:
        return _model, _device

    if torch.cuda.is_available():
        _device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        _device = torch.device('mps')
    else:
        _device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=_device)

    # Auto-detect model type from checkpoint
    model_type = checkpoint.get('model_type', 'attention')
    if model_type in ('attention', 'attention_v1', 'attn'):
        _model = AttentionUNet(in_channels=1, out_channels=1)
    elif model_type in ('v2', 'attention_v2', 'attn_v2'):
        _model = AttentionUNetV2(in_channels=1, out_channels=1)
    else:
        _model = UNet(in_channels=1, out_channels=1)

    _model.load_state_dict(checkpoint['model_state'])
    _model = _model.to(_device)
    _model.eval()

    print(f"[unet] Model      : {model_type}")
    print(f"[unet] Device     : {_device}")
    print(f"[unet] Checkpoint : {checkpoint_path}")
    print(f"[unet] Epoch      : {checkpoint.get('epoch', '?')}")
    print(f"[unet] Val Dice   : {checkpoint.get('val_dice', float('nan')):.4f}")
    print(f"[unet] Val IoU    : {checkpoint.get('val_iou',  float('nan')):.4f}")
    print(f"[unet] Precision  : {checkpoint.get('val_prec', float('nan')):.4f}")
    print(f"[unet] Recall     : {checkpoint.get('val_rec',  float('nan')):.4f}")

    return _model, _device


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

    # A.Normalize(mean=(0.5,), std=(0.5,))
def _preprocess(img_gray):
    resized    = cv2.resize(img_gray, (IMG_W, IMG_H),
                            interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(normalised).unsqueeze(0).unsqueeze(0)


def _postprocess(prob_map, original_shape, threshold=0.25):
    prob     = prob_map.squeeze().cpu().numpy()
    oh, ow   = original_shape[:2]
    prob_big = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return (prob_big > threshold).astype(np.uint8) * 255, prob_big


# ─────────────────────────────────────────────────────────────────────────────
# TEST TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_with_tta(model, img_gray, device):
    """
    Run model on original + horizontally flipped image,
    average the probability maps.
    Typically adds +2–4% Dice with zero extra training.
    """
    t_orig  = _preprocess(img_gray).to(device)
    p_orig  = model(t_orig).squeeze().cpu().numpy()

    # Horizontal flip
    img_flip = cv2.flip(img_gray, 1)
    t_flip   = _preprocess(img_flip).to(device)
    p_flip   = model(t_flip).squeeze().cpu().numpy()
    p_flip   = np.fliplr(p_flip)   # flip back

    # Also try brightness augmentation
    img_bright = np.clip(img_gray.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    t_bright   = _preprocess(img_bright).to(device)
    p_bright   = model(t_bright).squeeze().cpu().numpy()

    img_dark   = np.clip(img_gray.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    t_dark     = _preprocess(img_dark).to(device)
    p_dark     = model(t_dark).squeeze().cpu().numpy()

    # Average all predictions
    prob_avg = (p_orig + p_flip + p_bright + p_dark) / 4.0
    return prob_avg


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_nerve_mask_unet(img_gray,
                           checkpoint_path='attention_unet_best.pth',
                           threshold=0.35,
                           use_tta=True):
    """
    Run Attention U-Net on a panoramic X-ray and return binary nerve mask.

    Args:
        img_gray        : (H, W) uint8 grayscale numpy array
        checkpoint_path : path to .pth checkpoint
        threshold       : confidence threshold (0.15–0.40 works well)
        use_tta         : use Test Time Augmentation (recommended, ~4× slower)

    Returns:
        (H, W) uint8 numpy array – 255=canal, 0=background
    """
    model, device = _load_model(checkpoint_path)

    if use_tta:
        prob = _predict_with_tta(model, img_gray, device)
    else:
        tensor = _preprocess(img_gray).to(device)
        prob   = model(tensor).squeeze().cpu().numpy()

    # Resize to original size
    oh, ow   = img_gray.shape[:2]
    prob_big = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
    mask     = (prob_big > threshold).astype(np.uint8) * 255

    # Save probability heatmap for debug
    prob_vis   = (prob_big * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    cv2.imwrite('debug_unet_probability.jpg', prob_color)

    canal_px = int(np.sum(mask > 0))
    tta_str  = " (TTA)" if use_tta else ""
    print(f"[unet] Canal pixels{tta_str}: {canal_px}  "
          f"(threshold={threshold})")
    print(f"[unet] Prob range: "
          f"min={prob_big.min():.3f}  "
          f"max={prob_big.max():.3f}  "
          f"mean={prob_big.mean():.3f}")

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/images/125.jpg'
    img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: cannot load {img_path}")
        sys.exit(1)

    mask = build_nerve_mask_unet(img, use_tta=True)

    colour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colour[mask > 0] = [0, 220, 100]
    cv2.imwrite('attention_unet_preview.jpg', colour)
    print("Saved: attention_unet_preview.jpg")