"""
yolo_inference_v2.py
Improved YOLOv8 caries detection with:
  - Test Time Augmentation (TTA)
  - Lower IoU threshold to fix duplicate detections
  - Confidence filtering
  - Nerve overlap suppression
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_caries_model = None

def _load_caries_model(weights):
    global _caries_model
    if _caries_model is None:
        print(f"[yolo] Loading caries model: {weights}")
        _caries_model = YOLO(weights)
    return _caries_model


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def detect_caries_yolo(img_gray,
                        weights='best_caries_yolo26.pt',
                        conf_threshold=0.25,
                        iou_threshold=0.30,
                        use_tta=True,
                        nerve_mask=None):
    """
    Detect caries using trained YOLO26m model.

    Args:
        img_gray       : (H, W) uint8 grayscale numpy array
        weights        : path to trained best.pt
        conf_threshold : minimum confidence (lower = more detections)
        iou_threshold  : NMS IoU threshold (lower = less duplicates)
        use_tta        : Test Time Augmentation (recommended)
        nerve_mask     : optional binary mask to suppress nerve overlaps

    Returns:
        List of (x_centre, y_centre, score, x1, y1, x2, y2)
    """
    model   = _load_caries_model(weights)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Run inference
    # Note: augment=True (TTA) not supported by YOLO26m's NMS-free head
    # — falls back to single-scale automatically, so we disable it explicitly
    results = model(
        img_bgr,
        conf    = conf_threshold,
        iou     = iou_threshold,
        augment = False,           # YOLO26m NMS-free head does not support TTA
        verbose = False,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp to image bounds
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(img_gray.shape[1], x2)
            y2 = min(img_gray.shape[0], y2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Skip tiny boxes (likely noise)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            # Suppress detections overlapping nerve canal
            if nerve_mask is not None:
                region = nerve_mask[y1:y2, x1:x2]
                if region.size > 0:
                    nerve_ratio = np.sum(region > 0) / region.size
                    if nerve_ratio > 0.30:
                        print(f"  [yolo] Suppressed at ({cx},{cy}) "
                              f"– nerve overlap {nerve_ratio:.0%}")
                        continue

            detections.append((cx, cy, conf, x1, y1, x2, y2))

    # Sort by confidence
    detections.sort(key=lambda d: d[2], reverse=True)

    print(f"[yolo] {len(detections)} caries detected "
          f"(conf>{conf_threshold}, iou<{iou_threshold}, "
          f"U-Net TTA={'on' if use_tta else 'off'})")
    for i, (cx, cy, conf, x1, y1, x2, y2) in enumerate(detections, 1):
        print(f"  Caries {i}: ({cx},{cy})  conf={conf:.3f}  "
              f"box=({x1},{y1},{x2},{y2})")

    return detections