"""
final_integrated_detection_v3.py
Full pipeline with all improvements:
  Nerve  → Attention U-Net + TTA + post-processing
  Caries → YOLOv8m + TTA + IoU NMS fix
"""

import cv2
import numpy as np
from unet_inference_v2            import build_nerve_mask_unet
from nerve_postprocessing_updated import post_process_nerve_mask
from yolo_inference_v2            import detect_caries_yolo


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG – edit paths here
# ─────────────────────────────────────────────────────────────────────────────

UNET_CHECKPOINT = 'attention_unet_best.pth'
YOLO_WEIGHTS    = './runs/detect/runs/detect/caries_runs/train-2/weights/best_caries_yolo26.pt'
CONF_THRESHOLD  = 0.25    # lower = more detections
IOU_THRESHOLD   = 0.30    # lower = fewer duplicates
USE_TTA         = True    # Test Time Augmentation


def detect_all(img_path, output_path='final_detection_v3.jpg'):
    print("DENTAL DETECTION  [Attention U-Net + YOLOv8]")
    print("=" * 60)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot load {img_path}")
        return
    print(f"Image: {img_path}  {img.shape}")

    # ── STEP 1: Nerve (Attention U-Net + TTA) ────────────────────
    print("\n[STEP 1] Nerve canal  (Attention U-Net + TTA)...")
    raw_nerve  = build_nerve_mask_unet(
        img,
        checkpoint_path = UNET_CHECKPOINT,
        threshold       = 0.35,
        use_tta         = USE_TTA,
    )
    nerve_mask = post_process_nerve_mask(raw_nerve.copy(), img=img)
    nerve_px   = int(np.sum(nerve_mask > 0))
    print(f"Nerve pixels: {nerve_px}")

    # ── STEP 2: Caries (YOLOv8 + TTA + IoU fix) ─────────────────
    print("\n[STEP 2] Caries  (YOLOv8m + TTA)...")
    caries = detect_caries_yolo(
        img,
        weights        = YOLO_WEIGHTS,
        conf_threshold = CONF_THRESHOLD,
        iou_threshold  = IOU_THRESHOLD,
        use_tta        = USE_TTA,
        nerve_mask     = nerve_mask,
    )

    # ── STEP 3: Annotate ─────────────────────────────────────────
    print("\n[STEP 3] Annotating...")
    annotated = annotate(img, nerve_mask, caries)
    cv2.imwrite(output_path, annotated)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  Image         : {img_path}")
    print(f"  Nerve pixels  : {nerve_px}")
    print(f"  Caries found  : {len(caries)}")
    print(f"  Output        : {output_path}")
    for i, (cx, cy, conf, x1, y1, x2, y2) in enumerate(caries, 1):
        print(f"    Caries {i}: ({cx},{cy})  conf={conf:.3f}")
    return annotated


def annotate(img, nerve_mask, caries):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── Nerve: semi-transparent green ───────────────────────────
    if int(np.sum(nerve_mask > 0)) > 0:
        overlay              = np.zeros_like(out)
        overlay[nerve_mask > 0] = [0, 220, 100]
        mask_bool            = cv2.cvtColor(nerve_mask,
                                            cv2.COLOR_GRAY2BGR).astype(bool)
        out[mask_bool] = (
            0.65 * out[mask_bool] +
            0.35 * overlay[mask_bool]
        ).astype(np.uint8)

    # ── Caries: red box + confidence label ──────────────────────
    for cx, cy, conf, x1, y1, x2, y2 in caries:
        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Label with background
        label        = f"Caries {conf:.2f}"
        (tw, th), _  = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out,
                      (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 4, y1),
                      (0, 0, 200), -1)
        cv2.putText(out, label,
                    (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)

        # Centre dot
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

    return out


if __name__ == '__main__':
    import sys 
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/images/1194.png'
    detect_all(img_path)