"""
yolo_kfold.py
5-fold cross validation for YOLOv8 caries detection.
Gives reliable mAP50, Precision, Recall for supervisor report.

Run on Colab:
    !python yolo_kfold.py
"""

import os
import shutil
import random
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    'image_dir'  : 'caries_dataset/images/train',
    'label_dir'  : 'caries_dataset/labels/train',
    'n_folds'    : 5,
    'epochs'     : 100,         # shorter per fold to save time
    'imgsz'      : 640,
    'batch'      : 8,
    'model'      : 'yolov8m.pt',
    'seed'       : 42,
}


def run_kfold():
    print("YOLOv8 5-FOLD CROSS VALIDATION")
    print("=" * 60)

    # Get all image stems
    img_exts = {'.jpg', '.jpeg', '.png'}
    stems    = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(CONFIG['image_dir'])
        if os.path.splitext(f)[1].lower() in img_exts
    ])

    random.seed(CONFIG['seed'])
    random.shuffle(stems)

    n         = len(stems)
    fold_size = n // CONFIG['n_folds']
    print(f"Total images: {n}  |  Fold size: ~{fold_size}")

    fold_results = []

    for fold in range(CONFIG['n_folds']):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{CONFIG['n_folds']}")
        print(f"{'='*60}")

        # Split stems
        val_stems   = stems[fold * fold_size : (fold + 1) * fold_size]
        train_stems = [s for s in stems if s not in val_stems]
        print(f"Train: {len(train_stems)}  Val: {len(val_stems)}")

        # Create fold dataset folder
        fold_dir = f'kfold_data/fold_{fold+1}'
        for split in ['train', 'val']:
            os.makedirs(f'{fold_dir}/images/{split}', exist_ok=True)
            os.makedirs(f'{fold_dir}/labels/{split}', exist_ok=True)

        # Find correct image extension for each stem
        def find_img(stem):
            for ext in ['.jpg', '.jpeg', '.png']:
                p = os.path.join(CONFIG['image_dir'], stem + ext)
                if os.path.exists(p):
                    return p, ext
            return None, None

        # Copy files to fold folders
        for split_stems, split in [(train_stems, 'train'),
                                    (val_stems,   'val')]:
            for stem in split_stems:
                img_path, ext = find_img(stem)
                lbl_path      = os.path.join(CONFIG['label_dir'],
                                             stem + '.txt')
                if img_path:
                    shutil.copy2(img_path,
                                 f'{fold_dir}/images/{split}/{stem}{ext}')
                if os.path.exists(lbl_path):
                    shutil.copy2(lbl_path,
                                 f'{fold_dir}/labels/{split}/{stem}.txt')

        # Write data.yaml for this fold
        yaml_path = f'{fold_dir}/data.yaml'
        abs_dir   = os.path.abspath(fold_dir)
        with open(yaml_path, 'w') as f:
            f.write(f"path  : {abs_dir}\n"
                    f"train : images/train\n"
                    f"val   : images/val\n"
                    f"nc    : 1\n"
                    f"names : ['caries']\n")

        # Train this fold
        model   = YOLO(CONFIG['model'])
        results = model.train(
            data         = yaml_path,
            epochs       = CONFIG['epochs'],
            imgsz        = CONFIG['imgsz'],
            batch        = CONFIG['batch'],
            project      = f'kfold_runs',
            name         = f'fold_{fold+1}',
            verbose      = False,
            hsv_h        = 0.0,
            hsv_s        = 0.0,
            hsv_v        = 0.5,
            fliplr       = 0.5,
            mosaic       = 1.0,
            mixup        = 0.2,
            degrees      = 5.0,
            scale        = 0.3,
        )

        # Validate
        val_results = model.val(data=yaml_path, verbose=False)
        metrics = {
            'fold'     : fold + 1,
            'mAP50'    : val_results.box.map50,
            'mAP50_95' : val_results.box.map,
            'precision': val_results.box.mp,
            'recall'   : val_results.box.mr,
        }
        fold_results.append(metrics)

        print(f"Fold {fold+1} → "
              f"mAP50={metrics['mAP50']:.4f}  "
              f"Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)

    for metric in ['mAP50', 'mAP50_95', 'precision', 'recall']:
        vals = [r[metric] for r in fold_results]
        print(f"{metric:<15} "
              f"{np.mean(vals):>8.4f} "
              f"{np.std(vals):>8.4f} "
              f"{np.min(vals):>8.4f} "
              f"{np.max(vals):>8.4f}")

    print(f"\n{'='*60}")
    print("Per-fold breakdown:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: "
              f"mAP50={r['mAP50']:.4f}  "
              f"Prec={r['precision']:.4f}  "
              f"Rec={r['recall']:.4f}")

    # Save summary to text file
    with open('kfold_results.txt', 'w') as f:
        f.write("YOLOv8 5-FOLD CROSS VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n")
        for metric in ['mAP50', 'mAP50_95', 'precision', 'recall']:
            vals = [r[metric] for r in fold_results]
            f.write(f"{metric}: mean={np.mean(vals):.4f} "
                    f"std={np.std(vals):.4f}\n")
        f.write("\nPer-fold:\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']}: mAP50={r['mAP50']:.4f} "
                    f"Prec={r['precision']:.4f} "
                    f"Rec={r['recall']:.4f}\n")

    print("\nSaved: kfold_results.txt")
    return fold_results


if __name__ == '__main__':
    run_kfold()
