"""
yolo_train.py
Train YOLOv8 for dental caries detection.

Run on Google Colab for best speed:
    !python yolo_train.py

Requirements:
    pip install ultralytics
"""

import os
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    'data'        : 'caries_dataset/data.yaml',
    'model'       : 'yolo26m.pt',
    'epochs'      : 80,              # reduced – finishes in ~40 min on Colab
    'imgsz'       : 640,
    'batch'       : 16,              # larger batch = more stable, faster
    'patience'    : 20,              # stop early if no improvement
    'lr0'         : 0.01,            # higher LR – reaches good solution faster
    'lrf'         : 0.1,             # final LR ratio
    'warmup_epochs': 5,              # warm up LR for first 5 epochs
    'weight_decay': 0.0005,
    'dropout'     : 0.0,             # YOLO handles regularization via augmentation

    # Augmentation
    'hsv_h'       : 0.0,             # no hue (X-rays are grayscale)
    'hsv_s'       : 0.0,             # no saturation
    'hsv_v'       : 0.5,             # brightness variation
    'flipud'      : 0.0,             # no vertical flip
    'fliplr'      : 0.5,             # horizontal flip
    'mosaic'      : 1.0,             # always mosaic
    'mixup'       : 0.15,
    'degrees'     : 5.0,
    'translate'   : 0.1,
    'scale'       : 0.3,
    'shear'       : 2.0,
    'copy_paste'  : 0.1,             # copy-paste augmentation for small objects

    'project'     : 'runs/detect/caries_runs',
    'name'        : 'train',
    'save'        : True,
    'save_period' : 10,              # save checkpoint every 10 epochs
}


def train():
    print("YOLO26m Caries Detection Training")
    print("=" * 50)

    # Verify dataset exists
    if not os.path.exists(CONFIG['data']):
        print(f"ERROR: {CONFIG['data']} not found!")
        print("Run first:  python yolo_prepare_dataset.py")
        return

    # Load pretrained YOLO26m model
    # Downloads automatically on first run (~22MB for yolov8s)
    print(f"Loading model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])

    print(f"Training for {CONFIG['epochs']} epochs...")
    print(f"Dataset: {CONFIG['data']}")
    print("=" * 50)

    # Train
    results = model.train(
        data          = CONFIG['data'],
        epochs        = CONFIG['epochs'],
        imgsz         = CONFIG['imgsz'],
        batch         = CONFIG['batch'],
        patience      = CONFIG['patience'],
        lr0           = CONFIG['lr0'],
        lrf           = CONFIG['lrf'],
        warmup_epochs = CONFIG['warmup_epochs'],
        weight_decay  = CONFIG['weight_decay'],
        hsv_h         = CONFIG['hsv_h'],
        hsv_s         = CONFIG['hsv_s'],
        hsv_v         = CONFIG['hsv_v'],
        flipud        = CONFIG['flipud'],
        fliplr        = CONFIG['fliplr'],
        mosaic        = CONFIG['mosaic'],
        mixup         = CONFIG['mixup'],
        degrees       = CONFIG['degrees'],
        translate     = CONFIG['translate'],
        scale         = CONFIG['scale'],
        shear         = CONFIG['shear'],
        copy_paste    = CONFIG['copy_paste'],
        project       = CONFIG['project'],
        name          = CONFIG['name'],
        save          = CONFIG['save'],
        save_period   = CONFIG['save_period'],
        verbose       = True,
    )

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best model saved to: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print("\nNext step → run:  python yolo_inference.py")


if __name__ == '__main__':
    train()
