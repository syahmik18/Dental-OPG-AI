# DentalScan AI 🦷

An AI-based dental OPG (orthopantomogram / panoramic X-ray) analysis system that combines two deep learning models to assist in dental diagnostics: **caries detection** and **mandibular nerve canal segmentation**. Built as my Final Year Project, with a working demo deployed as a Streamlit web app.

## Overview

Dentists reading panoramic X-rays need to identify caries (cavities) and locate the mandibular nerve canal — critical for procedures like implant placement, where nerve damage is a serious risk. This project automates both tasks using a dual-model pipeline trained on 445 annotated OPG images.

| Task | Model | Key Metric |
|---|---|---|
| Caries detection | YOLO26m | mAP50: 0.4705, Precision: 0.6498 |
| Nerve canal segmentation | Attention U-Net | Dice: 0.5335, IoU: 0.3693, Recall: 0.6726 |

## How it works

1. **Caries Detection** — a YOLO26m object detector trained on labeled OPG images flags likely caries locations with bounding boxes.
2. **Nerve Canal Segmentation** — an Attention U-Net produces a pixel-level segmentation mask of the mandibular nerve canal, with post-processing to clean up and refine the predicted path.
3. **Integrated Pipeline** — both outputs are combined into a single annotated image, giving a combined view of caries risk and nerve canal position for safer treatment planning.

## Project structure

```
ml/
├── app.py                     # Streamlit web app entry point
├── docs/                      # Architecture diagrams, workflow, sample results
├── evaluation/
│   └── evaluate_and_visualize.py
└── src/
    ├── caries/                # YOLO training, inference, dataset prep, k-fold validation
    │   ├── yolo_train_V2.py
    │   ├── yolo_inference_v2.py
    │   ├── yolo_prepare_dataset.py
    │   └── yolo_kfold.py
    ├── nerve/                 # Attention U-Net model, training, inference, post-processing
    │   ├── attention_unet_model.py
    │   ├── attention_unet_v2.py
    │   ├── unet_train_v3.py
    │   ├── unet_inference_v2.py
    │   └── nerve_postprocessing_updated.py
    └── pipeline/
        └── final_integrated_detection_v3.py   # Combines both models' outputs
```

## Dataset

- 445 OPG (panoramic dental X-ray) images
- Annotated using LabelMe (segmentation masks for nerve canals, bounding boxes for caries)
- Training conducted on Google Colab and Kaggle (T4×2 GPUs)

## Tech stack

- **Python**, **PyTorch** (Attention U-Net), **Ultralytics YOLO** (YOLO26m)
- **Streamlit** for the interactive web demo
- **LabelMe** for dataset annotation

## Running the demo

```bash
pip install -r requirements.txt
streamlit run ml/app.py
```

> Note: trained model weights (`.pth` / `.pt`) are not included in this repo due to file size — see below if you'd like to reproduce results or request the weights.

## Results

See `docs/` for the architecture diagram, training workflow, and sample test predictions.

## Acknowledgements

Final Year Project completed as part of my Computer Science degree, supervised by Dr. Shakirah Binti Hashim.
