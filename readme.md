# Steel Surface Defect Detection and Segmentation

Real-time steel surface defect inspection using a hybrid computer vision pipeline:

- YOLO26s for fast defect localization and type detection
- MobileSAM for pixel-level segmentation on detected regions
- SAM ViT-B + LoRA as a full segmentation baseline
- Flask backend for inference, video processing, logging, and pattern analysis
- Flutter frontend for live monitoring from a local backend camera or mobile camera upload

![Example result](https://github.com/user-attachments/assets/69b7041b-3508-4e06-ae70-17ef3ce06b85)

## Current Project Status

The repository is currently organized as a full-stack prototype for training, testing, and running steel defect detection in real time.

<img width="1162" height="536" alt="arch" src="https://github.com/user-attachments/assets/195d0000-77d0-4125-8e58-514b71297941" />


| Area | Current implementation |
| --- | --- |
| Detection | YOLO26s trained on NEU-DET style bounding-box annotations |
| Segmentation | Fine-tuned MobileSAM for efficient defect masks |
| Baseline segmentation | SAM ViT-B with LoRA adapters saved under `backend/sam_steel_lora/` |
| Hybrid runtime | YOLO26 first, then MobileSAM only on the union ROI of detected boxes |
| Backend | Flask API in `backend/backend.py` with model switching, live camera snapshots, upload inference, logs, and pattern summaries |
| Frontend | Flutter app in `frontend/` with live method switching and detection status panels |
| Logging | CSV, JSON, raw frames, overlays, masks, and pattern summaries under `backend/logs/` |

## Methods

### YOLO26 Detection

YOLO26s is used as the real-time screening model. It detects six NEU-DET defect classes:

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled_in_scale`
- `scratches`

The training script is [backend/yolo26/train_neudet.py](backend/yolo26/train_neudet.py). Pascal VOC annotations can be converted to YOLO format with [backend/yolo26/voc_to_yolo.py](backend/yolo26/voc_to_yolo.py).

YOLO26s performance:

<img width="2400" height="1200" alt="YOLO_result" src="https://github.com/user-attachments/assets/f5392021-63e3-4c9a-a940-d4d5d3aaf8f1" />


| Metric | Value |
| --- | --- |
| Precision | about 0.70-0.72 |
| Recall | about 0.70 |
| mAP@50 | 0.753 |
| mAP@50-95 | about 0.44 |
| Video speed | over 30 FPS |

### MobileSAM Segmentation

MobileSAM is used for efficient pixel-level segmentation. The current training approach freezes the image encoder and fine-tunes the prompt encoder and mask decoder. The loss combines Dice, Focal, and Boundary losses:

```text
0.6 Dice + 0.3 Focal + 0.1 Boundary
```

Training script: [backend/train_mobile_sam.py](backend/train_mobile_sam.py)

MobileSAM performance:

<img width="1600" height="533" alt="training_performance" src="https://github.com/user-attachments/assets/c6b134dd-986c-4710-bf2c-15c36c3e2571" />


| Metric | Value |
| --- | --- |
| Validation IoU | about 0.56-0.57 peak |
| Dice score | about 0.68-0.69 peak |
| Video speed | about 8-10 FPS |

### SAM + LoRA Baseline

The SAM baseline uses `facebook/sam-vit-base` with LoRA adapters applied to attention and feed-forward layers. Only adapter parameters are trained while the base model remains frozen.

Training script: [backend/train_sam.py](backend/train_sam.py)

Saved adapters are in:

```text
backend/sam_steel_lora/
backend/sam_steel_lora/checkpoint_epoch_2/
backend/sam_steel_lora/checkpoint_epoch_4/
backend/sam_steel_lora/checkpoint_epoch_6/
```

SAM + LoRA performance:

<img width="800" height="500" alt="training_progress_SAM" src="https://github.com/user-attachments/assets/85f33628-db6a-48f1-860d-deebba72850a" />

| Metric | Value |
| --- | --- |
| Validation IoU | about 0.60 |
| Video speed | about 1-3 FPS |

### Hybrid Real-Time Pipeline

The preferred deployment mode is:

```text
camera/image frame -> YOLO26s detection -> union bounding-box ROI -> MobileSAM mask -> overlay + logs + pattern analysis
```

This keeps inference fast by avoiding segmentation on frames where no defect is detected.

### Pattern Analysis

The backend records detection events and analyzes recurring defects over time. Pattern analysis groups detections into fixed time windows, counts defect labels, and flags windows where one defect type dominates. This is intended to help identify repeated process issues such as misalignment, contamination, or recurring manufacturing faults.

Implementation: [backend/pattern_service.py](backend/pattern_service.py)

## Project Structure

```text
steel_defect_detection/
|-- Journal_Paper.pdf
|-- readme.md
|-- backend/
|   |-- backend.py                    # Flask inference server and live camera worker
|   |-- pattern_service.py            # CSV-based recurring defect analysis
|   |-- train_sam.py                  # SAM ViT-B + LoRA training
|   |-- train_mobile_sam.py           # MobileSAM fine-tuning
|   |-- requirements.txt
|   |-- dataset/
|   |   |-- source_images/            # Segmentation training images
|   |   `-- ground_truth/             # Segmentation masks
|   |-- sam_steel_lora/               # Saved SAM LoRA adapters
|   |-- mobilesam_defect_optimized/   # MobileSAM best/final weights
|   |-- yolo26/
|   |   |-- train_neudet.py
|   |   |-- voc_to_yolo.py
|   |   |-- NEU-DET/                  # Original XML/image dataset layout
|   |   |-- neu_det/                  # YOLO-formatted dataset
|   |   `-- runs/                     # YOLO training outputs
|   |-- sample/                       # Sample images and visual results
|   |-- logs/                         # Runtime logs, masks, overlays, pattern JSON
|   `-- others/                       # Experiment and test scripts
`-- frontend/
    |-- lib/
    |   |-- main.dart
    |   `-- frontend.dart
    `-- pubspec.yaml
```

## Setup

### Backend

Run backend commands from the `backend/` directory.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The backend expects model assets in these locations, or compatible fallback paths defined in `backend.py`:

```text
backend/yolo26/runs/detect/train5/weights/best.pt
backend/yolo26/runs/yolo26s_ppy/exp_light/weights/best.pt
backend/yolo26/yolo26s.pt
backend/mobile_sam.pt
backend/MobileSAM/weights/mobile_sam.pt
backend/mobilesam_defect_optimized/best_model.pth
backend/mobilesam_defect_optimized/final_model.pth
backend/sam_steel_lora/
```

### Frontend

```bash
cd frontend
flutter pub get
```

The Flutter app currently points to:

```text
http://127.0.0.1:5000
```

For mobile-device camera upload, update `apiHost` in [frontend/lib/frontend.dart](frontend/lib/frontend.dart) to a reachable backend URL.

## Training

### 1. Prepare segmentation data

Both SAM and MobileSAM training expect:

```text
backend/dataset/
|-- source_images/
`-- ground_truth/
```

Image and mask filenames should share the same stem.

### 2. Train SAM + LoRA

```bash
cd backend
python train_sam.py
```

Outputs are saved to `backend/sam_steel_lora/`.

### 3. Train MobileSAM

Place `mobile_sam.pt` in `backend/` or `backend/MobileSAM/weights/`, then run:

```bash
cd backend
python train_mobile_sam.py
```

Outputs are saved to `backend/mobilesam_defect_optimized/`.

### 4. Prepare NEU-DET for YOLO26

Expected original dataset layout:

```text
backend/yolo26/NEU-DET/
|-- ANNOTATIONS/
`-- IMAGES/
```

Convert XML annotations to YOLO labels:

```bash
cd backend/yolo26
python voc_to_yolo.py
```

This creates:

```text
backend/yolo26/neu_det/
|-- images/train/
|-- images/val/
|-- labels/train/
|-- labels/val/
`-- data.yaml
```

### 5. Train YOLO26

```bash
cd backend/yolo26
python train_neudet.py
```

## Running Inference

### Backend server

```bash
cd backend
source .venv/bin/activate
python backend.py
```

By default, the backend starts a local camera worker using camera index `0`. To run the API without opening the local camera:

```bash
ENABLE_LOCAL_CAMERA=0 python backend.py
```

The server runs on:

```text
http://127.0.0.1:5000
```

### Flutter app

In a second terminal:

```bash
cd frontend
flutter run
```

The app supports these live methods:

- `SAM`
- `mobileSAM`
- `yolo26`
- `yolo26+mobileSAM`

## Backend API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/methods` | GET | List supported methods |
| `/switch/options` | GET | List frontend switch options |
| `/switch` | POST | Switch live backend mode |
| `/config` | GET/POST | Read or update runtime config |
| `/models/status` | GET | Inspect model path candidates and active models |
| `/snapshot` | GET | Get latest local-camera processed frame |
| `/predict` | POST | Upload one image for inference |
| `/video_feed` | GET | MJPEG stream from latest backend frames |
| `/logs/current` | GET | Return active CSV/JSON log paths |
| `/pattern/manual` | POST | Generate pattern summary from a detection CSV |
| `/pattern/latest` | GET | Return latest generated pattern summary |

Example upload inference:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "option=yolo26_mobilesam" \
  -F "image=@sample/In_4.bmp"
```

## Runtime Outputs

Hybrid detections are logged under `backend/logs/`:

```text
backend/logs/session_<timestamp>.csv
backend/logs/detections_<timestamp>.json
backend/logs/raw/
backend/logs/overlay/
backend/logs/mask/
backend/logs/pattern/
```

The CSV includes timestamps, frame indices, saved image paths, SAM confidence, defect area, YOLO labels, confidences, and bounding boxes.

## Datasets Referenced

- **SD-Saliency-900**: pixel-level steel surface defect segmentation data used for SAM/MobileSAM training.
- **NEU-DET**: bounding-box steel surface defect dataset used for YOLO26 training.

## Citation / Paper Context

Additional referenced work:

```bibtex
@ARTICLE{11062120,
  author={Su, Jiaojiao and Luo, Qiwu and Gui, Weihua and Yang, Chunhua},
  journal={IEEE Transactions on Industrial Informatics},
  title={Few-Shot Parameter Efficient Finetuning for SAM in Salient Steel Surface Defect Detection},
  year={2025},
  volume={21},
  number={10},
  pages={7742-7753},
  doi={10.1109/TII.2025.3574815}
}
```
