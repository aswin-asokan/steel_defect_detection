# 🤖 Steel Defect Detection (SAM + LoRA)

## 🌟 Project Overview

This repository hosts a comprehensive solution for **automated steel surface defect detection and segmentation**. It utilizes state-of-the-art computer vision techniques, specifically fine-tuning the **Segment Anything Model (SAM)** using **LoRA (Low-Rank Adaptation)**, to accurately identify and delineate defects like pits, cracks, and scratches on steel surfaces.

The project includes both the machine learning model implementation (backend) and a cross-platform user interface (frontend) for practical deployment and visualization.

## ✨ Key Features

- **High Accuracy Segmentation:** Leverages the power of the Segment Anything Model (SAM) for precise defect localization.
- **Efficient Fine-Tuning:** Uses LoRA to fine-tune the SAM base model on custom steel defect datasets with minimal computational resources.
- **Full-Stack Application:** Includes a Python backend for model serving and a **Flutter** frontend for a user-friendly, cross-platform interface.
- **Kaggle Dataset Ready:** Built to handle the standard steel surface defect segmentation datasets (e.g., Severstal Steel Defect Detection).

<img width="1920" height="1080" alt="result" src="https://github.com/user-attachments/assets/cc070838-a104-4407-98e7-b0260641b87e" />

## 🚀 Technology Stack

| Component       | Technology                       | Description                                                |
| :-------------- | :------------------------------- | :--------------------------------------------------------- |
| **Model**       | **SAM (Segment Anything Model)** | Foundation model for image segmentation.                   |
| **Fine-Tuning** | **LoRA**                         | Efficiently adapts SAM to the steel domain data.           |
| **Backend/ML**  | **Python**, **PyTorch**          | Training, testing, and serving the defect detection model. |
| **Frontend**    | **Dart**, **Flutter**            | Provides a responsive, cross-platform GUI for inference.   |

## 🛠️ Installation and Setup

### 1. Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.8+**
- **Flutter SDK** (for the frontend)
- **Git**

### 2. Clone the Repository

```bash
git clone [https://github.com/aswin-asokan/steel_defect_detection.git](https://github.com/aswin-asokan/steel_defect_detection.git)
cd steel_defect_detection
```

### Backend Setup

The machine learning core is built with Python.

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Linux/macOS
# venv\Scripts\activate # On Windows
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Frontend Setup

The user interface is a Flutter application.

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Get packages and ensure Flutter is ready:

```bash
flutter pub get
flutter doctor
```

### 🧠 Usage

#### A. Training the Model

1. Ensure your steel defect images and segmentation masks are placed within the `./dataset` directory following the structure expected by the `train1.py` script.

2. Run the training script, which will apply the LoRA fine-tuning to the base SAM model:

```bash
python train1.py
```

The trained LoRA weights will be saved in the `./sam_steel_lora` directory.

#### B. Running the Backend Server

The `backend.py` file exposes an API endpoint for performing defect detection inference.

1. Start the backend server:

```bash
python backend.py
```

#### C. Running the Frontend Application

The Flutter application connects to the Python backend to perform inference.

1. Ensure the backend server is running (see step B).

2. From the `./frontend` directory, run the Flutter app on your desired platform (preferably web):

```bash
flutter run
```

3. Upload a steel surface image in the application to view the segmented defect masks.

### 📂 Project Structure

```
steel_defect_detection/
├── dataset/                     # Directory for training and validation data (images and masks)
├── frontend/                    # Flutter/Dart cross-platform user interface
├── sam_steel_lora/              # Saved LoRA weights for the fine-tuned SAM model
├── sample/                      # Sample input/output images
├── .gitignore                   # Standard ignore file
├── backend.py                   # Python script to serve the trained model as an API
├── requirements.txt             # Python dependencies list
├── test1.py                     # Script for evaluating the model on a test set
└── train1.py                    # Script for training and fine-tuning the SAM model with LoRA
```

### 📔 Citations

**Base Paper:** [Few-Shot Parameter Efficient Finetuning for SAM in Salient Steel Surface Defect Detection](https://ieeexplore.ieee.org/document/11062120)

```bibtext
@ARTICLE{11062120,
  author={Su, Jiaojiao and Luo, Qiwu and Gui, Weihua and Yang, Chunhua},
  journal={IEEE Transactions on Industrial Informatics},
  title={Few-Shot Parameter Efficient Finetuning for SAM in Salient Steel Surface Defect Detection},
  year={2025},
  volume={21},
  number={10},
  pages={7742-7753},
  keywords={Steel;Strips;Defect detection;Decoding;Transformers;Visualization;Training;Biomedical imaging;Adaptation models;Remote sensing;Defect detection;finetune segment anything model (SAM);parameter-efficient fine-tuning (PEFT);strip steel},
  doi={10.1109/TII.2025.3574815}}
```

**Dataset:** [SD-saliency-900](https://www.kaggle.com/datasets/alex000kim/sdsaliency900)
