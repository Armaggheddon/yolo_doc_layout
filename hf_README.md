---
license: mit
datasets:
- docling-project/DocLayNet-v1.2
language:
- en
library_name: ultralytics
base_model:
- Ultralytics/YOLO26
pipeline_tag: object-detection
tags:
- object-detection
- document-layout
- yolov26
- ultralytics
- document-layout-analysis
- document-ai
---

# YOLOv26 for Advanced Document Layout Analysis

<p align="center">
  <img src="images/logo.png" alt="Logo" width="100%"/>
</p>

This repository hosts three YOLOv26 models (**nano, small, and medium**) fine-tuned for high-performance **Document Layout Analysis** on the challenging [DocLayNet v1.2 dataset](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2).

The goal is to accurately detect and classify key layout elements in a document, such as text, tables, figures, and titles. This is a fundamental task for document understanding and information extraction pipelines.

### ✨ Model Highlights
*   **🚀 Three Powerful Variants:** Choose between `nano`, `small`, and `medium` models to fit your performance needs.
*   **🎯 High Accuracy:** Trained on the comprehensive DocLayNet v1.2 dataset to recognize 11 distinct layout types.
*   ⚡ **Optimized for Efficiency:** The recommended **`yolo26n` (nano) model** offers an exceptional balance of speed and accuracy, making it ideal for production environments.

---

## 🚀 Get Started

Get up and running with just a few lines of code.

### 1. Installation

First, install the necessary libraries.

```bash
pip install ultralytics huggingface_hub
```

### 2. Inference Example

This Python snippet shows how to download a model from the Hub and run inference on a local document image.

```python
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Define the local directory to save models
DOWNLOAD_PATH = Path("./models")
DOWNLOAD_PATH.mkdir(exist_ok=True)

# Choose which model to use
# 0: nano, 1: small, 2: medium
model_files = [
    "yolo26n_doc_layout.pt",
    "yolo26s_doc_layout.pt",
    "yolo26m_doc_layout.pt",
]
selected_model_file = model_files[0] # Using the recommended nano model

# Download the model from the Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Armaggheddon/yolo26-document-layout",
    filename=selected_model_file,
    repo_type="model",
    local_dir=DOWNLOAD_PATH,
)

# Initialize the YOLO model
model = YOLO(model_path)

# Run inference on an image
# Replace 'path/to/your/document.jpg' with your file
results = model('path/to/your/document.jpg')

# Process and display results
results[0].print()  # Print detection details
results[0].show()   # Display the image with bounding boxes
```

---

## 📊 Model Performance & Evaluation

We fine-tuned three YOLOv26 variants, allowing you to choose the best model for your use case.

*   **`yolo26n_doc_layout.pt`**: **Recommended.** The nano model offers the best trade-off between speed and accuracy.
*   **`yolo26s_doc_layout.pt`**: A larger, slightly more accurate model.
*   **`yolo26m_doc_layout.pt`**: The largest model, providing the highest accuracy with a corresponding increase in computational cost.

As shown in the analysis below, performance gains are marginal when moving from the `small` to the `medium` model, making the `nano` and `small` variants the most practical choices.

### Nano vs. Small vs. Medium Comparison

Here's how the three models stack up across key metrics. The plots compare their performance for each document layout label.

| **mAP@50-95** (Strict IoU) | **mAP@50** (Standard IoU) |
| :---: | :---: |
| <img src="images/nsm_map50_95_per_label.png" alt="mAP@50-95" width="400"> | <img src="images/nsm_map50_per_label.png" alt="mAP@50" width="400"> |

| **Precision** (Box Quality) | **Recall** (Detection Coverage) |
| :---: | :---: |
| <img src="images/nsm_box_precision_per_label.png" alt="Precision" width="400"> | <img src="images/nsm_recall_per_label.png" alt="Recall" width="400"> |

<details>
<summary><b>Click to see detailed Training Metrics & Confusion Matrices</b></summary>

| Model | Training Metrics | Normalized Confusion Matrix |
| :---: | :---: | :---: |
| **`yolo26n`** | <img src="images/n_results.png" alt="yolo26n results" height="200"> | <img src="images/n_confusion_matrix_normalized.png" alt="yolo26n confusion matrix" height="200"> |
| **`yolo26s`** | <img src="images/s_results.png" alt="yolo26s results" height="200"> | <img src="images/s_confusion_matrix_normalized.png" alt="yolo26s confusion matrix" height="200"> |
| **`yolo26m`** | <img src="images/m_results.png" alt="yolo26m results" height="200"> | <img src="images/m_confusion_matrix_normalized.png" alt="yolo26m confusion matrix" height="200"> |

</details>


## 📚 About the Dataset: DocLayNet

The models were trained on the [DocLayNet v1.2 dataset](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2), which provides a rich and diverse collection of document images annotated with 11 layout categories:

*   **Text**, **Title**, **Section-header**
*   **Table**, **Picture**, **Caption**
*   **List-item**, **Formula**
*   **Page-header**, **Page-footer**, **Footnote**

**Training Resolution:** All models were trained at **1280x1280** resolution. Initial tests at the default 640x640 resulted in a significant performance drop, especially for smaller elements like `footnote` and `caption`.

<img src="images/class_distribution.jpg" alt="DocLayNet v1.2 Samples" width="500px"/>

## Yolo26 🆚 Yolo11
Comparing the new YOLOv26 models to the previous YOLOv11 baseline, we see significant improvements across all metrics, particularly in mAP@50-95 and recall. The `nano` model alone outperforms the `yolo11m` model, demonstrating the effectiveness of the YOLOv26 architecture for document layout analysis.

![YOLOv26 vs YOLOv11 Comparison](images/yolo_v11_vs_v26_comparison.png)

---

## 💻 Code & Training Details

This model card focuses on results and usage. For the complete end-to-end pipeline, including training scripts, dataset conversion utilities, and detailed examples, please visit the main GitHub repository:

➡️ **[GitHub Repo: yolo_doc_layout](https://github.com/Armaggheddon/yolo_doc_layout)**