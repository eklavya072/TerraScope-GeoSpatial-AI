# TerraScope — Geospatial AI Land Cover Classification

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/model-ResNet50%20%2895.67%25%20acc%29-informational)](#model-details)
[![License: MIT](https://img.shields.io/github/license/eklavya072/TerraScope-GeoSpatial-AI)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/eklavya072/TerraScope-GeoSpatial-AI)](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/commits/master)

**TerraScope** is a satellite image land cover classification system powered by a deep learning model. It provides an interactive web interface for uploading satellite imagery, running real-time AI inference, and visualizing classification confidence across 10 EuroSAT land cover classes.

---

## Features

### Upload & Classify
Upload satellite or aerial imagery (PNG, JPG, JPEG, TIFF) and run inference against a fine-tuned ResNet50-based model. The app accepts images through a clean inline upload control and immediately makes them available for classification.

<img width="696" height="725" alt="Screenshot 2026-06-30 at 2 17 59 AM" src="https://github.com/user-attachments/assets/e40c863c-b09f-44c1-98a2-dd6025b9c1de" />

### Real-Time AI Predictions
Once an image is uploaded, click **Run Classification** to invoke the model. The system processes the image and returns the predicted land cover class along with a confidence score.

<img width="2938" height="1878" alt="image" src="https://github.com/user-attachments/assets/aafa194f-8b40-40c6-bef0-b037feeff9f3" />


### Confidence Score Chart
A horizontal bar chart displays the model's confidence for all 10 EuroSAT classes — AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. This gives immediate insight into the model's reasoning and how decisively it classified the image.

<img width="1068" height="1034" alt="image" src="https://github.com/user-attachments/assets/bdf1a581-7676-4064-ba25-44aa4b2811cd" />


### Status Tracking
The classification panel includes a real-time status indicator that transitions from IDLE → PROCESSING → COMPLETE, providing clear feedback on what the system is doing.

### Model Evaluation
The Charts section includes a model performance graph (training accuracy & loss curves) so users can assess the model's training history and generalization quality.

<img width="3366" height="1886" alt="image" src="https://github.com/user-attachments/assets/cb5d845c-f73e-41e2-b2e4-173c0e71e080" />


### Image Analysis Tools
Analyze uploaded images beyond classification:
- **RGB Color Histograms** — Per-channel intensity distributions
- **Image Statistics** — Mean brightness, standard deviation, min/max values
- **Edge Detection** — Visualize edges using standard algorithms
- **Intensity Map** — Interactive Plotly-based intensity heatmap

  
<img width="2924" height="2024" alt="image" src="https://github.com/user-attachments/assets/679d2c4f-5380-41c5-a780-54748b5b5180" />


<img width="2916" height="1876" alt="image" src="https://github.com/user-attachments/assets/51b1203b-50c0-4057-b59a-0297e30afb6b" />


### TerraScope Class Reference
The Classes page provides a browsable reference for all 10 EuroSAT categories, each with a description and representative sample image displayed in collapsible expanders.

<img width="3396" height="1894" alt="image" src="https://github.com/user-attachments/assets/f478cb64-6884-4ec3-8a31-b333fa542085" />


---

## Tech Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| Frontend     | [Streamlit](https://streamlit.io)       |
| Backend      | Python 3.9+                             |
| Model        | ResNet50 fine-tuned on EuroSAT          |
| Framework    | TensorFlow / Keras                      |
| Charts       | Plotly, Matplotlib                      |
| Styling      | Custom CSS, Material Symbols, Inter font|

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/eklavya072/TerraScope-GeoSpatial-AI.git
cd TerraScope-GeoSpatial-AI

# Install dependencies
pip install -r requirements.txt
```

### Download the Model

The pretrained model weights (`resnet50_eurosat_ft.h5`) and class index mapping (`class_indices.npy`) are included in the repository under `models/`. If you need to retrain from scratch, use the provided `train_resnet50.py` / `finetune_resnet50.py` scripts or the `Train_ResNet50_Colab.ipynb` notebook.

### Run the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` (or the URL shown in the terminal).

---

## Usage

1. **Upload an image** — Click the `+` button or drag a file into the upload area on the Home page.
2. **Preview** — The uploaded image appears in the Image Preview panel.
3. **Classify** — Click the **Run Classification** button.
4. **Review results** — The predicted class, confidence score, and a full confidence bar chart are displayed.
5. **Explore** — Switch to the Charts tab for model evaluation metrics and advanced image analysis tools.

---

## Project Structure

```
├── .streamlit/
│   └── config.toml           # Streamlit server configuration
├── assets/                   # Static images (class samples, satellite banner, model perf chart)
├── models/                   # Trained model weights and class indices
│   ├── resnet50_eurosat.h5      # ResNet50, frozen backbone (head-only training)
│   ├── resnet50_eurosat_ft.h5   # ResNet50, fine-tuned (served model)
│   └── class_indices.npy
├── app.py                    # Main Streamlit application entry point
├── config.py                 # Class names, model & data configuration
├── model_handler.py          # TensorFlow model loading and inference
├── visualizer.py             # Plotly/Matplotlib chart generation
├── train_resnet50.py         # ResNet50 training script (frozen backbone)
├── finetune_resnet50.py      # ResNet50 fine-tuning script (unfreezes last stage)
├── Train_ResNet50_Colab.ipynb # Google Colab training notebook
└── requirements.txt          # Python dependencies
```

---

## Model Details

- **Architecture**: ResNet50 (pretrained on ImageNet, fine-tuned on EuroSAT — last stage unfrozen)
- **Input shape**: 64 × 64 × 3 (RGB)
- **Output**: 10 EuroSAT land cover classes
- **Training data**: [EuroSAT](https://github.com/phelber/eurosat) dataset (Sentinel-2 satellite imagery)
- **Performance**: 95.67% test accuracy after fine-tuning

---

## Configuration

Edit `config.py` to adjust:

- `CLASS_NAMES` — Class label mappings
- `MODEL_CONFIG` — Model path and input shape
- `DATA_CONFIG` — Allowed upload formats and max image dimensions

---

## License

This project is licensed under the [MIT License](LICENSE).

---

**Repository**: [eklavya072/TerraScope-GeoSpatial-AI](https://github.com/eklavya072/TerraScope-GeoSpatial-AI)
