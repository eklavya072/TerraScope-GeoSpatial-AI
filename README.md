# TerraScope — Geospatial AI Land Cover Classification

**TerraScope** is a satellite image land cover classification system powered by a deep learning model. It provides an interactive web interface for uploading satellite imagery, running real-time AI inference, and visualizing classification confidence across 10 EuroSAT land cover classes.

![Hero Screenshot]()

---

## Features

### Upload & Classify
Upload satellite or aerial imagery (PNG, JPG, JPEG, TIFF) and run inference against a trained ResNet50-based model. The app accepts images through a clean inline upload control and immediately makes them available for classification.

![Upload Interface]()

### Real-Time AI Predictions
Once an image is uploaded, click **Run Classification** to invoke the model. The system processes the image and returns the predicted land cover class along with a confidence score.

![Classification Result]()

### Confidence Score Chart
A horizontal bar chart displays the model's confidence for all 10 EuroSAT classes — AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. This gives immediate insight into the model's reasoning and how decisively it classified the image.

![Confidence Chart]()

### Status Tracking
The classification panel includes a real-time status indicator that transitions from IDLE → PROCESSING → COMPLETE, providing clear feedback on what the system is doing.

### Model Evaluation
The Charts section includes a model performance graph (training accuracy & loss curves) so users can assess the model's training history and generalization quality.

![Model Evaluation]()

### Image Analysis Tools
Analyze uploaded images beyond classification:
- **RGB Color Histograms** — Per-channel intensity distributions
- **Image Statistics** — Mean brightness, standard deviation, min/max values
- **Edge Detection** — Visualize edges using standard algorithms
- **Intensity Map** — Interactive Plotly-based intensity heatmap

![Image Analysis]()

### Land Cover Class Reference
The Classes page provides a browsable reference for all 10 EuroSAT categories, each with a description and representative sample image displayed in collapsible expanders.

![Classes Reference]()

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

The pretrained model weights (`ResNet50_eurosat.h5`) and class index mapping (`class_indices.npy`) are included in the repository under `models/`. If you need to retrain from scratch, use the provided `train.py` script or the `Train_on_Colab.ipynb` notebook.

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
├── assets/                   # Static images (class samples, hero, model perf chart)
├── models/                   # Trained model weights and class indices
│   ├── ResNet50_eurosat.h5
│   └── class_indices.npy
├── app.py                    # Main Streamlit application entry point
├── config.py                 # Class names, model & data configuration
├── model_handler.py          # TensorFlow model loading and inference
├── visualizer.py             # Plotly/Matplotlib chart generation
├── data_processor.py         # Image preprocessing utilities
├── train.py                  # Model training script
├── Train_on_Colab.ipynb      # Google Colab training notebook
└── requirements.txt          # Python dependencies
```

---

## Model Details

- **Architecture**: ResNet50 (pretrained on ImageNet, fine-tuned on EuroSAT)
- **Input shape**: 64 × 64 × 3 (RGB)
- **Output**: 10 EuroSAT land cover classes
- **Training data**: [EuroSAT](https://github.com/phelber/eurosat) dataset (Sentinel-2 satellite imagery)
- **Performance**: ~87.8% validation accuracy

---

## Configuration

Edit `config.py` to adjust:

- `CLASS_NAMES` — Class label mappings
- `MODEL_CONFIG` — Model path and input shape
- `DATA_CONFIG` — Allowed upload formats and max image dimensions

---

## License

This project is licensed under the MIT License.

---

**Repository**: [eklavya072/TerraScope-GeoSpatial-AI](https://github.com/eklavya072/TerraScope-GeoSpatial-AI)
