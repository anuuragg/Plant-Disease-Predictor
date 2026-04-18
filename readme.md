# 🌿 Plant Disease Predictor

A Streamlit app that classifies plant leaf diseases from images using a fine-tuned MobileNetV2 model trained on 38 classes.

## Project Structure

```
├── app.py
├── model/
│   ├── plant_disease_model.pt
│   └── labels.json
└── notebook/
    └── train.ipynb
```

## Setup

```bash
pip install streamlit torch torchvision pillow
streamlit run app.py
```

## Usage

Upload a leaf image (JPG/PNG) and the app returns the top 3 predicted disease classes with confidence scores.

## Model

- Architecture: MobileNetV2
- Classes: 38 plant disease categories
- Trained via: `train.ipynb` (PyTorch, HuggingFace Datasets)