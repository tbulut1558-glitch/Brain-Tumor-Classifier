Brain Tumor Classifier

A deep learning system for classifying brain tumors from MRI scans using ResNet50, transfer learning, and a Gradio web interface.
The model predicts four categories:

Glioma

Meningioma

Pituitary Tumor

No Tumor


Key Features

    End-to-end MRI classification pipeline

    ResNet50 backbone with two-stage fine-tuning

    98.17% test accuracy

    SavedModel deployment using TFSMLayer

    Gradio interface for real-time predictions


Project Structure

    project_root/
    │
    ├── data/
    │   └── raw/
    │       └── BrainTumor/                     # Dataset (train/test)
    │
    ├── models/
    │   └── brain_tumor_resnet_v1_savedmodel/   # Exported SavedModel
    │
    ├── notebooks/
    │   └── braintumor.ipynb                    # Training & fine-tuning notebook
    │
    ├── reports/
    │   └── ...                                 # Training graph, confusion matrix
    │
    ├── src/
    │   └── bt.py                               # Deployment (Gradio web app)
    │
    └── requirements.txt


Model Overview

This classifier is built on a deep Convolutional Neural Network (CNN) designed specifically for medical image analysis.

Base Model — ResNet50
The core of the model is ResNet50, a well-established architecture pretrained on ImageNet.
Its biggest strength comes from its Residual Blocks (skip connections), which help the network train smoothly without running into vanishing-gradient issues which is a common problem in deep networks.
During the first training phase, all ResNet50 layers were kept frozen, allowing the model to rely on the strong visual features it had already learned.

Custom Classification Head

The original classification layers of ResNet50 were removed (include_top=False) and replaced with a custom head tailored to the four tumor classes:

    Flatten – turns the 3D feature maps into a single vector.
    Dense (512, ReLU) – learns high-level tumor-specific patterns.
    Dropout (0.5) – reduces overfitting by randomly disabling half of the neurons during training.
    Dense (4, Softmax) – outputs the probability for each tumor class.

Two-Stage Fine-Tuning Strategy

To get the best performance while keeping the model stable, training was done in two phases:

    Phase 1	Train only the new classification head (ResNet50 frozen)	Helps the added layers adapt quickly to the pretrained features	1e-4
    Phase 2	Unfreeze the last 40 convolutional layers	Fine-tunes high-level features to better capture tumor characteristics	1e-5


Results

Test Accuracy: 98.17%
Test Loss: 0.0531

Deployment — Gradio Web App

Run the application:

    python src/bt.py


Installation & Usage

    1. git clone https://github.com/tbulut1558-glitch/Brain-Tumor-Classifier.git
    cd Brain-Tumor-Classifier

    2. (Optional) Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate     # macOS / Linux
    .\.venv\Scripts\activate      # Windows

    3. Install dependencies
    pip install -r requirements.txt

    4. Run the web interface
    python src/bt.py

Future Work

    Grad-CAM heatmaps for explainability
    TensorFlow Lite conversion for mobile/edge deployment
    
