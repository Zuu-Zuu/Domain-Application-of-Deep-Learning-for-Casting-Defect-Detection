# Domain-Application-of-Deep-Learning-for-Casting-Defect-Detection

This project implements an automated image classification system for detecting casting defects using a hybrid deep learning model that combines a Custom CNN with ResNet50. The goal is to enhance quality control in manufacturing by providing a fast, accurate, and scalable alternative to manual inspection.

---

## Project Overview

- **Problem**: Manual casting inspections are slow, error-prone, and not scalable.
- **Objective**: Build a robust model to classify grayscale images of castings into `Defective` or `OK`.
- **Solution**: A hybrid deep learning model combining Custom CNN and ResNet50, trained on a labeled industrial dataset.

---

## Model Architecture

- **Custom CNN**: Extracts domain-specific features unique to casting defects.
- **ResNet50**: Pre-trained network for general feature extraction.
- **Ensemble Design**: The output from the custom CNN is passed into ResNet50 for final classification.

---

## Dataset

- Source: Public dataset containing 7,348 grayscale images of submersible pump castings
- Split:
  - **Training**: 6,633 images  
  - **Testing**: 715 images
- Classes:
  - `ok_front` — non-defective
  - `def_front` — defective

---

## Preprocessing

- Image resizing to 224×224
- Normalization (mean = 0.5, std = 0.5)
- Data augmentation: horizontal flip, slight rotation (±10°)

---

## Training Details

- Framework: PyTorch
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Epochs: 10
- Batch Size: 32
- Accuracy: **Up to 99.72%**

---

## Evaluation Metrics

- Accuracy, Loss (Train & Validation)
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Sample predictions with confidence levels

---

## 🧾 License

This project is for academic and educational purposes.

---

## Acknowledgments

- Dataset sourced from [Kaggle: Casting Product Image Data](https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- Built using PyTorch and torchvision

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/casting-defect-classification.git
cd casting-defect-classification

# Install dependencies
pip install -r requirements.txt

# Run training or inference
python train.py
python predict.py --image_path path/to/test_image.jpeg
