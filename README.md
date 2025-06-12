# 🐶 Dog Breed Classification using Transfer Learning
![Screenshot 2025-04-09 104857](https://github.com/user-attachments/assets/c500b7fc-cf9e-426e-9283-d4c121e491c2)
![Screenshot 2025-04-09 104933](https://github.com/user-attachments/assets/7bf8e925-7889-40b0-86e0-f4f00ef4c49d)

## 📌 Project Overview

This project presents an efficient deep learning solution to the **dog breed classification** problem, a fine-grained image recognition task. Using **transfer learning** with the pre-trained **InceptionResNetV2** model, the system classifies dog images into **120 distinct breeds** from the Stanford Dogs Dataset.

The model is designed for high precision, robust generalization, and practical applications in:
- Pet identification
- Animal welfare systems

---

## 🧠 Model Highlights

- **Backbone Model**: InceptionResNetV2 (pre-trained on ImageNet)
- **Classifier**: Custom Dense Neural Network with Dropout regularization
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Regularization**: Dropout + EarlyStopping
- **Evaluation Metrics**: Accuracy, Validation Accuracy, Loss

---

## 📁 Dataset

The project uses the [Stanford Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification/data), containing:

- 📸 10,222 labeled images
- 🐕 120 different breeds
- 🧭 Bounding box annotations and labels

---

## ⚙️ Preprocessing Pipeline

- ✅ Image resizing to **299x299**
- ✅ Normalization to pixel values in [0, 1]
- ✅ Data augmentation: random flip, rotation, zoom, shear
- ✅ Oversampling using `RandomOverSampler` to handle class imbalance
- ✅ Flattening InceptionResNetV2 feature maps for dense classifier

---

## 🧪 Training Performance

| Metric        | Value         |
|---------------|---------------|
| Training Accuracy  | **92.67%** |
| Validation Accuracy| **86.90%** |
| Loss (Train/Val)   | 0.3195 / 0.8465 |

<p align="center">
  <img src="images/accuracy_plot.png" width="450"/> <img src="images/loss_plot.png" width="450"/>
  <br>
  <em>Training and Validation Accuracy & Loss</em>
</p>

---

## 🐾 Model Architecture

```text
InceptionResNetV2 (Frozen)
      ↓
Flatten Layer
      ↓
Dense(512) + ReLU + Dropout
      ↓
Dense(512) + ReLU + Dropout
      ↓
Dense(120) + Softmax
