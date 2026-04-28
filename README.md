# Air Pollution Mini-Model (ESP32-CAM)

This project contains a complete, end-to-end machine learning pipeline designed to train, evaluate, and deploy a highly optimized custom Convolutional Neural Network (CNN) specifically tailored for the ESP32-CAM microcontroller.

The overarching goal is to classify images into four distinct environmental categories (e.g., **Normal**, **Smoke**, **Dust**, **Fog**) achieving **90%+ accuracy**, while rigidly adhering to the ESP32's severe memory constraints. The final compiled model is engineered to be under **600 KB** (specifically ~398 KB) via full INT8 quantization.

---

## 🌟 Key Features

- **Custom Lightweight Architecture**: Built from scratch using Keras, carefully balancing feature extraction capabilities (Conv2D, MaxPooling) with parameter count reduction.
- **Robust Data Augmentation**: Integrates runtime image augmentation (rotations, zooms, flips, brightness tweaks) to prevent overfitting and improve real-world generalization.
- **Advanced Optimization Techniques**: Utilizes `EarlyStopping` and `ReduceLROnPlateau` callbacks to dynamically adjust learning rates and prevent overfitting during training.
- **Full INT8 Quantization**: Employs TFLite's representative dataset generation to quantize weights and activations from Float32 to INT8, drastically shrinking model size by ~4x with minimal accuracy loss.
- **Interactive Streamlit App**: A sleek web interface that simulates the microcontroller's exact INT8 preprocessing and inference logic, allowing instant drag-and-drop validation of the model.
- **C-Header Generation**: A direct pipeline to convert the binary `.tflite` model into a `model.h` C byte array, making firmware integration seamless.

---

## 📁 Project Structure

```text
MiniProjCustomCNN/
├── app.py                # Streamlit web interface for testing the model
├── convert_to_h.py       # Script to convert .tflite to model.h byte array
├── count.py              # Utility to tally dataset class distribution
├── train_pipeline.py     # Main training, evaluation, and quantization script
├── labels.txt            # Auto-generated label map from training
├── best_model.keras      # Best saved weights during training (unquantized)
├── model_quant.tflite    # Final INT8-quantized model for ESP32
├── model.h               # Final C-header file for ESP32 firmware
└── dataset/              # Image data directory (train, val, test)
```

---

## 🖼️ Dataset Setup

Before training, ensure your dataset is organized correctly. You must place your images inside the `dataset` folder, structured into `train`, `val`, and `test` splits.

```text
dataset/
├── train/
│   ├── Normal/
│   ├── Smoke/
│   ├── Dust/
│   └── Fog/
├── val/
│   ├── Normal/
...
└── test/
    ├── Normal/
...
```
*Tip: You can use `python count.py` to quickly verify the number of images distributed across your classes and splits.*

---

## ⚙️ Installation & Setup

1. **Clone the repository** (if applicable) and navigate to the project directory.
2. **Create a Virtual Environment** (Highly Recommended):
   ```bash
   python -m venv venv
   ```
3. **Activate the Environment**:
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`
4. **Install Dependencies**:
   ```bash
   pip install tensorflow streamlit pillow numpy
   ```

---

## 🚀 Usage Guide

### 1. Training & Quantization
The `train_pipeline.py` script handles model creation, training, evaluation, and TFLite conversion automatically.
```bash
python train_pipeline.py
```
**What happens during this step?**
- Loads the dataset and applies augmentations.
- Trains the CNN for up to 40 epochs (early stopping enabled).
- Evaluates against the test set.
- Generates a representative dataset to calibrate quantization ranges.
- Outputs `model_quant.tflite` (the final compact model).

*(Optional)* Run a quick architecture and size check without training:
```bash
python train_pipeline.py --dry-run
```

### 2. Local Testing (Streamlit)
Test your quantized model using the interactive Streamlit dashboard. The dashboard mirrors the ESP32's internal handling by simulating INT8 quantization on the uploaded image.
```bash
streamlit run app.py
```
- Open the provided `localhost` URL in your browser.
- Upload an image (jpg, jpeg, png).
- View the model's prediction and the calculated confidence scores.

### 3. ESP32 Deployment Preparation
Microcontrollers cannot load raw `.tflite` files directly from a filesystem in the same way a PC does. The model must be compiled directly into the firmware memory.
```bash
python convert_to_h.py
```
This generates a `model.h` file containing the model as a `const unsigned char` array.
- Copy `model.h` into your Arduino IDE or ESP-IDF project directory.
- Include the header file (`#include "model.h"`) in your main sketch.
- Use the TensorFlow Lite Micro library to load the array and run inference on the ESP32 camera feed.

---

## 📊 Model Architecture Overview

The CNN architecture is heavily optimized for edge deployment, prioritizing a low parameter count without sacrificing the representational power needed to distinguish between subtle environmental features like Fog and Dust.

### 1. Input Constraints
The model accepts **96x96 RGB images**. This specific resolution was chosen because it provides enough spatial detail for classification while keeping the input tensor small enough to fit into the ESP32's limited SRAM during inference.

### 2. Feature Extraction (Convolutional Blocks)
The network relies on four progressive Convolutional blocks to extract spatial hierarchies:
- **Block 1**: `Conv2D (16 filters)` + `MaxPooling` ➔ *Captures basic edges and colors. Reduces resolution to 48x48.*
- **Block 2**: `Conv2D (32 filters)` + `MaxPooling` ➔ *Captures textures. Reduces resolution to 24x24.*
- **Block 3**: `Conv2D (64 filters)` + `MaxPooling` ➔ *Identifies complex object parts. Reduces resolution to 12x12.*
- **Block 4**: `Conv2D (128 filters)` + `MaxPooling` ➔ *Abstract, high-level feature maps. Reduces resolution to 6x6.*

Each Conv2D layer utilizes 'same' padding and a `ReLU` activation function to introduce non-linearity efficiently.

### 3. Classification Head
- **Flatten Layer**: Unrolls the final 6x6x128 feature map into a 1D vector.
- **Dense Layer (64 units)**: A fully connected layer to interpret the extracted features.
- **Dropout (50%)**: Randomly disables half the neurons during training. This is critical for preventing the model from memorizing the training data (overfitting), forcing it to learn robust features.
- **Output Layer**: A final Dense layer with 4 units and a `softmax` activation to output a probability distribution across the 4 classes.

### 4. INT8 Quantization Strategy
By default, the trained Keras model uses 32-bit floating-point (Float32) numbers for weights, resulting in a model size of **~1.5 MB**.
Through TFLite's **Full Integer Quantization**, we convert all Float32 weights and activations into 8-bit integers (INT8). 
- **Size Reduction**: This shrinks the model size by roughly 4x, bringing it down to **~398 KB**, easily fitting in the ESP32-CAM's flash memory.
- **Speed**: INT8 operations are significantly faster to compute on microcontrollers lacking dedicated floating-point units.
