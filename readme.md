# Fashion Multi-Task Tagging & Stylist App
A compact project that trains multi-task classification models on a fashion dataset and serves predictions via a Streamlit web app.

## 📖 Table of Contents

- [Dataset](#-dataset)
- [Features](#-features)
- [Evaluation Results & Model Rationale](#-evaluation-results--model-rationale)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Running the App](#-running-the-app)


---

## 📊 Dataset

A brief overview of the fashion dataset used for training:

- **Source & Format**: Data originates from a curated `styles.csv` metadata file containing image filenames and 7 attribute columns: `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, and `usage`.
- **Size**: \~44,000 images stored under `data/images/`, each linked via an `image` column in the CSV.
- **Preprocessing Steps**:
  1. Removed classes with fewer than 50 samples to reduce noise.
  2. Normalized label distributions by merging rarely-used categoriesing.
  3. Resized images to a uniform target size (`TARGET_SIZE`) and applied standard normalization (`MEAN`, `STD`).

This dataset powers our multi-task models, ensuring balanced learning across all 7 fashion attributes.

## 🚀 Features

- **Data Preprocessing & EDA**: Cleans and visualizes `styles.csv`, removes noisy categories, merges and normalizes labels.
- **Multi-Task Learning**: Implements three PyTorch models (ResNet50, MobileNetV2, ResNet18) with shared backbones and separate classification heads for Mentioned 7 attributes
- **Training & Evaluation**: Splits data (80/20), applies transforms, tracks per-task and total loss, and prints classification reports with weighted F1.
- **Streamlit Front-End**: Upload an image, get predicted attributes with confidences, ask styling questions via LLM, or generate short captions.

---
## 📈 Evaluation Results & Model Rationale

### 📊 Performance Overview

After training, both ResNet variants were evaluated on the held-out 20% test split, reporting weighted F1-scores per attribute:

| Attribute      | ResNet18 F1   | ResNet50 F1   |
| -------------- | -----------   | -----------   |
| gender         | 0.915         | 0.779         |
| masterCategory | 0.996         | 0.937         |
| subCategory    | 0.969         | 0.880         |
| articleType    | 0.912         | 0.772         |
| baseColour     | 0.693         | 0.576         |
| season         | 0.752         | 0.584         |
| usage          | 0.917         | 0.831         |
| **Average**    | **0.879**     | **0.766**     |

### 🧠 Model Choice & Task Allocation

Based on these results, we revised our backbone-task assignment with the following observations:

1. **ResNet18 tends to outperform ResNet50 across most attributes, suggesting ResNet50 may be overfitting for some tasks.**
   ResNet18 achieves notably higher F1 scores for attributes like `gender`, `masterCategory`, `subCategory`, and `articleType`. This indicates that a lighter backbone can generalize better on these attributes, likely because ResNet50’s greater complexity leads to overfitting on the available data.

2. **ResNet50 performs consistently worse on `baseColour` and `gender`, with the gap especially large for color prediction.**
   This suggests ResNet50 struggles with the color attribute, possibly due to insufficient or noisy color-specific features or overfitting that reduces generalization. Conversely, `gender` prediction also appears better handled by ResNet18, indicating ResNet50’s capacity is unnecessary or even detrimental for this attribute.

3. **Given the above, we propose using ResNet18 for all attributes except `baseColour`.**
   Although `baseColour` shows relatively low F1 scores on both models, ResNet18 still outperforms ResNet50, so it remains the preferred backbone for color prediction as well.

4. **Overall, ResNet18 emerges as the more robust and generalizable backbone for this dataset and task setup,**
   while ResNet50 appears to overfit on multiple attributes, especially those with fewer or less complex classes.

---

## 🏛️ Architecture

Below is an overview of the app’s architecture.:

![App Architecture](app/architecture.jpg)

This diagram illustrates the flow from image upload in Streamlit to prediction engines (ResNet18 & ResNet50) and LLM-powered styling module.

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/aminnademi/Predict-Fashion-Product-Images.git
   cd Predict-Fashion-Product-Images
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. You can download or load the dataset from  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small and place images them in observed variables.

---

## 🚀 Running the App

1. Place model files (`.pth`) and `label2idx.json` alongside `app.py` .
2. Set your Groq API key (you can create it on https://console.groq.com/keys for free) in `app.py`:
   ```python
   API_KEY = "your_api_key"
   ```
3. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```
4. In the browser:
   - Upload a clothing image
   - View predicted attributes & confidences
   - Ask styling questions (powered by LLM)
   - Generate short fashion captions

---