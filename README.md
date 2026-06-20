# 🫁 Automated Chest X-Ray Pneumonia Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Mansoorrr/chest-xray-pneumonia-detector)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Deep Learning Lab Project | BS Artificial Intelligence — Semester 6**  
> **University of Management and Technology (UMT), Lahore, Pakistan**

---

## 🌐 Live Demo

🔗 **[Try the App on Hugging Face](https://huggingface.co/spaces/Mansoorrr/chest-xray-pneumonia-detector)**

---

## 📌 Project Overview

Pneumonia kills over **2.5 million people annually** worldwide. In Pakistan and many
developing countries, access to trained radiologists is critically limited — especially
in rural hospitals.

This project builds a complete **AI-powered diagnostic assistance system** that:
- Automatically classifies chest X-rays as **NORMAL** or **PNEUMONIA**
- Provides **confidence scores** for every prediction
- Generates **Grad-CAM heatmaps** showing exactly where pneumonia is detected
- Produces professional **AI-generated clinical PDF reports**
- Deploys as a full **Streamlit web application**

---

## 📊 Model Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | Train Time |
|-------|----------|-----|-----------|--------|----------|------------|
| ⭐ ResNet50 | **91.67%** | **0.9698** | **92.46%** | **94.36%** | **93.40%** | 35 min |
| VGG16 | 90.54% | 0.9566 | 90.27% | 95.13% | 92.63% | 93 min |
| Custom CNN | 87.34% | 0.9356 | 88.40% | 91.79% | 90.06% | 322 min |
| MobileNetV2 | 84.78% | 0.9370 | 81.45% | 97.95% | 88.94% | 18 min |

**ResNet50** was selected as the deployment model with **91.67% accuracy** and **AUC of 0.9698**.

---

## ✨ Web Application Features

| Feature | Description |
|---------|-------------|
| 🏠 Home & Prediction | Upload X-ray, validate, predict, confidence gauge |
| 🖼️ Multi-Image Analysis | Batch upload, pie chart, CSV download |
| ⚖️ All Models Comparison | Compare all 4 models per image |
| 📊 Model Performance | Bar, radar, bubble, AUC charts |
| 📈 3D Analytics | 3D scatter, surface, heatmap, sunburst |
| 📄 Generate Report | Groq AI (LLaMA3-70B) 8-section clinical PDF |
| ℹ️ About | Project info, tech stack, disclaimer |

---

## 📁 Project Structure
chest-xray-pneumonia-detection/

│

├── 📓 notebooks/

│   ├── 01_EDA.ipynb                ← Exploratory Data Analysis

│   ├── 02_custom_cnn.ipynb         ← Custom CNN training

│   ├── 03_vgg16.ipynb              ← VGG16 training

│   ├── 04_resnet50.ipynb           ← ResNet50 training (BEST)

│   ├── 05_mobilenetv2.ipynb        ← MobileNetV2 training

│   └── 06_comparison.ipynb         ← All models comparison

│

├── 📊 outputs/

│   ├── resnet50_results.json       ← Best model results

│   ├── vgg16_results.json

│   ├── custom_cnn_results.json

│   ├── mobilenetv2_results.json

│   └── *.png                       ← Training curves, confusion matrices

│

├── 🤖 models/                      ← Trained model files (via Git LFS)

│   ├── resnet50.keras

│   ├── vgg16.keras

│   ├── custom_cnn.keras

│   └── mobilenetv2.keras

│

├── 🌐 app.py                       ← Streamlit web application

├── 📋 requirements.txt             ← Python dependencies

└── 📖 README.md


---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.x, Keras |
| Models | ResNet50, VGG16, MobileNetV2, Custom CNN |
| Explainability | Grad-CAM |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib |
| Image Processing | OpenCV, Pillow |
| AI Reports | Groq API (LLaMA3-70B), ReportLab |
| Deployment | Hugging Face Spaces |
| Version Control | Git, Git LFS |

---

## 📦 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle (Paul Mooney / NIH Clinical Center)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

📥 Download: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> ⚠️ Dataset is NOT included in this repository due to size (1GB+).
> Download from Kaggle and place in `data/chest_xray/` folder.

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/engrmansoor304/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download from Kaggle and place in:

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.x, Keras |
| Models | ResNet50, VGG16, MobileNetV2, Custom CNN |
| Explainability | Grad-CAM |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib |
| Image Processing | OpenCV, Pillow |
| AI Reports | Groq API (LLaMA3-70B), ReportLab |
| Deployment | Hugging Face Spaces |
| Version Control | Git, Git LFS |

---

## 📦 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle (Paul Mooney / NIH Clinical Center)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

📥 Download: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> ⚠️ Dataset is NOT included in this repository due to size (1GB+).
> Download from Kaggle and place in `data/chest_xray/` folder.

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/engrmansoor304/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Download from Kaggle and place in:

data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/

### 4. Train models (run notebooks in order)
01_EDA.ipynb → 02_custom_cnn.ipynb → 03_vgg16.ipynb

→ 04_resnet50.ipynb → 05_mobilenetv2.ipynb → 06_comparison.ipynb

### 5. Run the web app
```bash
streamlit run app.py
```

> 💡 Models are included via Git LFS — they will download automatically when you clone.

---

## 🔑 Environment Variables

Create a `.env` file or add to your system environment:

Get your free Groq API key at: [console.groq.com](https://console.groq.com)

---

## 🧠 Key Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance (73% Pneumonia) | Weighted loss functions |
| Overfitting on small dataset | Dropout, BatchNorm, EarlyStopping |
| Black-box AI | Grad-CAM visualization |
| Non-X-ray image uploads | 7-metric image validation algorithm |
| Slow CPU training | Transfer Learning (2-phase fine-tuning) |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.
It is **NOT** a substitute for professional medical diagnosis.
Always consult a qualified radiologist or physician.

---

## 👨‍💻 Author

**Mansoor Ali**
BS Artificial Intelligence — Semester 6
University of Management and Technology (UMT), Lahore, Pakistan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mansoor%20Ali-blue?logo=linkedin)](https://www.linkedin.com/in/YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-engrmansoor304-black?logo=github)](https://github.com/engrmansoor304)
[![HuggingFace](https://img.shields.io/badge/🤗-Mansoorrr-yellow)](https://huggingface.co/Mansoorrr)

---

## 📚 References

1. He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Simonyan, K. & Zisserman, A. (2015). Very Deep CNNs for Large-Scale Image Recognition. ICLR.
3. Howard, A. G. et al. (2017). MobileNets: Efficient CNNs for Mobile Vision. arXiv.
4. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
5. Mooney, P. (2018). Chest X-Ray Images (Pneumonia). Kaggle Dataset.
6. Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection. arXiv.
7. Chollet, F. (2021). Deep Learning with Python. Manning Publications.

---

*Built with ❤️ using TensorFlow · Deployed on Hugging Face · Explained with Grad-CAM*
