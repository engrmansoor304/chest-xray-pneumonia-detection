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