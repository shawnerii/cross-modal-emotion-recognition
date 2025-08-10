# Cross-Modal Emotion Recognition

This repository contains the implementation of deep learning models for **emotion recognition** using two benchmark datasets: **FER2013** (facial images) and **RAVDESS** (speech audio). The project compares the performance of **CNN**, **LSTM**, and **CNN-LSTM** (CLSTM) architectures across modalities and includes an interactive real-time inference interface built with **Gradio**.

---

## 📌 Features
- **Two datasets:** FER2013 (35k+ facial images), RAVDESS (1.4k+ speech clips)
- **Model architectures:** CNN, LSTM, and hybrid CNN-LSTM
- **Comprehensive evaluation:** Accuracy, F1-score (macro & weighted), confusion matrices
- **Cross-modality analysis:** Direct comparison of same architecture on audio vs. visual data
- **Real-time inference UI:** Upload images or audio for instant predictions

---

## 📊 Results Overview
| Dataset   | Model  | Accuracy | Macro F1 | Weighted F1 |
|-----------|--------|----------|----------|-------------|
| FER2013   | CNN    | 61%      | 0.55     | 0.62        |
| RAVDESS   | CNN    | 81%      | 0.80     | 0.80        |
| RAVDESS   | LSTM   | 80%      | 0.79     | 0.80        |
| RAVDESS   | CLSTM  | 79%      | 0.78     | 0.79        |

**Key insight:** Audio-based CNN models deliver ~30% higher class consistency than visual-based CNN models.

---

## 🛠 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cross-modal-emotion-recognition.git
cd cross-modal-emotion-recognition
```

## 🚀 Usage

1.	Train Models
	•	DL_FER.py → Train on FER2013 dataset
	•	DL_RAVDESS.py → Train on RAVDESS dataset
	
2.	Run Inference UI
    python app.py
