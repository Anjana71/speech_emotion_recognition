# 🎙️ Real-Time Speech Emotion Recognition (SER)

This project implements an advanced, real-time **Speech Emotion Recognition** system using deep learning. It combines **audio feature extraction**, a hybrid **LSTM-GRU** model for emotion classification, and **OpenAI Whisper** for speech-to-text transcription. The system is deployed via an interactive web interface using **Streamlit**.

---

## 🚀 Features

- 🎧 Real-time emotion detection from live or uploaded audio
- 🧠 Hybrid LSTM-GRU model with attention mechanism
- 🗣️ Whisper-powered speech transcription
- 📊 Performance metrics: Accuracy, Precision, Recall, F1-Score
- 📈 Emotion segmentation and visualization
- 🌐 Interactive and lightweight UI built with Streamlit

---

## 🧠 Emotion Classes

- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise
- Boredom
- Excited

---

## 🗂️ Project Structure

├── data/ # Audio datasets and extracted features
├── models/ # Trained model files (.h5)
├── Feature_extraction.py # Audio preprocessing and feature extraction
├── lstm_gru_model.py # Deep learning model (LSTM + GRU + Attention)
├── Rebuild_model.py # Load and rebuild trained model
├── frontend.py # Streamlit frontend with real-time inference
├── utils/ # Utility functions (e.g., segmentation, metrics)
├── requirements.txt # Required Python packages


## 📦 Datasets Used

- [RAVDESS](https://zenodo.org/record/1188976)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [TESS](https://dataverse.library.yorku.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF)
- [SAVEE](https://www.kaggle.com/datasets/barelydedicated/savee-dataset)
- [EMO-DB](http://emodb.bilderbar.info/start.html)

> All datasets are converted to 16kHz mono `.wav` format for consistency.


## 🛠️ Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
Install Requirements


pip install -r requirements.txt
Install FFmpeg
Make sure FFmpeg is installed and added to your system PATH for audio conversion:


▶️ How to Run

streamlit run frontend.py
Then open http://localhost:8501 in your browser.

You can:
Upload audio files (.wav, .mp3, etc.)
Record real-time audio via microphone
View predicted emotion and transcribed speech
Monitor live emotion segments

🏗️ Model Architecture
LSTM (256 units) for long-term temporal dependencies
GRU (128 + 64 units) for efficient short-term context
Multi-Head Attention for focus on emotional segments
Dropout, BatchNorm for regularization
Softmax output for 9-class classification

📊 Results
Metric	Score
Accuracy	97.4%
Precision	96.8%
Recall	97.1%
F1-Score	97.0%

Trained on combined features (MFCC + Delta + Delta Delta + Chroma + Mel)

🔮 Future Enhancements
Facial emotion recognition integration
Sentiment-aware response generation
Multilingual emotion support
Web/mobile deployment (Flask/FastAPI)
Transformer-based models (e.g., Wav2Vec2, BERT)


📌 Acknowledgements
Librosa
TensorFlow
OpenAI Whisper
Streamlit
