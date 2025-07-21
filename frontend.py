import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import librosa
import asyncio
import nest_asyncio
import cv2
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import sounddevice as sd
import queue
import threading
import time
from utils.evalutio import get_model_metrics  # Import the evaluation function

# Apply nest_asyncio for async handling
nest_asyncio.apply()

# Load models
whisper_model = whisper.load_model("medium")
emotion_model = load_model("models/lstm_gru_fixed.h5")

# Updated Emotion labels mapping
EMOTION_LABELS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Disgust",
    6: "Surprise",
    7: "Boredom",
    8: "Excited"
}

# ---------------------- Display Model Metrics ----------------------
def display_model_metrics():
    model_path = "models/lstm_gru_0.9740.h5"
    features_csv = "data/features/extracted_features.csv"

    try:
        metrics = get_model_metrics(model_path, features_csv)
        st.markdown("## 📊 Model Performance Metrics")
        st.success(f"✅ **Accuracy:** {metrics['accuracy']}%")
        st.info(f"📏 **Precision:** {metrics['precision']}%")
        st.info(f"📋 **Recall:** {metrics['recall']}%")
        st.info(f"🎯 **F1 Score:** {metrics['f1_score']}%")
    except Exception as e:
        st.error(f"❌ Error loading model metrics: {e}")

# Add metrics section to the main interface
st.sidebar.markdown("### Model Information")
if st.sidebar.button("📊 Show Model Performance"):
    display_model_metrics()

# ---------------------- Audio Conversion ----------------------
def convert_audio(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
    except Exception as e:
        st.error(f"❌ Audio conversion failed: {e}")

# ---------------------- Feature Extraction ----------------------
def extract_features(segment, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta_delta_mean = np.mean(delta_delta, axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr), axis=1)

    return np.hstack((mfcc_mean, delta_mean, delta_delta_mean, chroma, mel))

# ---------------------- Emotion Prediction ----------------------
def predict_emotion(segment_features):
    features = np.expand_dims(segment_features, axis=(0, -1))
    prediction = emotion_model.predict(features)
    return EMOTION_LABELS[np.argmax(prediction)]

# ---------------------- Segment Audio ----------------------

def segment_audio(audio_path, sr=16000, segment_duration=3):
    audio, _ = librosa.load(audio_path, sr=sr)
    segments = []

    # ✅ Handle short audio files (less than 3 seconds)
    if len(audio) < sr * segment_duration:
        features = extract_features(audio)
        segments.append((0.0, predict_emotion(features)))
        return segments

    # ✅ Standard segmentation logic for longer audio
    step = int(sr * 2.5)  # 2.5-second overlap
    for start in range(0, len(audio) - int(sr * segment_duration), step):
        segment = audio[start : start + int(sr * segment_duration)]
        features = extract_features(segment)
        segments.append((start / sr, predict_emotion(features)))

    return segments


# ---------------------- Transcription and Prediction ----------------------
async def transcribe_and_predict(audio_path):
    temp_converted_path = "temp_converted.wav"
    convert_audio(audio_path, temp_converted_path)

    loop = asyncio.get_running_loop()

    with st.spinner("🔍 Whisper is transcribing... (This may take a while)"):
        result = await loop.run_in_executor(None, whisper_model.transcribe, temp_converted_path)

    transcription = result['text']
    emotion_segments = segment_audio(temp_converted_path)
    
    os.remove(temp_converted_path)  # Clean up temporary file

    return transcription, emotion_segments

# ---------------------- Sync Wrapper for Async Function ----------------------
def transcribe_and_predict_sync(audio_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(transcribe_and_predict(audio_path))

import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import time

# ✅ Real-Time Emotion Detection Class
class RealTimeEmotionDetection:
    def __init__(self):
        self.q = queue.Queue()
        self.running = False
        self.stream = None
        self.audio_buffer = np.array([])

        # ✅ Initialize session state variables
        if "latest_emotion" not in st.session_state:
            st.session_state.latest_emotion = "Waiting for prediction..."
        if "latest_confidence" not in st.session_state:
            st.session_state.latest_confidence = None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[ERROR] Audio Stream Status: {status}")
        self.q.put(indata.copy())

    def predict_real_time_emotion(self):
        while self.running:
            if not self.q.empty():
                audio_chunk = self.q.get().flatten()
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))

                if len(self.audio_buffer) >= 16000 * 3:
                    # ✅ Extract features from the recorded audio
                    features = extract_features(self.audio_buffer[:16000 * 3])

                    if features is None or len(features) == 0:
                        print("[WARNING] No valid features extracted. Skipping prediction.")
                        st.session_state.latest_emotion = "No speech detected"
                        st.session_state.latest_confidence = None
                    else:
                        # ✅ Ensure predict_emotion returns (emotion, confidence)
                        result = predict_emotion(features)
                        if isinstance(result, tuple) and len(result) == 2:
                            emotion, confidence = result
                        else:
                            emotion, confidence = result, None  # If confidence is not returned

                        st.session_state.latest_emotion = emotion
                        st.session_state.latest_confidence = confidence

                    # ✅ Maintain 2.5s overlap for better prediction continuity
                    self.audio_buffer = self.audio_buffer[int(16000 * 2.5):]

                time.sleep(0.5)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=16000,
                blocksize=16000
            )
            try:
                self.stream.start()
                threading.Thread(target=self.predict_real_time_emotion, daemon=True).start()
            except Exception as e:
                print(f"[ERROR] Failed to start audio stream: {e}")
                self.running = False

    def stop_detection(self):
        if self.stream and self.running:
            try:
                self.stream.stop()
            except Exception as e:
                print(f"[ERROR] Failed to stop audio stream: {e}")
        self.running = False


# ✅ Initialize Real-Time Emotion Detector
real_time_detector = RealTimeEmotionDetection()

# ---------------------- Streamlit UI ----------------------
st.title("🎯 Speech Emotion Recognition System")
st.write("### 🎙️ Real-Time Emotion Detection")

# ✅ Dynamic UI updates
emotion_display = st.empty()

# ✅ Function to update UI smoothly
def update_ui():
    while real_time_detector.running:
        # ✅ Prevent session state access error
        if "latest_confidence" not in st.session_state:
            st.session_state.latest_confidence = None
        if "latest_emotion" not in st.session_state:
            st.session_state.latest_emotion = "Waiting for prediction..."

        # ✅ Update UI
        if st.session_state.latest_confidence is not None:
            emotion_display.write(f"### Current Emotion: **{st.session_state.latest_emotion}** (Confidence: {st.session_state.latest_confidence:.2f})")
        else:
            emotion_display.write(f"### Current Emotion: **{st.session_state.latest_emotion}**")
        time.sleep(1)

# ✅ Start/Stop Buttons
if st.button("Start Real-Time Detection"):
    real_time_detector.start_detection()
    st.success("🎙️ Real-Time Emotion Detection Started")

    # ✅ Start UI Update Thread
    threading.Thread(target=update_ui, daemon=True).start()

if st.button("Stop Real-Time Detection"):
    real_time_detector.stop_detection()
    st.session_state.latest_emotion = "Detection Stopped"
    st.session_state.latest_confidence = None
    emotion_display.write("### Current Emotion: **Detection Stopped**")
    st.warning("🛑 Real-Time Emotion Detection Stopped")



# ---------------------- Uploading Audio/Video Files ----------------------
uploaded_file = st.file_uploader("Upload Audio/Video File", type=['wav', 'mp3', 'mp4', 'm4a'])

if uploaded_file is not None:
    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav" if uploaded_file.type.startswith('audio') else ".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if uploaded_file.type.startswith('video'):
        extracted_audio_path = "extracted_audio.wav"
        convert_audio(temp_path, extracted_audio_path)
        audio_path = extracted_audio_path
    else:
        audio_path = temp_path

    if uploaded_file.type.startswith('audio'):
        st.audio(audio_path)
    else:
        st.video(temp_path)

    with st.spinner("⏳ Processing... Please wait."):
        transcription, emotion_segments = transcribe_and_predict_sync(audio_path)

    st.success("✅ Processing complete!")

    # ---------------------- Display Results ----------------------
    st.write("### Transcription with Emotion Prediction")
    st.write(transcription)

    st.write("### Emotion Segments")
    for start_time, emotion in emotion_segments:
        st.write(f"⏱️ {start_time:.2f}s - Emotion: {emotion}")
