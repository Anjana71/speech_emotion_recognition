import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

# Load the model with custom objects
model = tf.keras.models.load_model(
    "models/lstm_gru_0.9740.h5",
    custom_objects={"MultiHeadAttention": MultiHeadAttention}
)

def predict_emotion(audio_path):
    # Load and preprocess audio
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)

    # Extract features (must match training)
    def extract_features(segment, sr=16000, n_mfcc=40):
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        mel = librosa.feature.melspectrogram(y=segment, sr=sr)

        return np.hstack([
            np.mean(mfcc, axis=1), 
            np.mean(delta, axis=1), 
            np.mean(delta_delta, axis=1),
            np.mean(chroma, axis=1), 
            np.mean(mel, axis=1)
        ])

    features = extract_features(audio)
    features = np.expand_dims(features, axis=[0, -1])  # Match model input shape

    # Predict emotion
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions)

    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    return emotion_labels[predicted_label]

