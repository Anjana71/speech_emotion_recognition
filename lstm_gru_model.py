import librosa
import soundfile as sf
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# Enable GPU and Mixed Precision Conditionally
if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs available.")
    except RuntimeError as e:
        print(e)
else:
    mixed_precision.set_global_policy('float32')
    print("No compatible GPU detected. Using float32 for CPU training.")

# Enhanced Feature Extraction with Additional Features
def extract_features(segment, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta_delta_mean = np.mean(delta_delta, axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr), axis=1)
    #zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment))
    #rmse = np.mean(librosa.feature.rms(y=segment))
    #spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr), axis=1)

    return np.hstack((mfcc_mean, delta_mean, delta_delta_mean, chroma, mel))

# Modified Model Architecture for Faster CPU Training
def build_lstm_gru_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Reduced LSTM and GRU units for faster CPU training
    x = LSTM(256, return_sequences=True)(inputs) 
    x = BatchNormalization()(x)

    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)  # Optimized for CPU
    x = LayerNormalization()(x)

    x = GRU(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = GRU(64)(x)  

    x = Dropout(0.5)(x)  # Reduced Dropout for CPU Efficiency
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Learning Rate Decay for faster convergence
    lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.85)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data Preparation with Label Mapping
def prepare_data(features_csv):
    df = pd.read_csv(features_csv)

    # Define the label mapping for consistent emotion labels
    label_mapping = {
        'calm': 'Calm', 'happy': 'Happy', 'sad': 'Sad', 'angry': 'Angry', 'fearful': 'Fear',
        'disgust': 'Disgust', 'surprised': 'Surprise', 'neutral': 'Neutral',
        'a': 'Angry', 'd': 'Disgust', 'f': 'Fear', 'h': 'Happy', 'n': 'Neutral',
        'sa': 'Sad', 'su': 'Surprise', 'W': 'Angry', 'L': 'Boredom',
        'E': 'Disgust', 'A': 'Fear', 'F': 'Happy', 'T': 'Sad', 'N': 'Neutral'
    }

    # Apply mapping
    df['label'] = df['label'].map(label_mapping)

    # Extract features and labels
    X = df.iloc[:, :-1].values
    y = df['label'].values

    # Encode labels into integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Force correct number of emotion labels
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes=num_classes)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape for model compatibility
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test, num_classes

# Example Usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_classes = prepare_data("extracted_features.csv")
    model = build_lstm_gru_model((X_train.shape[1], 1), num_classes)
    
    # Early stopping for stable convergence
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, min_delta=0.001)

# Removed ReduceLROnPlateau since ExponentialDecay handles learning rate reduction
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          epochs=30, batch_size=64,
          callbacks=[early_stopping])



    # Early stopping with reduced patience for quicker termination
    '''early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, min_delta=0.001)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=30, batch_size=64,  # Reduced epochs & increased batch size
              callbacks=[early_stopping, lr_scheduler])'''

    # Accuracy Evaluation
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test Loss: {loss:.4f}")

    model.save(f"lstm_gru_{accuracy:.4f}.h5")
    print(f"✅ Model saved as 'lstm_gru_{accuracy:.4f}.h5'")    