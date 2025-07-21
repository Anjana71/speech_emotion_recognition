import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Model Architecture
def build_lstm_gru_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(256, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = GRU(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = GRU(64)(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and Prepare Data
def prepare_data(features_csv):
    df = pd.read_csv(features_csv)

    # Label Mapping
    label_mapping = {
        'calm': 'Calm', 'happy': 'Happy', 'sad': 'Sad', 'angry': 'Angry', 'fearful': 'Fear',
        'disgust': 'Disgust', 'surprised': 'Surprise', 'neutral': 'Neutral',
        'a': 'Angry', 'd': 'Disgust', 'f': 'Fear', 'h': 'Happy', 'n': 'Neutral',
        'sa': 'Sad', 'su': 'Surprise', 'W': 'Angry', 'L': 'Boredom',
        'E': 'Disgust', 'A': 'Fear', 'F': 'Happy', 'T': 'Sad', 'N': 'Neutral'
    }
    df['label'] = df['label'].map(label_mapping)

    X = df.iloc[:, :-1].values
    y = df['label'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = to_categorical(y, num_classes=len(np.unique(y)))

    X = np.expand_dims(X, axis=-1)
    return X, y

# Evaluation Function
def get_model_metrics(model_path, features_csv):
    X, y = prepare_data(features_csv)

    model = build_lstm_gru_model((X.shape[1], 1), y.shape[1])
    model.load_weights(model_path)

    y_pred = np.argmax(model.predict(X), axis=1)
    y_true = np.argmax(y, axis=1)


    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted') * 100
    recall = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100

    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2)
    }
