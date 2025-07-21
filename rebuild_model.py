import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization

# Rebuild the original model architecture
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load model weights instead of the full model
emotion_model = build_lstm_gru_model((260, 1), 9)  # Input shape matches your feature size
emotion_model.load_weights("models/lstm_gru_0.9740.h5")

# Resave the model in the correct format
emotion_model.save("models/lstm_gru_fixed.h5")
print("✅ Model successfully rebuilt and saved as 'lstm_gru_fixed.h5'")
