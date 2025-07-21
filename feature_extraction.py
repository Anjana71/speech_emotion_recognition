import librosa
import soundfile as sf
import numpy as np
import os
import pandas as pd

# Audio Preprocessing Function
def preprocess_audio(audio_path, output_path, target_sr=16000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = librosa.util.normalize(audio)
        sf.write(output_path, audio, target_sr)
    except Exception as e:
        print(f"❌ Skipping {audio_path} due to error: {e}")


'''def preprocess_audio(audio_path, output_path, target_sr=16000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    audio, sr = librosa.load(audio_path, sr=None)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = librosa.util.normalize(audio)
    sf.write(output_path, audio, target_sr)'''

# Feature Extraction Function
'''def extract_features(audio_path, sr=16000, n_mfcc=13):
    audio, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    return np.hstack((np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(mel, axis=1)))'''

# Feature Extraction with Delta and Delta-Delta Features
def extract_features(audio_path, sr=16000, n_mfcc=40):
    audio, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta_delta_mean = np.mean(delta_delta, axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1)

    return np.hstack((mfcc_mean, delta_mean, delta_delta_mean, chroma, mel))


# Dataset Processing Functions
def process_audioWav(input_dir, output_dir):
    return [(os.path.join(root, file), file.split('_')[2]) 
            for root, _, files in os.walk(input_dir) for file in files if file.endswith('.wav')]

def process_RAVDESS(input_dir, output_dir):
    return [(os.path.join(root, file), file.split('-')[2]) 
            for root, _, files in os.walk(input_dir) for file in files if file.endswith('.wav')]

def process_EMO_DB(input_dir, output_dir):
    return [(os.path.join(root, file), file[5]) 
            for root, _, files in os.walk(input_dir) for file in files if file.endswith('.wav')]

'''def process_IEMOCAP(csv_file):
    df = pd.read_csv(csv_file)
    base_dir = "data/raw/IEMOCAP/"  # Add the correct base path
    return [(os.path.join(base_dir, row['path']), row['emotion']) 
            for _, row in df.iterrows() if row['emotion'] != 'xxx']'''

def process_SAVEE(input_dir):
    return [(os.path.join(input_dir, file), file[0]) for file in os.listdir(input_dir) if file.endswith('.wav')]

def process_TESS(input_dir):
    return [(os.path.join(root, file), root.split('_')[-1].lower()) 
            for root, _, files in os.walk(input_dir) for file in files if file.endswith('.wav')]

# Master Dataset Processor
# Master Dataset Processor
def process_dataset(output_csv):
    datasets = [
        process_audioWav("data/raw/audioWav", "data/processed/"),
        process_RAVDESS("data/raw/RAVDESS", "data/processed/"),
        process_EMO_DB("data/raw/EMO-DB/wav", "data/processed/"),
        #process_IEMOCAP("data/raw/IEMOCAP/iemocap_full_dataset.csv"),
        process_SAVEE("data/raw/SAVEE/ALL"),
        process_TESS("data/raw/TESS/TESS Toronto emotional speech set data")
    ]

    features, labels = [], []
    for dataset in datasets:
        for audio_path, label in dataset:
            processed_path = os.path.join("data/processed/", os.path.basename(audio_path))
            
            # Skip already processed files
            if os.path.exists(processed_path):
                print(f"Skipping {processed_path} (already processed)")
            else:
                preprocess_audio(audio_path, processed_path)
            
            features.append(extract_features(processed_path))
            labels.append(label)

    # Save features to CSV
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_csv, index=False)

# Example Usage
if __name__ == "__main__":
    process_dataset("data/features/extracted_features.csv")
