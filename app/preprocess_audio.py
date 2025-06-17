import os
import librosa
import numpy as np
import pandas as pd
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

data_dir = 'data/animal_sounds'
features = []
labels = []

print("Processing animal_sounds...")
for subdir in os.listdir(data_dir):
    animal_dir = os.path.join(data_dir, subdir)
    if os.path.isdir(animal_dir):
        for file in os.listdir(animal_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(animal_dir, file)
                try:
                    y, sr = librosa.load(file_path, sr=16000, mono=True)  # Resample to 16 kHz, mono
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_scaled = np.mean(mfcc.T, axis=0)
                    features.append(mfcc_scaled)
                    labels.append(subdir)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Save extracted features to CSV
df = pd.DataFrame(features)
df['label'] = labels
df.to_csv('features.csv', index=False)

print("Feature extraction completed. Saved to features.csv.")


