import librosa
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Path to the directory containing animal sound files
data_dir =r"C:\Users\ADMIN\project\data\animal_sounds"

# Arrays to hold features and labels
features = []
labels = []

# Extract features from all audio files in the directory
for subdir in os.listdir(data_dir):
    animal_dir = os.path.join(data_dir, subdir)
    if os.path.isdir(animal_dir):
        for file in os.listdir(animal_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(animal_dir, file)
                y, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_scaled = np.mean(mfcc.T, axis=0)
                features.append(mfcc_scaled)
                labels.append(subdir)  # Animal name is the folder name

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels (animal names)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train a classifier (e.g., Random Forest)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y_encoded)

# Save the trained model and label encoder
joblib.dump(model, 'models/animal_sound_model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("Model training complete and saved.")



