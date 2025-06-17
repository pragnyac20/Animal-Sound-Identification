from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('models/animal_sound_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Ensure the uploads folder exists
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Route to the home page where the user uploads a file
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Save the uploaded file
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Process the audio file
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Predict the animal sound
    prediction = model.predict(mfcc_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)

    # Print predicted animal sound for debugging
    print(f"Predicted animal sound: {predicted_label[0]}")  # Debugging line

    # Render the prediction page with the predicted animal sound
    return render_template('prediction.html', predicted_animal=predicted_label[0])

if __name__ == '__main__':
    app.run(debug=True)


