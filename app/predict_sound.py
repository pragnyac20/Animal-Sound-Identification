import joblib
from preprocess_audio import preprocess_audio

def predict_audio(file_path):
    # Load the model
    model = joblib.load("models/animal_sound_model.pkl")

    # Preprocess audio and predict
    features = preprocess_audio(file_path)
    if features is not None:
        prediction = model.predict([features])
        return prediction[0]
    else:
        return "Error: Could not process the file."


    
