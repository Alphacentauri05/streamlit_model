import streamlit as st
import librosa
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import gdown

# Model file path
MODEL_PATH = "./best_rf_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1xtdK73bVV2XOx9iXcVN2xKbwy82QeqNQ"  # Replace with your actual file ID

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model into memory
model = joblib.load(MODEL_PATH)

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    
    pitch_values = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean, pitch_std, pitch_range = np.mean(pitch_values), np.std(pitch_values), np.ptp(pitch_values)
    
    rms_energy = librosa.feature.rms(y=y).flatten()
    intensity_mean, intensity_std, intensity_range = np.mean(rms_energy), np.std(rms_energy), np.ptp(rms_energy)
    
    duration = librosa.get_duration(y=y, sr=sr)
    peaks = librosa.effects.split(y, top_db=30)
    speech_rate = len(peaks) / duration if duration > 0 else 0
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    feature_vector = np.hstack([
        pitch_mean, pitch_std, pitch_range,
        intensity_mean, intensity_std, intensity_range,
        speech_rate, spectral_centroid, spectral_rolloff, zcr,
        mfccs_mean
    ])
    
    return feature_vector

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

# Streamlit UI
st.title("🎭 Voice Emotion Detection")

# File Upload
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    emotion = predict_emotion(file_path)
    st.subheader(f"🎭 Predicted Emotion: **{emotion}**")
