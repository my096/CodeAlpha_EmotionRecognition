import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import tempfile

model = load_model("emotion_model.h5")
encoder = joblib.load("label_encoder.pkl")

st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    audio, sample_rate = librosa.load(temp_path, duration=3)

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfccs_scaled = np.mean(mfccs.T, axis=0)

    features = mfccs_scaled.reshape(1, 40, 1)

    prediction = model.predict(features)

    predicted_class = np.argmax(prediction)

    emotion = encoder.inverse_transform([predicted_class])[0]

    st.success(f"Predicted Emotion: {emotion}")
