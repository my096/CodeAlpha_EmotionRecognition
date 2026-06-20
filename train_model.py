import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import joblib

DATASET_PATH = "dataset/"

emotions = []
features = []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        file_path = os.path.join(DATASET_PATH, file)

        audio, sample_rate = librosa.load(file_path, duration=3)

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(mfccs.T, axis=0)

        emotion = file.split("_")[0]

        features.append(mfccs_scaled)
        emotions.append(emotion)

X = np.array(features)
y = np.array(emotions)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, "label_encoder.pkl")

y_categorical = to_categorical(y_encoded)

X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(
    Conv1D(
        64,
        kernel_size=3,
        activation='relu',
        input_shape=(40, 1)
    )
)

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("emotion_model.h5")

print("Model saved successfully.")
