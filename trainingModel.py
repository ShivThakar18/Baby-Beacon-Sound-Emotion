# IMPORT LIBRARIES --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np 
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import tensorflow as tf
import keras
from keras import models, layers

# CONSTANTS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DIR = "C:\\Users\\shivt\\Code\\BabyBeacon\\Baby-Beacon-Sound-Emotion\\"
AUDIO_PATH = DIR+"data\\dataset\\expanded_dataset\\"
EMOTIONS = ["belly_pain","burping","discomfort","hungry","tired"]
MODEL_PATH = DIR + "model\\"
SAMPLE_RATE = 22050     # standard sample rate for librosa
DURATION = 7            # audio files in the dataset are approx 7 seconds long
N_MFCC = 40             # number of Mel Frequency Cepstral Coefficients, 40 chosen for efficiency

# AUDIO PREPROCESSING -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr = SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

# LOAD DATASET ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
features, labels = [], [] 
for idx, emotion in enumerate(EMOTIONS):
    folder_path = os.path.join(AUDIO_PATH, emotion)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path,file)
        features.append(extract_features(file_path))
        labels.append(idx)

features = np.array(features)
labels = keras.utils.to_categorical(labels,num_classes=len(EMOTIONS))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# BUILD MODEL -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(N_MFCC,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN THE MODEL ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=760, batch_size=16)

# SAVE THE MODEL ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.save(MODEL_PATH+"\\expanded\\baby_emotions_model.keras")
model.save(MODEL_PATH+"\\expanded\\baby_emotions_model.h5")
model.save_weights(MODEL_PATH+"\\expanded\\baby_emotions.weights.h5")

# EVALUATE MODEL BASED ON TEST DATASET
loss, accuracy = model.evaluate(X_test,y_test)
print(f"Test: Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Generate a classification report
print(classification_report(y_test_classes, y_pred_classes, target_names=EMOTIONS))


