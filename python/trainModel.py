# IMPORT LIBRARIES --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import models, layers, utils

# CONSTANTS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DIR = "C:/Users/shivt/Code/BabyBeacon/Baby-Beacon-Sound-Emotion/"
AUDIO_PATH = os.path.join(DIR, "data", "dataset", "expanded_dataset")
OUTPUT_PATH = os.path.join(DIR, "output")
MODEL_PATH = os.path.join(DIR, "model")
EMOTIONS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
SAMPLE_RATE = 22050
DURATION = 7
N_MFCC = 40

# FUNCTION TO LOAD TXT FILE -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def extract_txtFile(file_path):
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                value = float(line.strip())
                data_list.append(value)
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    return data_list

# LOAD DATASET ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
features, labels = [], []
for idx, emotion in enumerate(EMOTIONS):
    folder_path = os.path.join(OUTPUT_PATH, emotion)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features.append(extract_txtFile(file_path))
        labels.append(idx)

features = np.array(features)
labels = utils.to_categorical(labels, num_classes=len(EMOTIONS))

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

# SAVE THE MODEL (only .h5 format is supported in TF 2.4) -----------------------------------------------------------------------------------------------------------------------------------------------------
model.save(os.path.join(MODEL_PATH, "baby_emotions_model.h5"))  # Full model
model.save_weights(os.path.join(MODEL_PATH, "baby_emotions_model.weights.h5"))  # Just weights

# EVALUATE MODEL ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print(classification_report(y_test_classes, y_pred_classes, target_names=EMOTIONS))
