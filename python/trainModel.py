import numpy as np
import os
import librosa.display
import tensorflow as tf
from tensorflow.keras import models, layers, utils

# CONSTANTS -------------------------------------------------------------------
DIR = "C:/Users/shivt/Code/BabyBeacon/Baby-Beacon-Sound-Emotion/"
AUDIO_PATH = os.path.join(DIR, "data", "dataset", "expanded_dataset")
OUTPUT_PATH = os.path.join(DIR, "output")
MODEL_PATH = os.path.join(DIR, "model")
EMOTIONS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
SAMPLE_RATE = 22050
DURATION = 7
N_MFCC = 40

# FUNCTION TO LOAD FEATURE VECTOR FROM .TXT -----------------------------------
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

# LOAD DATASET ----------------------------------------------------------------
features, labels = [], []
for idx, emotion in enumerate(EMOTIONS):
    folder_path = os.path.join(OUTPUT_PATH, emotion)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features.append(extract_txtFile(file_path))
        labels.append(idx)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)
labels = utils.to_categorical(labels, num_classes=len(EMOTIONS))

# MANUAL SHUFFLE + SPLIT (80% train, 20% test) --------------------------------
combined = list(zip(features, labels))
np.random.shuffle(combined)
features[:], labels[:] = zip(*combined)

split_idx = int(0.8 * len(features))
X_train, y_train = np.array(features[:split_idx]), np.array(labels[:split_idx])
X_test, y_test = np.array(features[split_idx:]), np.array(labels[split_idx:])

# BUILD MODEL -----------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(N_MFCC,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN MODEL -----------------------------------------------------------------
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=760, batch_size=16)

# SAVE MODEL ------------------------------------------------------------------
model.save(os.path.join(MODEL_PATH, "baby_emotions_model.h5"))
model.save_weights(os.path.join(MODEL_PATH, "baby_emotions_model.weights.h5"))

# EVALUATE MODEL --------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# SIMPLE CLASSIFICATION REPORT -----------------------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

def simple_classification_report(y_true, y_pred, class_names):
    from collections import Counter
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    print(f"\nSimple Accuracy: {accuracy:.2f}\n")

    counts = Counter(y_pred)
    print("Predicted class distribution:")
    for i, label in enumerate(class_names):
        print(f"{label}: {counts[i]}")

simple_classification_report(y_test_classes, y_pred_classes, EMOTIONS)
