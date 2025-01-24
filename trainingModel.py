import matplotlib.pyplot as plt 
from scipy import signal 
from scipy.io import wavfile as wav 
import numpy as np 
from numpy.lib import stride_tricks
import torch 
import torchvision 
from torchvision import transforms
import cv2 
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import keras
from keras import models, layers
# CONSTANTS
DIR = "C:\\Users\\shivt\\Code\\BabyBeacon\\Baby-Beacon-Sound-Emotion\\"
AUDIO_PATH = DIR+"data\\dataset\\"
EMOTIONS = ["belly_pain","burping","discomfort","hungry","tired"]
MODEL_PATH = DIR + "model\\"
SAMPLE_RATE = 22050     # standard sample rate for librosa
DURATION = 7            # audio files in the dataset are approx 7 seconds long
N_MFCC = 40             # number of Mel Frequency Cepstral Coefficients, 40 chosen for efficiency

# AUDIO PREPROCESSING
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr = SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

# LOAD DATASET
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

# BUILD MODEL
# Build the model
model = models.Sequential([
    layers.Input(shape=(N_MFCC,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save the model
model.save(MODEL_PATH+"baby_emotions_model.keras")
model.save(MODEL_PATH+"baby_emotions_model.h5")
model.save_weights(MODEL_PATH+"baby_emotions.weights.h5")



