import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load("C:\\Users\\shivt\\Code\\BabyBeacon\\Baby-Beacon-Sound-Emotion\\data\\testing_data\\belly_pain.wav", sr=22050)
plt.plot(y)
plt.title("Waveform - Python Librosa")
plt.show()
