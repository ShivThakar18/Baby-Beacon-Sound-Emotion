import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras

DIR = "C:\\Users\\shivt\\Code\\AudioProcessing\\"

CRYING_FILE = DIR+"\\data\\crying"
LAUGHING_FILE = DIR+"\\data\\laughing"

H5_MODEL = DIR + "\\model\\Emotion_Voice_Detection_Model"
PB_MODEL = DIR + "\\model\\model"

def waveform_spectrogram_mfccs(audio_file, file,export_raw=True):
    # Convert MP3 to WAV if needed and load the audio
    if audio_file.endswith(".mp3"):
        audio = AudioSegment.from_file(audio_file, format="mp3")
        audio = audio.set_frame_rate(44100).set_channels(1)  # Mono, 44.1 kHz
        audio.export("temp.wav", format="wav")
        audio_file = "temp.wav"
    
    # Load audio data
    y, sr = librosa.load(audio_file, sr=None)  # y: raw audio, sr: sampling rate
    
    # Save raw data if needed
    if export_raw:
        np.savetxt(file+".txt", y, delimiter=",")
        print("Raw audio data exported to \'"+file+".txt\'")


    # compute MFCCs 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC Shape: ", mfccs.shape)


    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(mfccs.T).T
    
    max_len = 100  # Adjust based on your dataset
    if mfccs_normalized.shape[1] < max_len:
        pad_width = max_len - mfccs_normalized.shape[1]
        mfccs_padded = np.pad(mfccs_normalized, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs_padded = mfccs_normalized[:, :max_len]

    
    """ # Plot waveform
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.title("Waveform")
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    plt.title("Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis="time", y_axis="mel", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # plot mfccs
    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfccs, x_axis="time", sr=sr)
    plt.colorbar()
    plt.title("MFCC")

    plt.tight_layout()
    plt.show() """
    
    return y, sr, mfccs_padded # Return raw data and sample rate

def audioPreprocessing(filename):
    audio_file = filename+".wav"  # Replace with your file path
    print("Input File = "+audio_file)
    raw_data, sample_rate, mfccs_padded = waveform_spectrogram_mfccs(audio_file,filename)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Raw data (first 10 samples): {raw_data[:10]}")

    return raw_data, mfccs_padded

def reshapeModelInput(mfccs_padded,filename):
    model_input = np.expand_dims(mfccs_padded.T,axis=0)
    print("Model Input Shape: ", model_input.shape)

    np.save(filename+"_features.npy",model_input)
    print(f"Processed features saved to {filename+'_features.npy'}")

    modelPredictions(model_input)

def modelPredictions(model_input):

   """  model = keras.models.load_model(H5_MODEL+".h5")
  
    predictions = model.predict(model_input)
    print("Predictions:", predictions)  """ 
   
   model = keras.models.Sequential()
   model.load_weights(H5_MODEL+".h5")

def main():
    raw_crying, mfccs_crying = audioPreprocessing(CRYING_FILE)
   #raw_crying, mfccs_crying = audioPreprocessing(LAUGHING_FILE)

    reshapeModelInput(mfccs_crying,CRYING_FILE)



main()