import librosa
import numpy as np
import tensorflow as tf
import keras

# CONSTANTS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DIR = "C:\\Users\\shivt\\Documents\\Baby-Beacon-Sound-Emotion\\"
AUDIO_PATH = DIR+"data\\test_data\\"
MODEL_PATH = DIR + "model\\"
SAMPLE_RATE = 22050  # Sample rate for librosa
DURATION = 3         # Duration of the audio file (seconds)
N_MFCC = 40          # Number of MFCCs to extract
EMOTIONS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# LOAD KERAS MODEL --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = keras.models.load_model("C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\augmented_baby_emotions_model.keras") # ADD "augmented_" or "expanded_" to test different versions of the model

# EXRACT AUDIO FEATURES ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    

    features = np.array(np.mean(mfcc.T, axis=0))

    print(features)

    # Take the mean of the MFCCs along the time axis to get a fixed-length feature vector
    return features

# PREDICT EMOTION ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def predict_emotion(file_path):
    # Extract features from the audio file
    features = extract_features(file_path)
    
    # Reshape the features to match the input shape expected by the model
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    
    # Return the predicted emotion label
    return EMOTIONS[predicted_class[0]]

# TEST FILE ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def testFile(audio_file,emotions):
    # Predict the emotion
    predicted_emotion = predict_emotion(audio_file)
    print("Actual = Belly Pain")
    print(f"The predicted emotion is: {predicted_emotion}")

# PREDICT FROM TXT FILE
def predictFromFile(file_path):
    features = np.loadtxt(file_path, dtype=np.float64)
    print(features) 

    # Reshape the features to match the input shape expected by the model
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    
    # Return the predicted emotion label
    return EMOTIONS[predicted_class[0]]
# TEST TEXT FILE
def test_txt(audio_file,emotions):
    # Predict the emotion
    predicted_emotion = predictFromFile(audio_file)
    print("Actual = Belly Pain")
    print(f"The predicted emotion is: {predicted_emotion}")

# PASS AUDIO FILES --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" for emote in EMOTIONS:
    testFile(AUDIO_PATH+emote+".wav",emote) """

test_txt("C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\output\\mfcc_features.txt",EMOTIONS)
testFile("C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\data\\testing_data\\belly_pain.wav",EMOTIONS)
