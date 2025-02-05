import tensorflow as tf
import keras
h5_model = "C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\python\\model.h5"

model = keras.models.load_model(h5_model)

pb = "model_saved"

tf.saved_model.save(model,pb)
print("saved")